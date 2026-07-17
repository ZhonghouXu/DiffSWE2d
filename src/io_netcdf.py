"""NetCDF readers and remapping to an independent custom model grid."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence
import numpy as np
import torch
import xarray as xr
from .forcing import RainfallForcing


@dataclass(frozen=True)
class ModelGrid:
    """Uniform Cartesian grid used by the numerical model.

    x and y are cell-centre coordinates. The model grid is independent of both
    the source DEM grid and the source rainfall grid.
    """
    x: torch.Tensor
    y: torch.Tensor
    dx: float
    dy: float

    @property
    def nx(self): return int(self.x.numel())
    @property
    def ny(self): return int(self.y.numel())

    @classmethod
    def from_cell_centres(cls, x, y, dtype=torch.float64, device=None):
        x = torch.as_tensor(x, dtype=dtype, device=device)
        y = torch.as_tensor(y, dtype=dtype, device=device)
        if x.ndim != 1 or y.ndim != 1 or x.numel() < 2 or y.numel() < 2:
            raise ValueError("x and y must be 1-D cell-centre arrays with at least two entries")
        dxs, dys = torch.diff(x), torch.diff(y)
        if not bool(torch.all(dxs > 0)) or not bool(torch.all(dys > 0)):
            raise ValueError("Model-grid coordinates must be strictly increasing")
        dx, dy = float(dxs.median()), float(dys.median())
        if not bool(torch.allclose(dxs, torch.full_like(dxs, dx), rtol=1e-5, atol=max(1e-10,dx*1e-8))):
            raise ValueError("The SWE solver requires a uniform model x grid")
        if not bool(torch.allclose(dys, torch.full_like(dys, dy), rtol=1e-5, atol=max(1e-10,dy*1e-8))):
            raise ValueError("The SWE solver requires a uniform model y grid")
        if not np.isclose(dx, dy, rtol=1e-8, atol=max(1e-10, dx*1e-10)):
            raise ValueError("The simplified model grid requires dx == dy")
        return cls(x=x, y=y, dx=dx, dy=dy)

    @classmethod
    def from_bounds(cls, xmin, xmax, ymin, ymax, resolution,
                    dtype=torch.float64, device=None):
        """Build a square-cell grid from outer-edge bounds and one resolution.

        The extent must be divisible by `resolution` within floating-point
        tolerance. This avoids silently changing the user-requested cell size.
        """
        if resolution <= 0:
            raise ValueError("resolution must be positive")
        nx_float=(xmax-xmin)/resolution; ny_float=(ymax-ymin)/resolution
        nx=int(round(nx_float)); ny=int(round(ny_float))
        if nx < 2 or ny < 2:
            raise ValueError("Use at least two cells in each direction")
        if not np.isclose(nx_float,nx,rtol=1e-10,atol=1e-10) or not np.isclose(ny_float,ny,rtol=1e-10,atol=1e-10):
            raise ValueError("Model bounds must be exactly divisible by resolution")
        x=xmin+(torch.arange(nx,dtype=dtype,device=device)+0.5)*resolution
        y=ymin+(torch.arange(ny,dtype=dtype,device=device)+0.5)*resolution
        return cls(x=x,y=y,dx=float(resolution),dy=float(resolution))


@dataclass(frozen=True)
class GridData:
    bed: torch.Tensor
    roughness_length: torch.Tensor
    x: torch.Tensor
    y: torch.Tensor
    dx: float
    dy: float
    valid_mask: torch.Tensor
    crs: Optional[str]


def _find_variable(ds, requested: Optional[str], candidates: Sequence[str]):
    if requested is not None:
        if requested not in ds.data_vars:
            raise KeyError(f"Variable {requested!r} not found; available={list(ds.data_vars)}")
        return requested
    lookup={n.lower():n for n in ds.data_vars}
    for candidate in candidates:
        if candidate.lower() in lookup: return lookup[candidate.lower()]
    raise KeyError(f"Could not infer variable from {candidates}; available={list(ds.data_vars)}")


def _ascending(da,x_name,y_name):
    da=da.transpose(y_name,x_name)
    if float(da[x_name][0])>float(da[x_name][-1]): da=da.isel({x_name:slice(None,None,-1)})
    if float(da[y_name][0])>float(da[y_name][-1]): da=da.isel({y_name:slice(None,None,-1)})
    return da


def _crs_text(ds,da):
    mapping=da.attrs.get("grid_mapping")
    if mapping and mapping in ds:
        attrs=ds[mapping].attrs
        return attrs.get("crs_wkt") or attrs.get("spatial_ref") or str(attrs)
    return ds.attrs.get("crs") or da.attrs.get("crs")


def _unit_factor(units):
    u=units.lower().replace(" ","").replace("**","^")
    factors={"m/s":1.,"ms-1":1.,"m.s-1":1.,"mm/s":1e-3,"mms-1":1e-3,
             "mm/h":1e-3/3600,"mm/hr":1e-3/3600,"mmh-1":1e-3/3600,
             "mmhour-1":1e-3/3600,"m/h":1/3600,"m/hr":1/3600,
             "mh-1":1/3600,"kgm-2s-1":1e-3}
    if u not in factors: raise ValueError(f"Unsupported rainfall units {units!r}")
    return factors[u]


def _time_seconds(da):
    v=np.asarray(da.values)
    if np.issubdtype(v.dtype,np.datetime64):
        return np.asarray((v-v[0])/np.timedelta64(1,"s"),dtype=np.float64)
    v=v.astype(np.float64); units=str(da.attrs.get("units","seconds")).lower(); factor=1.
    if units.startswith("minute"): factor=60.
    elif units.startswith("hour"): factor=3600.
    elif units.startswith("day"): factor=86400.
    return (v-v[0])*factor


def _remap(da, grid: ModelGrid, x_name, y_name, method, outside, field_name):
    """Interpolate one DataArray onto model-grid cell centres."""
    if method not in ("linear","nearest"): raise ValueError("method must be linear or nearest")
    if outside not in ("error","nearest","nan","zero"): raise ValueError("invalid outside policy")
    tx=np.asarray(grid.x.detach().cpu()); ty=np.asarray(grid.y.detach().cpu())
    sx=np.asarray(da[x_name].values); sy=np.asarray(da[y_name].values)
    is_out=tx.min()<sx.min() or tx.max()>sx.max() or ty.min()<sy.min() or ty.max()>sy.max()
    if is_out and outside=="error":
        raise ValueError(f"Model grid extends beyond {field_name} source coverage")
    out=da.interp({x_name:tx,y_name:ty},method=method)
    if outside=="nearest":
        nearest=da.interp({x_name:tx,y_name:ty},method="nearest",kwargs={"fill_value":"extrapolate"})
        out=out.fillna(nearest)
    elif outside=="zero": out=out.fillna(0.)
    return out


def load_dem_and_roughness_to_grid(path, model_grid: ModelGrid, dem_variable=None,
                                    roughness_path=None, roughness_variable=None, x_name="x", y_name="y",
                                    dem_method="linear", roughness_method="linear",
                                    outside_domain="error", dtype=torch.float64, device=None):
    """Remap source DEM and z0 from one NetCDF file to the custom model grid.

    z0 is interpolated in log space to guarantee positivity and better represent
    multiplicative roughness contrasts. The differentiable DEM parameter is
    created later on this model grid; file remapping itself is preprocessing.
    """
    with xr.open_dataset(Path(path),decode_cf=True,mask_and_scale=True) as ds:
        zn=_find_variable(ds,dem_variable,("zb","bed","bed_elevation","elevation","dem","z"))
        z=_ascending(ds[zn].squeeze(drop=True),x_name,y_name)
        
        if not np.isfinite(z.values).all():
            raise ValueError("Source DEM contains missing values")
            
        z_model=_remap(z,model_grid,x_name,y_name,dem_method,outside_domain,"DEM")
        z_values=np.asarray(z_model.values,dtype=np.float64)

        # Try to extract a roughness map from a file
        try:
            # Decide which file to look inside
            if roughness_path is not None:
                ds_r = xr.open_dataset(Path(roughness_path), decode_cf=True, mask_and_scale=True)
            else:
                ds_r = ds # Fallback to looking in the DEM file

            # Extract and align
            rn = _find_variable(ds_r, roughness_variable, ("roughness_length", "roughness", "z0", "zo", "n"))
            r = _ascending(ds_r[rn].squeeze(drop=True), x_name, y_name)
            
            if not np.isfinite(r.values).all() or np.any(r.values <= 0):
                raise ValueError("Source roughness length must be finite and positive")
                
            # Process the roughness map
            log_r = xr.apply_ufunc(np.log, r)
            log_r_model = _remap(log_r, model_grid, x_name, y_name, roughness_method, outside_domain, "roughness")
            r_model = np.exp(np.asarray(log_r_model.values, dtype=np.float64))
            
            # Clean up if we opened a separate file
            if roughness_path is not None:
                ds_r.close()
            
        except (KeyError, FileNotFoundError, OSError):
            # If no roughness file exists, or the variable isn't found, create a dummy map. 
            # solver.py will overwrite this later with default_roughness!
            r_model = np.ones_like(z_values)

        valid=np.isfinite(z_values)&np.isfinite(r_model)
        if not valid.all(): raise ValueError("Model grid contains cells not covered by DEM/roughness")
        crs=_crs_text(ds,z)

    return GridData(torch.as_tensor(z_values,dtype=dtype,device=device),
        torch.as_tensor(r_model,dtype=dtype,device=device),model_grid.x,model_grid.y,
        model_grid.dx,model_grid.dy,torch.as_tensor(valid,dtype=torch.bool,device=device),crs)


def _centres_to_edges(centres, name):
    """Infer cell edges from strictly increasing cell-centre coordinates."""
    c=np.asarray(centres,dtype=np.float64)
    if c.ndim != 1 or c.size < 2 or np.any(np.diff(c)<=0):
        raise ValueError(f"{name} must be strictly increasing 1-D cell centres")
    edges=np.empty(c.size+1,dtype=np.float64)
    edges[1:-1]=0.5*(c[:-1]+c[1:])
    edges[0]=c[0]-0.5*(c[1]-c[0])
    edges[-1]=c[-1]+0.5*(c[-1]-c[-2])
    return edges


def _overlap_matrix(source_edges, target_edges):
    """Return lengths of intersections between target and source cells."""
    left=np.maximum(target_edges[:-1,None],source_edges[None,:-1])
    right=np.minimum(target_edges[1:,None],source_edges[None,1:])
    return np.maximum(0.0,right-left)


def conservative_rectilinear_remap(values, source_x, source_y, target_grid,
                                    outside_domain="zero"):
    """Conservatively remap cell-average rainfall rates to square model cells.

    For each target cell T, R_T = sum_S(R_S * area(S intersect T))/area(T).
    Therefore sum(R*cell_area) is conserved over the geometrical overlap. If
    the target grid covers the full source domain, total precipitation volume
    is conserved to floating-point precision. Areas outside source coverage
    contribute zero; `outside_domain='error'` instead rejects partial coverage.
    """
    if outside_domain not in ("zero","error"):
        raise ValueError("Conservative rainfall supports outside_domain='zero' or 'error'")
    sx_edges=_centres_to_edges(source_x,"rainfall x")
    sy_edges=_centres_to_edges(source_y,"rainfall y")
    tx_edges=_centres_to_edges(np.asarray(target_grid.x.detach().cpu()),"model x")
    ty_edges=_centres_to_edges(np.asarray(target_grid.y.detach().cpu()),"model y")
    if outside_domain=="error":
        if tx_edges[0] < sx_edges[0] or tx_edges[-1] > sx_edges[-1] or ty_edges[0] < sy_edges[0] or ty_edges[-1] > sy_edges[-1]:
            raise ValueError("Model grid extends outside rainfall cell-edge coverage")
    Wx=_overlap_matrix(sx_edges,tx_edges)  # [target_x, source_x]
    Wy=_overlap_matrix(sy_edges,ty_edges)  # [target_y, source_y]
    target_area=np.diff(ty_edges)[:,None]*np.diff(tx_edges)[None,:]
    # values: [time, source_y, source_x]. Separable area-overlap integration.
    integrated=np.einsum("ai,tij,bj->tab",Wy,values,Wx,optimize=True)
    return integrated/target_area[None,:,:]


def load_rainfall_to_grid(path, model_grid: ModelGrid, variable=None,time_name="time",
                          x_name="x",y_name="y",units=None,
                          outside_domain="zero",expected_crs=None,
                          dtype=torch.float64,device=None):
    """Read and conservatively remap rainfall from its own NetCDF grid.

    Rainfall values are interpreted as cell-average intensities. Source and
    model coordinates are interpreted as cell centres. The model grid can have
    different dimensions and resolution, but both grids must be rectilinear in
    the same projected CRS.
    """
    with xr.open_dataset(Path(path),decode_cf=True,mask_and_scale=True) as ds:
        name=_find_variable(ds,variable,("rainfall","rain","precipitation_rate","precipitation"))
        rain=ds[name].transpose(time_name,y_name,x_name)
        if float(rain[x_name][0])>float(rain[x_name][-1]): rain=rain.isel({x_name:slice(None,None,-1)})
        if float(rain[y_name][0])>float(rain[y_name][-1]): rain=rain.isel({y_name:slice(None,None,-1)})
        rain_crs=_crs_text(ds,rain)
        if expected_crs and rain_crs and str(expected_crs)!=str(rain_crs):
            raise ValueError("Rainfall and DEM CRS metadata differ; reproject before loading")
        source_values=np.asarray(rain.values,dtype=np.float64)
        if not np.isfinite(source_values).all():
            raise ValueError("Source rainfall contains missing values")
        values=conservative_rectilinear_remap(
            source_values,np.asarray(rain[x_name].values),np.asarray(rain[y_name].values),
            model_grid,outside_domain=outside_domain)
        factor=_unit_factor(units or rain.attrs.get("units","")); times=_time_seconds(ds[time_name])
    return RainfallForcing(torch.as_tensor(times,dtype=dtype,device=device),
        torch.as_tensor(values*factor,dtype=dtype,device=device)[:,None])
