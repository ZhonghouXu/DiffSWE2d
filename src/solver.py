from __future__ import annotations
import torch
import torch.nn as nn
from .boundary import add_ghost_cells
from .config import SWEConfig
from .friction import apply_roughness_length_friction, apply_manning_friction
from .io_netcdf import (
    ModelGrid,
    load_dem_and_roughness_to_grid,
    load_rainfall_to_grid,
)
from .io_ascii import ascii_to_tensor, parse_model_grid
#from .io_ascii import ModelGrid as AsciiModelGrid

from .reconstruction import bflood_cn_reconstruction
from .riemann import bflood_hllc_flux

# Map scalar outputs automatically without breaking the graph
import torch._dynamo
torch._dynamo.config.capture_scalar_outputs = True


class SWE2D(nn.Module):
    """Differentiable B-Flood-style 2-D SWE solver on the DEM grid."""

    def __init__(self, bed_reference, roughness_length, dx, dy,
                 config=None, train_dem=True, maximum_dem_correction=None,
                 valid_mask=None, x=None, y=None, rainfall=None):
        super().__init__()
        self.cfg = config or SWEConfig()
        self.dx, self.dy = float(dx), float(dy)
        if bed_reference.ndim == 2:
            bed_reference = bed_reference[None, None]
        if roughness_length.ndim == 2:
            roughness_length = roughness_length[None, None]
        if bed_reference.shape != roughness_length.shape:
            raise ValueError("DEM and roughness length must have identical shapes")

        self.register_buffer("bed_reference", bed_reference.clone())
        self.register_buffer("roughness_length", roughness_length.clone())
        self.register_buffer("valid_mask", (
            torch.ones_like(bed_reference, dtype=torch.bool)
            if valid_mask is None else valid_mask.reshape_as(bed_reference).bool()
        ))
        if x is not None: self.register_buffer("x", x.clone())
        if y is not None: self.register_buffer("y", y.clone())

        # Optimising a correction rather than overwriting the source DEM makes
        # priors and bounds straightforward. With no bound, correction is direct.
        self.maximum_dem_correction = maximum_dem_correction
        self.dem_correction_parameter = nn.Parameter(
            torch.zeros_like(bed_reference), requires_grad=train_dem
        )
        self.rainfall = rainfall  # RainfallForcing is an nn.Module and moves with .to().
        # --- DYNAMIC BOUNDARIES ---
        self.use_dynamic_bc = False
        self.bc_times = None
        self.bc_wls = None

        # --- ADD THESE FOR MICRO-STEP GAUGE TRACKING ---
        self.track_gauges = False
        self.gauge_indices = {}      # {name: (iy, ix)}
        self.bed_elevations = {}     # {name: z}        
        self.ts_time = []
        self.ts_dt = []
        self.ts_h = {}               # {name: [h1, h2, ...]}
        self.ts_z = {}               # {name: [z1, z2, ...]}

    @property
    def bed(self):
        """Current differentiable DEM used by reconstruction and source terms."""
        if self.maximum_dem_correction is None:
            correction = self.dem_correction_parameter
        else:
            correction = self.maximum_dem_correction * torch.tanh(
                self.dem_correction_parameter
            )
        return self.bed_reference + correction * self.valid_mask

    @classmethod
    def from_netcdf(cls, dem_path, model_grid, rainfall_path=None, 
                    dem_variable=None, roughness_path=None, roughness_variable=None,
                    rainfall_variable=None, dem_x="x", dem_y="y",
                    rain_x="x", rain_y="y", rain_time="time",
                    rainfall_units=None, dem_method="linear",default_roughness=None,
                    roughness_method="linear", dem_outside_domain="error",
                    config=None, dtype=torch.float64, device=None,
                    train_dem=True, maximum_dem_correction=None):
        """Build on a user-defined grid independent of both input rasters.

        `model_grid` may be a ModelGrid or a dictionary accepted by
        ModelGrid.from_bounds, e.g. {xmin,xmax,ymin,ymax,dx,dy}.
        DEM and log(z0) are remapped from their shared source file, while
        rainfall is independently remapped from its own source file.
        """
        cfg = config or SWEConfig()
        if isinstance(model_grid, dict):
            model_grid = ModelGrid.from_bounds(dtype=dtype, device=device, **model_grid)
        if not isinstance(model_grid, ModelGrid):
            raise TypeError("model_grid must be ModelGrid or a bounds dictionary")
        grid = load_dem_and_roughness_to_grid(
            dem_path, model_grid, dem_variable, roughness_path=roughness_path, roughness_variable=roughness_variable,
            x_name=dem_x, y_name=dem_y, dem_method=dem_method,
            roughness_method=roughness_method,
            outside_domain=dem_outside_domain, dtype=dtype, device=device,
        )
        
        # Only try to load the NetCDF rainfall if a path was actually provided
        if rainfall_path is not None:
            rainfall = load_rainfall_to_grid(
                rainfall_path, model_grid, variable=rainfall_variable,
                time_name=rain_time, x_name=rain_x, y_name=rain_y,
                units=rainfall_units,
                outside_domain=cfg.rainfall_outside_domain,
                expected_crs=grid.crs, dtype=dtype, device=device,
            )
        else:
            rainfall = None

        # --- ROUGHNESS ---
        if roughness_variable is not None:
            rough_tensor = grid.roughness_length
        elif default_roughness is not None:
            rough_tensor = torch.full_like(grid.bed, fill_value=float(default_roughness))
        else:
            # Fallback to 0.03 if absolutely nothing is specified
            rough_tensor = torch.full_like(grid.bed, fill_value=0.025)
        # --------------------------------

        return cls(grid.bed, rough_tensor, grid.dx, grid.dy,
            config=cfg, train_dem=train_dem,
            maximum_dem_correction=maximum_dem_correction,
            valid_mask=grid.valid_mask, x=grid.x, y=grid.y,
            rainfall=rainfall)

    @classmethod
    def from_ascii(
        cls,
        dem_path,
        *,
        model_grid,
        roughness_path=None,
        config=None,
        dem_method="linear",
        roughness_method="nearest",
        dem_outside_domain="error",
        roughness_outside_domain="error",
        fill_dem_nodata=True,
        fill_roughness_nodata=True,
        default_roughness=None,
        train_dem=False,
        maximum_dem_correction=None,
        device=None,
        dtype=torch.float32,
        **model_kwargs,
    ):
        """
        Construct an SWE2D model from ESRI ASCII rasters.

        Parameters
        ----------
        dem_path : str or Path
            ESRI ASCII file containing bed elevation.

        model_grid : ModelGrid or mapping
            Target computational grid. Dictionary form:

            {
                "xmin": float,
                "xmax": float,
                "ymin": float,
                "ymax": float,
                "resolution": float,
            }

            Bounds are interpreted as model-domain outer edges.

        roughness_path : str or Path, optional
            ESRI ASCII roughness raster. It is interpolated onto the same
            model grid as the DEM.

        config : SWEConfig, optional
            SWE2D configuration.

        dem_method : {"linear", "nearest"}
            DEM interpolation method.

        roughness_method : {"linear", "nearest"}
            Roughness interpolation method.

        dem_outside_domain : {"error", "nearest", "nan"}
            DEM behaviour outside the source ASCII raster.

        roughness_outside_domain : {"error", "nearest", "nan"}
            Roughness behaviour outside its source ASCII raster.

        fill_dem_nodata : bool
            Fill internal DEM NoData cells with nearest valid values.

        fill_roughness_nodata : bool
            Fill internal roughness NoData cells with nearest valid values.

        default_roughness : float, optional
            Uniform roughness value used if roughness_path is not supplied.

        train_dem : bool
            If True, make the model bed trainable.

            This assumes that ``model.bed`` is the tensor/parameter used by
            the solver. Adjust this block if the package uses another name.

        maximum_dem_correction : float, optional
            Metadata describing the maximum permitted DEM correction.
            The actual enforcement must exist in the SWE2D forward model or
            its bed-correction parameterisation.

        device : str or torch.device, optional
            Device on which tensors are initially created.

        dtype : torch.dtype
            Tensor dtype.

        **model_kwargs
            Additional arguments passed to the normal SWE2D constructor.
        """

        grid = parse_model_grid(model_grid)

        bed, grid_spec = ascii_to_tensor(
            dem_path,
            grid,
            method=dem_method,
            outside_domain=dem_outside_domain,
            fill_internal_nodata=fill_dem_nodata,
            device=device,
            dtype=dtype,
        )

        if not torch.isfinite(bed).all():
            raise ValueError(
                "The regridded bed contains NaN or infinite values."
            )

        # create the roughness
        if roughness_path is not None:
            roughness_length, _ = ascii_to_tensor(
                roughness_path,
                grid,
                method=roughness_method,
                outside_domain=roughness_outside_domain,
                fill_internal_nodata=fill_roughness_nodata,
                device=device,
                dtype=dtype,)
        elif default_roughness is not None:
            roughness_length = torch.full_like(bed, fill_value=float(default_roughness))
        else:
            # Provide a fallback if no roughness is specified at all
            roughness_length = torch.full_like(bed, fill_value=0.03)

        # Construct the normal SWE2D model.
        model = cls(
            bed_reference=bed,
            roughness_length=roughness_length,
            dx=grid_spec.resolution,
            dy=grid_spec.resolution,
            config=config,
            **model_kwargs,
        )

        # Retain the original library ModelGrid. This is preferable because
        # it preserves dtype, device, coordinates, and package-specific functionality
        model.model_grid = model_grid
        # Retain the CPU/NumPy-compatible grid used by io_ascii.py.
        model.grid_spec = grid_spec

        model.dem_path = str(dem_path)
        model.train_dem = bool(train_dem)
        model.maximum_dem_correction = maximum_dem_correction

        
        # Optional trainable bed.
        #
        # This assumes model.bed is the bed used by the SWE2D solver.
        # If bed is registered as a buffer internally, replacing it with a
        # Parameter may require removing the buffer first.
        if train_dem:
            if not hasattr(model, "bed"):
                raise AttributeError(
                    "train_dem=True was requested, but the SWE2D model does "
                    "not expose a 'bed' attribute. Update from_ascii() to use "
                    "the actual bed attribute employed by the solver."
                )

            current_bed = model.bed.detach().clone()

            # If bed is already registered as a buffer, remove it before
            # assigning a trainable Parameter.
            if hasattr(model, "_buffers") and "bed" in model._buffers:
                del model._buffers["bed"]

            model.bed = torch.nn.Parameter(
                current_bed,
                requires_grad=True,
            )

        model.train_dem = bool(train_dem)
        model.maximum_dem_correction = maximum_dem_correction

        # Store source metadata for reproducibility.
        model.dem_path = str(dem_path)

        model.roughness_path = (
            None
            if roughness_path is None
            else str(roughness_path)
        )

        model.dem_interpolation_method = dem_method
        model.roughness_interpolation_method = roughness_method
        model.dem_outside_domain = dem_outside_domain
        model.roughness_outside_domain = roughness_outside_domain

        return model

    @staticmethod
    def _expand(field, batch):
        return field if field.shape[0] == batch else field.expand(batch, -1, -1, -1)

    def spatial_operator(self, U, rainfall_rate=None, infiltration=None,
                         return_wave_speed=False):
        """Compute the conservative tendency and optionally CFL wave speeds."""
        c = self.cfg
        batch, _, ny, nx = U.shape
        bed = self._expand(self.bed, batch)
        Up, zp = add_ghost_cells(
            U, 
            bed, 
            ng=2, 
            boundary_left=c.boundary_left,
            boundary_right=c.boundary_right,
            boundary_top=c.boundary_top,
            boundary_bottom=c.boundary_bottom,
            constant_value=c.constant_value,
            constant_level=c.constant_level
        )
        args = (c.gravity, c.epsilon, c.dry_depth,
                c.limiter_theta, c.water_slope_reset)
        ULx, URx, sxL, sxR = bflood_cn_reconstruction(Up, zp, -1, *args)
        ULy, URy, syL, syR = bflood_cn_reconstruction(Up, zp, -2, *args)
        Fx, ax = bflood_hllc_flux(ULx, URx, "x", c.gravity, c.epsilon, c.dry_depth)
        Gy, ay = bflood_hllc_flux(ULy, URy, "y", c.gravity, c.epsilon, c.dry_depth)

        # A face has two momentum fluxes because its topographic correction is
        # asymmetric. Mass and transverse-momentum fluxes remain identical.
        Fx_for_left = Fx.clone(); Fx_for_right = Fx.clone()
        Gy_for_bottom = Gy.clone(); Gy_for_top = Gy.clone()
        Fx_for_left[:, 1:2] -= sxL; Fx_for_right[:, 1:2] -= sxR
        Gy_for_bottom[:, 2:3] -= syL; Gy_for_top[:, 2:3] -= syR

        rhs = (
            Fx_for_right[:, :, 2:2+ny, 1:1+nx]
            - Fx_for_left[:, :, 2:2+ny, 2:2+nx]
        ) / self.dx
        rhs += (
            Gy_for_top[:, :, 1:1+ny, 2:2+nx]
            - Gy_for_bottom[:, :, 2:2+ny, 2:2+nx]
        ) / self.dy

        h = U[:, 0:1]
        rain = torch.zeros_like(h) if rainfall_rate is None else torch.broadcast_to(rainfall_rate, h.shape)
        loss = torch.zeros_like(h) if infiltration is None else torch.broadcast_to(infiltration, h.shape)
        rhs = torch.cat((rhs[:, 0:1] + rain - torch.clamp(loss, min=0.0),
                         rhs[:, 1:2], rhs[:, 2:3]), dim=1)

        if return_wave_speed:
            physical_ax = ax[:, :, 2:2+ny, 1:2+nx].amax()
            physical_ay = ay[:, :, 1:2+ny, 2:2+nx].amax()
            return rhs, physical_ax, physical_ay
        return rhs

    def stable_dt(self, U):
        _, ax, ay = self.spatial_operator(U, return_wave_speed=True)
        return self.cfg.cfl / torch.clamp(
            ax / self.dx + ay / self.dy, min=self.cfg.epsilon
        )

    def _clean_and_limit(self, old, candidate):
        if self.cfg.positivity_limiter:
            delta = candidate - old
            dh = delta[:, 0:1]
            theta = torch.where(
                dh < 0.0,
                torch.clamp(old[:, 0:1] / torch.clamp(-dh, min=self.cfg.epsilon), max=1.0),
                torch.ones_like(dh),
            )
            candidate = old + theta * delta
        h = torch.clamp(candidate[:, 0:1], min=0.0)
        wet = h > self.cfg.dry_depth
        return torch.cat((h,
            torch.where(wet, candidate[:, 1:2], torch.zeros_like(h)),
            torch.where(wet, candidate[:, 2:3], torch.zeros_like(h))), dim=1)

    def step(self, U, model_time, max_dt, infiltration=None):
        """Midpoint predictor-corrector with stage-time rainfall."""
        # Unpack the time tensor safely for the rainfall interpolator
        t_float = model_time.item()
        rain_n = None if self.rainfall is None else self.rainfall.at(t_float)
        
        # Compute k1 AND wave speeds at the exact same time (saves an entire calculation!)
        k1, ax, ay = self.spatial_operator(U, rain_n, infiltration, return_wave_speed=True)
        
        # Calculate the stable dt using those speeds
        dt_stable = self.cfg.cfl / torch.clamp(ax / self.dx + ay / self.dy, min=self.cfg.epsilon)
        # Native PyTorch tensor minimum (max_dt is already a tensor now!)
        dt_tensor = torch.minimum(dt_stable, max_dt)
        dt_float = dt_tensor.item()

        rain_half = None if self.rainfall is None else self.rainfall.at(t_float + 0.5 * dt_float)
        
        # Proceed with the predictor
        predictor = self._clean_and_limit(U, U + 0.5 * dt_tensor * k1)
        k2 = self.spatial_operator(predictor, rain_half, infiltration)
        updated = self._clean_and_limit(U, U + dt_tensor * k2)

        if self.cfg.apply_friction:
            roughness_tensor = self._expand(self.roughness_length, U.shape[0])
            if self.cfg.frictionmodel == "manning":
                updated = apply_manning_friction(
                    updated, roughness_tensor, dt_tensor, 
                    dry_depth=self.cfg.dry_depth, epsilon=self.cfg.epsilon,
                )
            else:
                updated = apply_roughness_length_friction(
                    updated, self._expand(self.roughness_length, U.shape[0]), dt_tensor,
                    dry_depth=self.cfg.dry_depth, epsilon=self.cfg.epsilon,
                )
        # Return both the updated U and the dt_value that was used
        return updated, dt_float

    def forward(self, U0, t_end, start_time=0.0, max_steps=1e10,
                infiltration=None, callback=None):
        """Advance from start_time to start_time+t_end with adaptive CFL steps."""
        U = U0
        hmax = U[:, 0].clone()
        elapsed = 0.0
        for iteration in range(int(max_steps)):
            if elapsed >= t_end - 1.0e-14:
                return U, hmax            
            
            # -------- STEP BOUNDARY INTERPOLATION ---------
            current_time = start_time + elapsed
            if self.use_dynamic_bc:
                # Cast current_time to a tensor on the same device to keep math fast
                ct_tensor = torch.tensor([current_time], dtype=torch.float64, device=self.bc_times.device)

                # Find exactly where we are in the time series
                idx = torch.searchsorted(self.bc_times, ct_tensor).item()
                
                if idx == 0:
                    self.cfg.constant_level = self.bc_wls[0].item()
                elif idx == len(self.bc_times):
                    self.cfg.constant_level = self.bc_wls[-1].item()
                else:
                    # Linear interpolation on the GPU
                    t0, t1 = self.bc_times[idx-1], self.bc_times[idx]
                    w0, w1 = self.bc_wls[idx-1], self.bc_wls[idx]
                    weight = (ct_tensor[0] - t0) / (t1 - t0)
                    current_wl = w0 + weight * (w1 - w0)
                    self.cfg.constant_level = current_wl.item()
            # ----------------------------------------------

            # Calculate remaining time to cap dt to avoid overshoot time out of range
            max_dt = t_end - elapsed
            # --- Hide the changing floats from Dynamo by wrapping them in Tensors ---
            max_dt_t = torch.tensor(max_dt, dtype=U.dtype, device=U.device)
            time_t = torch.tensor(start_time + elapsed, dtype=U.dtype, device=U.device)
            
            U, dt_value = self.step(U, time_t, max_dt_t, infiltration)
            
            hmax = torch.maximum(hmax, U[:, 0])
            elapsed += dt_value

            # --- MICRO-STEP TRACKING FOR GAUGES---
            if self.track_gauges:
                self.ts_time.append(start_time + elapsed)
                self.ts_dt.append(dt_value)
                
                # Loop through the dictionary of gauges
                for name, (iy, ix) in self.gauge_indices.items():
                    h_point = U[0, 0, iy, ix].item()
                    self.ts_h[name].append(h_point)
                    self.ts_z[name].append(h_point + self.bed_elevations[name])
            # --------------------------------

            if callback is not None:
                callback(iteration, start_time + elapsed, U)
        raise RuntimeError("max_steps reached before t_end")

    def dem_regularization(self, prior_weight=1.0, slope_weight=0.0,
                           curvature_weight=0.0):
        """Convenience regularizer for DEM inversion."""
        correction = self.bed - self.bed_reference
        prior = correction.square().mean()
        dx = self.bed[..., 1:] - self.bed[..., :-1]
        dy = self.bed[..., 1:, :] - self.bed[..., :-1, :]
        slope = dx.square().mean() + dy.square().mean()
        dxx = dx[..., 1:] - dx[..., :-1]
        dyy = dy[..., 1:, :] - dy[..., :-1, :]
        curvature = dxx.square().mean() + dyy.square().mean()
        return prior_weight * prior + slope_weight * slope + curvature_weight * curvature
    
