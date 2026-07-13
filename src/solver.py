from __future__ import annotations
import torch
import torch.nn as nn
from .boundary import add_ghost_cells
from .config import SWEConfig
from .friction import apply_roughness_length_friction
from .io_netcdf import (
    ModelGrid,
    load_dem_and_roughness_to_grid,
    load_rainfall_to_grid,
)
from .reconstruction import bflood_cn_reconstruction
from .riemann import bflood_hllc_flux


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
    def from_netcdf(cls, dem_path, rainfall_path, model_grid,
                    dem_variable=None, roughness_variable=None,
                    rainfall_variable=None, dem_x="x", dem_y="y",
                    rain_x="x", rain_y="y", rain_time="time",
                    rainfall_units=None, dem_method="linear",
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
            dem_path, model_grid, dem_variable, roughness_variable,
            x_name=dem_x, y_name=dem_y, dem_method=dem_method,
            roughness_method=roughness_method,
            outside_domain=dem_outside_domain, dtype=dtype, device=device,
        )
        rainfall = load_rainfall_to_grid(
            rainfall_path, model_grid, variable=rainfall_variable,
            time_name=rain_time, x_name=rain_x, y_name=rain_y,
            units=rainfall_units,
            outside_domain=cfg.rainfall_outside_domain,
            expected_crs=grid.crs, dtype=dtype, device=device,
        )
        return cls(grid.bed, grid.roughness_length, grid.dx, grid.dy,
            config=cfg, train_dem=train_dem,
            maximum_dem_correction=maximum_dem_correction,
            valid_mask=grid.valid_mask, x=grid.x, y=grid.y,
            rainfall=rainfall)

    @staticmethod
    def _expand(field, batch):
        return field if field.shape[0] == batch else field.expand(batch, -1, -1, -1)

    def spatial_operator(self, U, rainfall_rate=None, infiltration=None,
                         return_wave_speed=False):
        """Compute the conservative tendency and optionally CFL wave speeds."""
        c = self.cfg
        batch, _, ny, nx = U.shape
        bed = self._expand(self.bed, batch)
        Up, zp = add_ghost_cells(U, bed, ng=2, boundary=c.boundary)
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

    def step(self, U, dt, model_time, infiltration=None):
        """B-Flood midpoint predictor-corrector with stage-time rainfall."""
        dt = torch.as_tensor(dt, dtype=U.dtype, device=U.device)
        rain_n = None if self.rainfall is None else self.rainfall.at(model_time)
        rain_half = None if self.rainfall is None else self.rainfall.at(model_time + 0.5 * dt)

        k1 = self.spatial_operator(U, rain_n, infiltration)
        predictor = self._clean_and_limit(U, U + 0.5 * dt * k1)
        k2 = self.spatial_operator(predictor, rain_half, infiltration)
        updated = self._clean_and_limit(U, U + dt * k2)

        if self.cfg.apply_friction:
            updated = apply_roughness_length_friction(
                updated, self._expand(self.roughness_length, U.shape[0]), dt,
                dry_depth=self.cfg.dry_depth, epsilon=self.cfg.epsilon,
            )
        return updated

    def forward(self, U0, t_end, start_time=0.0, max_steps=100000,
                infiltration=None, callback=None):
        """Advance from start_time to start_time+t_end with adaptive CFL steps."""
        U = U0
        hmax = U[:, 0].clone()
        elapsed = 0.0
        for iteration in range(max_steps):
            if elapsed >= t_end - 1.0e-14:
                return U, hmax
            dt_value = min(float(self.stable_dt(U).detach()), t_end - elapsed)
            U = self.step(U, dt_value, start_time + elapsed, infiltration)
            hmax = torch.maximum(hmax, U[:, 0])
            elapsed += dt_value
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
