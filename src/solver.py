from __future__ import annotations

from numbers import Integral, Real
from typing import Optional, Union

import torch
import torch.nn as nn

from .boundary import add_ghost_cells
from .config import SWEConfig
from .friction import apply_manning_friction, apply_roughness_length_friction
from .io_ascii import ascii_to_tensor, parse_model_grid
from .io_netcdf import (
    ModelGrid,
    load_dem_and_roughness_to_grid,
    load_rainfall_to_grid,
)
from .reconstruction import bflood_cn_reconstruction
from .riemann import bflood_hllc_flux


ScalarOrTensor = Union[float, int, torch.Tensor]


# Reduces scalar-related graph breaks when torch.compile is enabled.
# Adaptive Python time stepping is still not fully compilable.
try:
    import torch._dynamo
    torch._dynamo.config.capture_scalar_outputs = True
except (ImportError, AttributeError):
    pass


class SWE2D(nn.Module):
    """
    Differentiable B-Flood-style 2-D shallow-water solver.

    State channels are [h, hu, hv].

    Notes
    -----
    ``forward(..., t_end=...)`` interprets ``t_end`` as the duration of
    that call, not an absolute ending time. To advance from 5.0 to 5.2 s:

        U, hmax = model(U, t_end=0.2, start_time=5.0)

    Do not use ``t_end=5.2`` unless integrating for another 5.2 seconds.
    """

    def __init__(
        self,
        bed_reference,
        roughness_length,
        dx,
        dy,
        config=None,
        train_dem=True,
        maximum_dem_correction=None,
        valid_mask=None,
        x=None,
        y=None,
        rainfall=None,
    ):
        super().__init__()
        self.cfg = config or SWEConfig()
        self.dx, self.dy = float(dx), float(dy)

        if self.dx <= 0.0 or self.dy <= 0.0:
            raise ValueError(f"dx and dy must be positive, got {self.dx}, {self.dy}.")

        if bed_reference.ndim == 2:
            bed_reference = bed_reference[None, None]
        if roughness_length.ndim == 2:
            roughness_length = roughness_length[None, None]

        if bed_reference.ndim != 4 or bed_reference.shape[1] != 1:
            raise ValueError(
                f"bed_reference must have shape [ny,nx] or [B,1,ny,nx], "
                f"got {tuple(bed_reference.shape)}."
            )
        if roughness_length.ndim != 4:
            raise ValueError(
                f"roughness_length must be 2-D or 4-D, "
                f"got {tuple(roughness_length.shape)}."
            )
        if bed_reference.shape != roughness_length.shape:
            raise ValueError(
                f"DEM and roughness shapes differ: {bed_reference.shape} "
                f"!= {roughness_length.shape}."
            )
        if not torch.isfinite(bed_reference).all():
            raise ValueError("bed_reference contains non-finite values.")
        if not torch.isfinite(roughness_length).all():
            raise ValueError("roughness_length contains non-finite values.")

        self.register_buffer("bed_reference", bed_reference.clone())
        self.register_buffer("roughness_length", roughness_length.clone())

        mask = (
            torch.ones_like(bed_reference, dtype=torch.bool)
            if valid_mask is None
            else valid_mask.reshape_as(bed_reference).bool()
        )
        self.register_buffer("valid_mask", mask)

        if x is not None:
            self.register_buffer("x", x.clone())
        if y is not None:
            self.register_buffer("y", y.clone())

        self.maximum_dem_correction = maximum_dem_correction
        self.dem_correction_parameter = nn.Parameter(
            torch.zeros_like(bed_reference), requires_grad=bool(train_dem)
        )
        self.train_dem = bool(train_dem)
        self.rainfall = rainfall

        # Dynamic boundary forcing.
        self.use_dynamic_bc = False
        self.bc_times = None
        self.bc_wls = None

        # Optional micro-step gauge tracking.
        self.track_gauges = False
        self.gauge_indices = {}
        self.bed_elevations = {}
        self.ts_time, self.ts_dt = [], []
        self.ts_h, self.ts_z = {}, {}

    # =================================================================
    # Properties and constructors
    # =================================================================

    @property
    def bed(self):
        """
        Current differentiable bed.

        Inversion acts on a correction parameter rather than replacing
        the original ``bed_reference`` tensor.
        """
        correction = self.dem_correction_parameter
        if self.maximum_dem_correction is not None:
            correction = float(self.maximum_dem_correction) * torch.tanh(correction)
        return self.bed_reference + correction * self.valid_mask.to(
            self.bed_reference.dtype
        )

    @classmethod
    def from_netcdf(
        cls,
        dem_path,
        model_grid,
        rainfall_path=None,
        dem_variable=None,
        roughness_path=None,
        roughness_variable=None,
        rainfall_variable=None,
        dem_x="x",
        dem_y="y",
        rain_x="x",
        rain_y="y",
        rain_time="time",
        rainfall_units=None,
        dem_method="linear",
        default_roughness=None,
        roughness_method="linear",
        dem_outside_domain="error",
        config=None,
        dtype=torch.float64,
        device=None,
        train_dem=True,
        maximum_dem_correction=None,
    ):
        """Construct a model on a user-defined grid from NetCDF data."""
        cfg = config or SWEConfig()

        if isinstance(model_grid, dict):
            model_grid = ModelGrid.from_bounds(
                dtype=dtype, device=device, **model_grid
            )
        if not isinstance(model_grid, ModelGrid):
            raise TypeError("model_grid must be a ModelGrid or bounds dictionary.")

        grid = load_dem_and_roughness_to_grid(
            dem_path,
            model_grid,
            dem_variable,
            roughness_path=roughness_path,
            roughness_variable=roughness_variable,
            x_name=dem_x,
            y_name=dem_y,
            dem_method=dem_method,
            roughness_method=roughness_method,
            outside_domain=dem_outside_domain,
            dtype=dtype,
            device=device,
        )

        rainfall = None
        if rainfall_path is not None:
            rainfall = load_rainfall_to_grid(
                rainfall_path,
                model_grid,
                variable=rainfall_variable,
                time_name=rain_time,
                x_name=rain_x,
                y_name=rain_y,
                units=rainfall_units,
                outside_domain=cfg.rainfall_outside_domain,
                expected_crs=grid.crs,
                dtype=dtype,
                device=device,
            )

        if roughness_variable is not None:
            roughness = grid.roughness_length
        else:
            roughness = torch.full_like(
                grid.bed,
                float(default_roughness) if default_roughness is not None else 0.025,
            )

        model = cls(
            grid.bed,
            roughness,
            grid.dx,
            grid.dy,
            config=cfg,
            train_dem=train_dem,
            maximum_dem_correction=maximum_dem_correction,
            valid_mask=grid.valid_mask,
            x=grid.x,
            y=grid.y,
            rainfall=rainfall,
        )
        model.model_grid = model_grid
        model.dem_path = str(dem_path)
        model.roughness_path = (
            None if roughness_path is None else str(roughness_path)
        )
        return model

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
        Construct a model from ESRI ASCII rasters.

        ``dem_correction_parameter`` is trainable when ``train_dem=True``;
        ``model.bed`` is not replaced by another Parameter.
        """
        grid_spec = parse_model_grid(model_grid)
        bed, returned_grid = ascii_to_tensor(
            dem_path,
            grid_spec,
            method=dem_method,
            outside_domain=dem_outside_domain,
            fill_internal_nodata=fill_dem_nodata,
            device=device,
            dtype=dtype,
        )
        if not torch.isfinite(bed).all():
            raise ValueError("The regridded bed contains non-finite values.")

        if roughness_path is not None:
            roughness, _ = ascii_to_tensor(
                roughness_path,
                grid_spec,
                method=roughness_method,
                outside_domain=roughness_outside_domain,
                fill_internal_nodata=fill_roughness_nodata,
                device=device,
                dtype=dtype,
            )
        else:
            value = float(default_roughness) if default_roughness is not None else 0.03
            roughness = torch.full_like(bed, value)

        if not torch.isfinite(roughness).all():
            raise ValueError("The regridded roughness contains non-finite values.")

        model = cls(
            bed,
            roughness,
            returned_grid.resolution,
            returned_grid.resolution,
            config=config,
            train_dem=train_dem,
            maximum_dem_correction=maximum_dem_correction,
            **model_kwargs,
        )
        model.model_grid, model.grid_spec = model_grid, returned_grid
        model.dem_correction_parameter.requires_grad_(bool(train_dem))
        model.train_dem = bool(train_dem)

        model.dem_path = str(dem_path)
        model.roughness_path = (
            None if roughness_path is None else str(roughness_path)
        )
        model.dem_interpolation_method = dem_method
        model.roughness_interpolation_method = roughness_method
        model.dem_outside_domain = dem_outside_domain
        model.roughness_outside_domain = roughness_outside_domain
        return model

    # =================================================================
    # Configuration helpers
    # =================================================================

    def set_dynamic_boundary(
        self,
        times: torch.Tensor,
        water_levels: torch.Tensor,
        enabled: bool = True,
    ) -> None:
        """
        Configure a differentiable time-varying boundary.

        Water levels are stored without detaching. Interval selection is
        discrete, but interpolation remains differentiable with respect
        to the water-level values.
        """
        if not torch.is_tensor(times) or not torch.is_tensor(water_levels):
            raise TypeError("times and water_levels must be tensors.")
        if times.ndim != 1 or water_levels.ndim != 1:
            raise ValueError("times and water_levels must be one-dimensional.")
        if times.numel() == 0 or times.numel() != water_levels.numel():
            raise ValueError("Boundary arrays must be non-empty and equal length.")
        if times.numel() > 1 and not torch.all(times[1:] > times[:-1]):
            raise ValueError("Dynamic boundary times must be strictly increasing.")
        if not torch.isfinite(times).all() or not torch.isfinite(water_levels).all():
            raise ValueError("Dynamic boundary arrays contain non-finite values.")

        self.bc_times, self.bc_wls = times, water_levels
        self.use_dynamic_bc = bool(enabled)

    def clear_dynamic_boundary(self) -> None:
        """Disable and remove dynamic boundary forcing."""
        self.use_dynamic_bc = False
        self.bc_times = self.bc_wls = None

    def reset_gauge_tracking(self) -> None:
        """Clear micro-step gauge records."""
        self.ts_time, self.ts_dt = [], []
        self.ts_h = {name: [] for name in self.gauge_indices}
        self.ts_z = {name: [] for name in self.gauge_indices}

    @staticmethod
    def _expand(field: torch.Tensor, batch: int) -> torch.Tensor:
        """Expand a one-member field over a model-state batch."""
        if field.shape[0] == batch:
            return field
        if field.shape[0] != 1:
            raise ValueError(
                f"Field batch must be 1 or {batch}, got {field.shape[0]}."
            )
        return field.expand(batch, -1, -1, -1)

    @staticmethod
    def _prepare_boundary_level(
        level: ScalarOrTensor, reference: torch.Tensor
    ) -> torch.Tensor:
        """
        Move a boundary value to the model dtype/device.

        ``Tensor.to`` preserves an existing autograd connection.
        """
        if torch.is_tensor(level):
            return level.to(dtype=reference.dtype, device=reference.device)
        return reference.new_tensor(level)

    def _interpolate_dynamic_boundary(
        self, current_time: float, reference: torch.Tensor
    ) -> torch.Tensor:
        """Interpolate the dynamic BC while preserving BC-value gradients."""
        if self.bc_times is None or self.bc_wls is None:
            raise RuntimeError("Dynamic boundary forcing is not configured.")
        if self.bc_times.ndim != 1 or self.bc_wls.ndim != 1:
            raise ValueError("bc_times and bc_wls must be one-dimensional.")
        if self.bc_times.numel() == 0 or self.bc_times.numel() != self.bc_wls.numel():
            raise ValueError("Dynamic boundary arrays are empty or unequal.")

        times = self.bc_times.to(dtype=reference.dtype, device=reference.device)
        levels = self.bc_wls.to(dtype=reference.dtype, device=reference.device)
        time = reference.new_tensor(current_time)

        # The index is discrete; boundary-time coordinates are not optimized.
        index = torch.searchsorted(times, time, right=False)
        index = torch.clamp(index, 1, times.numel() - 1)

        t0, t1 = times[index - 1], times[index]
        w0, w1 = levels[index - 1], levels[index]
        weight = (time - t0) / torch.clamp(
            t1 - t0, min=torch.finfo(times.dtype).eps
        )
        return w0 + weight * (w1 - w0)

    def _resolve_boundary_level(
        self,
        current_time: float,
        U: torch.Tensor,
        explicit_boundary_level: Optional[ScalarOrTensor],
    ) -> torch.Tensor:
        """
        Resolve BC priority: explicit value, dynamic series, config value.
        """
        if explicit_boundary_level is not None:
            return self._prepare_boundary_level(explicit_boundary_level, U)
        if self.use_dynamic_bc:
            return self._interpolate_dynamic_boundary(current_time, U)
        return self._prepare_boundary_level(self.cfg.constant_level, U)

    # =================================================================
    # Spatial operator
    # =================================================================

    @staticmethod
    def _forcing_tensor(value, h: torch.Tensor) -> torch.Tensor:
        """Convert and broadcast rainfall or infiltration to depth shape."""
        if value is None:
            return torch.zeros_like(h)
        value = (
            value.to(dtype=h.dtype, device=h.device)
            if torch.is_tensor(value)
            else h.new_tensor(value)
        )
        return torch.broadcast_to(value, h.shape)

    def spatial_operator(
        self,
        U,
        rainfall_rate=None,
        infiltration=None,
        return_wave_speed=False,
        boundary_level=None,
    ):
        """
        Calculate the conservative SWE tendency.

        ``boundary_level`` may be a differentiable scalar tensor. If it is
        omitted, ``cfg.constant_level`` is used.
        """
        if U.ndim != 4 or U.shape[1] != 3:
            raise ValueError(
                f"U must have shape [batch,3,ny,nx], got {tuple(U.shape)}."
            )

        cfg = self.cfg
        batch, _, ny, nx = U.shape
        bed = self._expand(self.bed, batch)
        level = self._prepare_boundary_level(
            cfg.constant_level if boundary_level is None else boundary_level, U
        )

        Up, zp = add_ghost_cells(
            U,
            bed,
            ng=2,
            boundary_left=cfg.boundary_left,
            boundary_right=cfg.boundary_right,
            boundary_top=cfg.boundary_top,
            boundary_bottom=cfg.boundary_bottom,
            constant_value=cfg.constant_value,
            constant_level=level,
        )

        args = (
            cfg.gravity,
            cfg.epsilon,
            cfg.dry_depth,
            cfg.limiter_theta,
            cfg.water_slope_reset,
        )
        ULx, URx, sxL, sxR = bflood_cn_reconstruction(Up, zp, -1, *args)
        ULy, URy, syL, syR = bflood_cn_reconstruction(Up, zp, -2, *args)

        Fx, ax = bflood_hllc_flux(
            ULx, URx, "x", cfg.gravity, cfg.epsilon, cfg.dry_depth
        )
        Gy, ay = bflood_hllc_flux(
            ULy, URy, "y", cfg.gravity, cfg.epsilon, cfg.dry_depth
        )

        # Topographic corrections are asymmetric across each face.
        Fx_left, Fx_right = Fx.clone(), Fx.clone()
        Gy_bottom, Gy_top = Gy.clone(), Gy.clone()
        Fx_left[:, 1:2] -= sxL
        Fx_right[:, 1:2] -= sxR
        Gy_bottom[:, 2:3] -= syL
        Gy_top[:, 2:3] -= syR

        rhs = (
            Fx_right[:, :, 2:2 + ny, 1:1 + nx]
            - Fx_left[:, :, 2:2 + ny, 2:2 + nx]
        ) / self.dx
        rhs += (
            Gy_top[:, :, 1:1 + ny, 2:2 + nx]
            - Gy_bottom[:, :, 2:2 + ny, 2:2 + nx]
        ) / self.dy

        h = U[:, 0:1]
        rain = self._forcing_tensor(rainfall_rate, h)
        loss = torch.clamp(self._forcing_tensor(infiltration, h), min=0.0)
        rhs = torch.cat(
            (rhs[:, 0:1] + rain - loss, rhs[:, 1:2], rhs[:, 2:3]), dim=1
        )

        if return_wave_speed:
            ax = ax[:, :, 2:2 + ny, 1:2 + nx].amax()
            ay = ay[:, :, 1:2 + ny, 2:2 + nx].amax()
            return rhs, ax, ay
        return rhs

    # =================================================================
    # Time stepping
    # =================================================================

    def stable_dt(self, U, boundary_level=None):
        """Calculate the CFL-stable timestep."""
        _, ax, ay = self.spatial_operator(
            U, return_wave_speed=True, boundary_level=boundary_level
        )
        denominator = torch.clamp(
            ax / self.dx + ay / self.dy, min=self.cfg.epsilon
        )
        return self.cfg.cfl / denominator

    def _clean_and_limit(self, old, candidate):
        """Apply positivity preservation and dry-cell momentum removal."""
        if self.cfg.positivity_limiter:
            delta = candidate - old
            dh = delta[:, 0:1]
            safe_change = torch.clamp(-dh, min=self.cfg.epsilon)
            theta = torch.where(
                dh < 0.0,
                torch.clamp(old[:, 0:1] / safe_change, max=1.0),
                torch.ones_like(dh),
            )
            candidate = old + theta * delta

        h = torch.clamp(candidate[:, 0:1], min=0.0)
        wet = h > self.cfg.dry_depth
        zero = torch.zeros_like(h)
        return torch.cat(
            (
                h,
                torch.where(wet, candidate[:, 1:2], zero),
                torch.where(wet, candidate[:, 2:3], zero),
            ),
            dim=1,
        )

    def step(
        self,
        U,
        model_time,
        max_dt,
        infiltration=None,
        boundary_level=None,
    ):
        """
        Advance one adaptive midpoint predictor-corrector step.

        Notes
        -----
        The CFL timestep is intentionally detached. It is treated as a
        numerical control-flow decision rather than an inversion variable.

        The state update remains differentiable with respect to U and the
        boundary value for the selected timestep sequence, but autograd does
        not differentiate through changes in the adaptive time grid.
        """
        model_time = (
            model_time.to(dtype=U.dtype, device=U.device)
            if torch.is_tensor(model_time)
            else U.new_tensor(model_time)
        )
        max_dt = (
            max_dt.to(dtype=U.dtype, device=U.device)
            if torch.is_tensor(max_dt)
            else U.new_tensor(max_dt)
        )
        level = self._prepare_boundary_level(
            self.cfg.constant_level if boundary_level is None else boundary_level,
            U,
        )

        # Time is not an inversion parameter.
        time_float = float(model_time.detach().cpu())
        rain_start = (
            None if self.rainfall is None else self.rainfall.at(time_float)
        )

        k1, ax, ay = self.spatial_operator(
            U,
            rainfall_rate=rain_start,
            infiltration=infiltration,
            return_wave_speed=True,
            boundary_level=level,
        )
        denominator = torch.clamp(
            ax / self.dx + ay / self.dy, min=self.cfg.epsilon
        )
        stable_dt = self.cfg.cfl / denominator

        # Detach adaptive CFL selection to stabilize inverse BPTT.
        dt = torch.minimum(stable_dt, max_dt).detach()
        if not torch.isfinite(dt):
            raise FloatingPointError("The calculated timestep is non-finite.")

        dt_float = float(dt.cpu())
        if dt_float <= 0.0:
            raise FloatingPointError(f"Non-positive timestep: {dt_float}.")

        half_time = time_float + 0.5 * dt_float
        rain_half = (
            None if self.rainfall is None else self.rainfall.at(half_time)
        )

        predictor = self._clean_and_limit(U, U + 0.5 * dt * k1)
        k2 = self.spatial_operator(
            predictor,
            rainfall_rate=rain_half,
            infiltration=infiltration,
            boundary_level=level,
        )
        updated = self._clean_and_limit(U, U + dt * k2)

        if self.cfg.apply_friction:
            roughness = self._expand(self.roughness_length, U.shape[0])
            friction_args = dict(
                dry_depth=self.cfg.dry_depth,
                epsilon=self.cfg.epsilon,
            )
            if self.cfg.frictionmodel == "manning":
                updated = apply_manning_friction(
                    updated, roughness, dt, **friction_args
                )
            else:
                updated = apply_roughness_length_friction(
                    updated, roughness, dt, **friction_args
                )

        return updated, dt_float

    # =================================================================
    # Forward integration
    # =================================================================

    @staticmethod
    def _scalar_float(value, name: str) -> float:
        """Convert a real scalar or scalar tensor to a Python float."""
        if torch.is_tensor(value):
            if value.numel() != 1:
                raise ValueError(f"{name} must be scalar.")
            return float(value.detach().cpu())
        if isinstance(value, Real):
            return float(value)
        raise TypeError(f"{name} must be a real scalar or scalar tensor.")

    def forward(
        self,
        U0,
        t_end,
        start_time=0.0,
        max_steps=1_000_000,
        infiltration=None,
        callback=None,
        boundary_level=None,
        max_dt_cap=None,
    ):
        """
        Advance the model for ``t_end`` seconds.

        Parameters
        ----------
        U0
            Initial state [batch, 3, ny, nx].
        t_end
            Duration of this call, retained under this historical name.
        start_time
            Absolute time corresponding to U0.
        max_steps
            Maximum number of internal CFL steps.
        infiltration
            Optional infiltration rate.
        callback
            Called after each step as callback(iteration, absolute_time, U).
        boundary_level
            Optional differentiable BC for this call. It takes priority over
            a dynamic boundary series and ``cfg.constant_level``.

        Returns
        -------
        U
            Final state.
        hmax
            Maximum depth reached during this call.
        """
        if U0.ndim != 4 or U0.shape[1] != 3:
            raise ValueError(
                f"U0 must have shape [batch,3,ny,nx], got {tuple(U0.shape)}."
            )

        duration = self._scalar_float(t_end, "t_end")
        absolute_start = self._scalar_float(start_time, "start_time")

        if (
            not isinstance(max_steps, Integral)
            or isinstance(max_steps, bool)
            or max_steps <= 0
        ):
            raise ValueError(f"max_steps must be positive, got {max_steps!r}.")
        if duration < 0.0:
            raise ValueError(f"t_end duration cannot be negative: {duration}.")
        if duration == 0.0:
            return U0, U0[:, 0].clone()

        U, hmax, elapsed = U0, U0[:, 0].clone(), 0.0
        tolerance = max(1e-12, 1e-12 * abs(duration))

        for iteration in range(max_steps):
            remaining = duration - elapsed
            if remaining <= tolerance:
                return U, hmax

            # --- CAP MAX DT IF STARTING ON DRY EVERYWHERE ---
            step_max_dt = remaining
            if max_dt_cap is not None:
                step_max_dt = min(remaining, self._scalar_float(max_dt_cap, "max_dt_cap"))

            current_time = absolute_start + elapsed
            level = self._resolve_boundary_level(
                current_time, U, boundary_level
            )
            U, dt_value = self.step(
                U,
                model_time=U.new_tensor(current_time),
                max_dt=U.new_tensor(step_max_dt),
                infiltration=infiltration,
                boundary_level=level,
            )

            # Per-step finite-state checks are omitted to avoid GPU
            # synchronization. A callback can provide stricter checking.
            hmax = torch.maximum(hmax, U[:, 0])
            elapsed += dt_value
            if elapsed > duration and elapsed - duration <= tolerance:
                elapsed = duration

            if self.track_gauges:
                self._record_gauges(U, absolute_start + elapsed, dt_value)

            if callback is not None:
                callback(iteration, absolute_start + elapsed, U)

        raise RuntimeError(
            f"max_steps={max_steps} reached after {elapsed:.8f} s; "
            f"requested duration={duration:.8f} s."
        )

    @torch._dynamo.disable
    def _record_gauges(
        self, U: torch.Tensor, absolute_time: float, dt_value: float
    ) -> None:
        """
        Record detached diagnostic values at every internal timestep.

        Diagnostic tracking is intentionally excluded from autograd.
        """
        self.ts_time.append(absolute_time)
        self.ts_dt.append(dt_value)
        current_bed = self.bed

        for name, (iy, ix) in self.gauge_indices.items():
            self.ts_h.setdefault(name, [])
            self.ts_z.setdefault(name, [])

            depth = float(U[0, 0, iy, ix].detach().cpu())
            bed = float(current_bed[0, 0, iy, ix].detach().cpu())
            self.ts_h[name].append(depth)
            self.ts_z[name].append(depth + bed)

    # =================================================================
    # DEM regularization
    # =================================================================

    def dem_regularization(
        self,
        prior_weight=1.0,
        slope_weight=0.0,
        curvature_weight=0.0,
    ):
        """Return prior, slope, and curvature regularization for DEM inversion."""
        bed = self.bed
        correction = bed - self.bed_reference
        dx, dy = bed[..., 1:] - bed[..., :-1], bed[..., 1:, :] - bed[..., :-1, :]

        prior = correction.square().mean()
        slope = dx.square().mean() + dy.square().mean()

        curvature_x = (
            (dx[..., 1:] - dx[..., :-1]).square().mean()
            if dx.shape[-1] >= 2
            else bed.new_zeros(())
        )
        curvature_y = (
            (dy[..., 1:, :] - dy[..., :-1, :]).square().mean()
            if dy.shape[-2] >= 2
            else bed.new_zeros(())
        )

        return (
            float(prior_weight) * prior
            + float(slope_weight) * slope
            + float(curvature_weight) * (curvature_x + curvature_y)
        )