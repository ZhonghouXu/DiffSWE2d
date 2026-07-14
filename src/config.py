from dataclasses import dataclass


@dataclass
class SWEConfig:
    """Numerical configuration for the structured-grid solver."""

    gravity: float = 9.81
    dry_depth: float = 1.0e-6
    epsilon: float = 1.0e-12
    cfl: float = 0.35
    limiter_theta: float = 1.3
    boundary: str = "water_level"  # "wall", "constant", "water_level", "transmissive", or "periodic"
    water_slope_reset: bool = True
    positivity_limiter: bool = True
    apply_friction: bool = True

    # Rainfall interpolation controls used by SWE2D.from_netcdf().
    # Conservative rainfall remapping supports zero outside source coverage,
    # or strict rejection when the model grid is not fully covered.
    rainfall_outside_domain: str = "zero"    # "zero" or "error"
