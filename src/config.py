from dataclasses import dataclass


@dataclass
class SWEConfig:
    """Numerical configuration for the structured-grid solver."""

    gravity: float = 9.81
    dry_depth: float = 1.0e-6
    epsilon: float = 1.0e-12
    cfl: float = 0.5
    limiter_theta: float = 1.3
    # Boundary options: "wall", "constant", "water_level", "transmissive", or "periodic"
    boundary_left: str = "water_level"   
    boundary_right: str = "water_level"
    boundary_top: str = "water_level"
    boundary_bottom: str = "water_level"
    constant_value: float = 0.0 # For "constant" boundary
    constant_level: float = 0.0 # For "water_level" boundary
    water_slope_reset: bool = True
    positivity_limiter: bool = True
    apply_friction: bool = True
    frictionmodel: str = "manning"

    # Rainfall interpolation controls used by SWE2D.from_netcdf().
    # Conservative rainfall remapping supports zero outside source coverage,
    # or strict rejection when the model grid is not fully covered.
    rainfall_outside_domain: str = "zero"    # "zero" or "error"
