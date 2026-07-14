"""Differentiable B-Flood-style 2-D shallow-water solver."""
from .config import SWEConfig
from .forcing import RainfallForcing
from .io_netcdf import ModelGrid, GridData, load_dem_and_roughness_to_grid, load_rainfall_to_grid
from .solver import SWE2D
from .io_ascii import ascii_to_tensor, parse_model_grid
__all__=[
    "SWEConfig",
    "SWE2D",
    "RainfallForcing",
    "ModelGrid",
    "GridData",
    "load_dem_and_roughness_to_grid",
    "load_rainfall_to_grid",
    "ascii_to_tensor",
    "parse_model_grid"
    ]
