import torch
from diffswe2d import SWE2D, SWEConfig
from diffswe2d.io_ascii import ascii_to_tensor, load_rainfall_txt

def load_dynamic_model(
    dem_filepath,     
    model_grid, 
    device,
    rain_filepath=None, 
    boundary_left="transmissive",
    boundary_right="wall",
    boundary_top="wall",
    boundary_bottom="water_level",
    frictionmodel = "manning",
    roughness_filepath=None,
    roughness_var_name="z0",
    default_roughness_value=0.025
):
    """
    Dynamically loads the SWE2D model based on the DEM file extension
    and applies custom boundary conditions.
    """
    # Initialize variables to None to ensure consistent returns
    rain_times, rain_rate_ms = None, None
    
    if dem_filepath.endswith('.nc'):
        print(f"Loading DEM from NetCDF: {dem_filepath}")
        # --- Dynamically set roughness arguments ---
        roughness_kwargs = {}
        if roughness_filepath:
            roughness_kwargs["roughness_path"] = roughness_filepath
            roughness_kwargs["roughness_variable"] = roughness_var_name
        else:
            roughness_kwargs["default_roughness"] = default_roughness_value

        # --- Dynamically set rainfall arguments ---
        rainfall_kwargs = {}
        if rain_filepath:
            rainfall_kwargs["rainfall_path"] = rain_filepath
            rainfall_kwargs["rainfall_variable"] = "rainfall"
            rainfall_kwargs["rainfall_units"] = "mm/h"

        model = SWE2D.from_netcdf(
            dem_filepath, 
            model_grid=model_grid,
            dem_variable="elevation",
            roughness_variable=roughness_var_name,
            config=SWEConfig(
                boundary_left=boundary_left,
                boundary_right=boundary_right,
                boundary_top=boundary_top,
                boundary_bottom=boundary_bottom,
                frictionmodel = frictionmodel
            ),
            dem_method="linear",
            roughness_method="linear",
            dem_outside_domain="nearest",
            train_dem=False,
            maximum_dem_correction=1.0,
            **roughness_kwargs,
            **rainfall_kwargs
        ).to(device)
        
        use_txt_rainfall = False

    elif dem_filepath.endswith('.asc'):
        print(f"Loading DEM from ASCII: {dem_filepath}")

        # --- Dynamically set roughness arguments ---
        roughness_kwargs = {}
        if roughness_filepath:
            roughness_kwargs["roughness_path"] = roughness_filepath
        else:
            roughness_kwargs["default_roughness"] = default_roughness_value

        bed_tensor, grid_spec = ascii_to_tensor(
            filepath=dem_filepath,
            model_grid=model_grid,
            method="linear",
            outside_domain="nearest",
            fill_internal_nodata=True,
            dtype=torch.float64,
            device=device,
        )
         
        model = SWE2D.from_ascii(
            dem_path=dem_filepath,
            model_grid=model_grid,
            config=SWEConfig(
                boundary_left=boundary_left,
                boundary_right=boundary_right,
                boundary_top=boundary_top,
                boundary_bottom=boundary_bottom,
                frictionmodel = frictionmodel
            ),
            dem_method="linear",
            roughness_method="nearest",
            dem_outside_domain="nearest",
            roughness_outside_domain="nearest",
            fill_dem_nodata=True,
            fill_roughness_nodata=True,
            train_dem=False,
            maximum_dem_correction=1.0,
            device=device,
            dtype=torch.float64,
            **roughness_kwargs
        ).to(device)
        
        # Load the text rainfall if provided
        if rain_filepath:
            rain_times, rain_rate_ms = load_rainfall_txt(rain_filepath)
            use_txt_rainfall = True
        else:
            use_txt_rainfall = False
            # rain_times and rain_amounts are already initialized to None at the top of the function

    else:
        raise ValueError("Unsupported DEM format! Please provide a .nc or .asc file.")
        
    return model, use_txt_rainfall, rain_times, rain_rate_ms