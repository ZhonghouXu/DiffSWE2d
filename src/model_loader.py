import torch
from diffswe2d import SWE2D, SWEConfig
from diffswe2d.io_ascii import ascii_to_tensor, load_rainfall_txt

def load_dynamic_model(
    dem_filepath, 
    rain_filepath, 
    model_grid, 
    device,
    boundary_left="transmissive",
    boundary_right="wall",
    boundary_top="wall",
    boundary_bottom="water_level",
    frictionmodel = "manning"
):
    """
    Dynamically loads the SWE2D model based on the DEM file extension
    and applies custom boundary conditions.
    """
    # Initialize variables to None to ensure consistent returns
    rain_times, rain_amounts = None, None
    
    if dem_filepath.endswith('.nc'):
        print(f"Loading DEM from NetCDF: {dem_filepath}")
        model = SWE2D.from_netcdf(
            dem_filepath, 
            rainfall_path=rain_filepath,
            model_grid=model_grid,
            dem_variable="elevation",
            roughness_variable="z0",
            rainfall_variable="rainfall",
            rainfall_units="mm/h",
            config=SWEConfig(
                boundary_left=boundary_left,
                boundary_right=boundary_right,
                boundary_top=boundary_top,
                boundary_bottom=boundary_bottom,
                frictionmodel = frictionmodel
            ),
            dem_method="linear",
            roughness_method="linear",
            dem_outside_domain="error",
            train_dem=False,
            maximum_dem_correction=1.0
        ).to(device)
        
        use_txt_rainfall = False

    elif dem_filepath.endswith('.asc'):
        print(f"Loading DEM from ASCII: {dem_filepath}")
        bed_tensor, grid_spec = ascii_to_tensor(
            filepath=dem_filepath,
            model_grid=model_grid,
            method="linear",
            outside_domain="error",
            fill_internal_nodata=True,
            dtype=torch.float64,
            device=device,
        )
         
        model = SWE2D.from_ascii(
            dem_path=dem_filepath,
            roughness_path="roughness.asc", 
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
            dem_outside_domain="error",
            roughness_outside_domain="nearest",
            fill_dem_nodata=True,
            fill_roughness_nodata=True,
            train_dem=False,
            maximum_dem_correction=1.0,
            device=device,
            dtype=torch.float64,
        ).to(device)
        
        # Load the text rainfall
        rain_times, rain_amounts = load_rainfall_txt(rain_filepath)
        use_txt_rainfall = True

    else:
        raise ValueError("Unsupported DEM format! Please provide a .nc or .asc file.")
        
    return model, use_txt_rainfall, rain_times, rain_amounts