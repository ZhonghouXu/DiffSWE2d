"""Run on a square-cell custom grid with conservative rainfall remapping."""
import torch
from DIFFSWE2D import SWE2D, SWEConfig, ModelGrid

torch.set_default_dtype(torch.float64)
device="cuda" if torch.cuda.is_available() else "cpu"

model_grid=ModelGrid.from_bounds(
    xmin=100000.,xmax=120000.,ymin=500000.,ymax=515000.,
    resolution=25.,dtype=torch.float64,device=device)

model=SWE2D.from_netcdf(
    dem_path="dem_and_roughness.nc",
    rainfall_path="rainfall.nc",
    model_grid=model_grid,
    dem_variable="elevation",
    roughness_variable="z0",
    rainfall_variable="rainfall",
    rainfall_units="mm/h",
    config=SWEConfig(rainfall_outside_domain="zero"),
    dem_method="linear",
    roughness_method="linear",
    dem_outside_domain="error",
    train_dem=False,
    maximum_dem_correction=1.0)

ny,nx=model.bed.shape[-2:]
U0=torch.zeros(1,3,ny,nx,dtype=torch.float64,device=device)
U0[:,0:1]=torch.clamp(1.0-model.bed,min=0.0)

# Run inference mode to avoid gradient tracking and reduce memory usage in forward pass
model.eval()
with torch.inference_mode():
    result=model(U0,t_end=3600.)
print("cells:",ny,nx,"square resolution:",model.dx)
print("depth range:",float(result[:,0].min()),float(result[:,0].max()))

# Output the maximum water depth to a NetCDF file
# Extract and save max water depth to NetCDF
max_depth = result[0, 0].cpu().numpy()  # Remove batch dimension and move to CPU

# Create xarray Dataset
ds = xr.Dataset(
    {
        "hmax": (["y", "x"], max_depth),
    },
    coords={
        "x": model_grid.x.cpu().numpy(),
        "y": model_grid.y.cpu().numpy(),
    },
    attrs={
        "description": "Maximum water depth",
        "units": "metres",
    }
)

# Save to NetCDF
ds.to_netcdf("hmax.nc")
print("Saved maximum water depth to hmax.nc")