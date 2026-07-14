"""Run on a square-cell custom grid with conservative rainfall remapping."""
import torch
from diffswe2d import SWE2D, SWEConfig, ModelGrid, io_ascii
from diffswe2d.io_ascii import ascii_to_tensor, load_rainfall_txt
from diffswe2d.model_loader import load_dynamic_model
import xarray as xr
import numpy as np

torch.set_default_dtype(torch.float64)
device="cuda" if torch.cuda.is_available() else "cpu"

# --- Define your inputs here ---
# The script will automatically adapt based on the extension you type here!
dem_filepath = "my_topography.asc"  # Change this to .nc to use the NetCDF method
rain_filepath = "my_rainfall.txt"
roughness_filepath = "my_manning_grid.asc"
roughness_var = "n"

# Set boundary conditions here
bnd_left = "transmissive"
bnd_right = "wall"
bnd_top = "wall"
bnd_bottom = "water_level"

# Select friction model ("roughness_length", "manning")
frictionmodel = "manning"

# model domain
model_grid=ModelGrid.from_bounds(
    xmin=0.,
    xmax=1580.,
    ymin=0.,
    ymax=780.,
    resolution=10.,
    dtype=torch.float64,
    device=device)

# --- Dynamically Load the Model ---
model, use_txt_rainfall, rain_times, rain_amounts = load_dynamic_model(
    dem_filepath=dem_filepath, 
    rain_filepath=rain_filepath, 
    model_grid=model_grid, 
    device=device,
    boundary_left=bnd_left,
    boundary_right=bnd_right,
    boundary_top=bnd_top,
    boundary_bottom=bnd_bottom,
    frictionmodel=frictionmodel,
    roughness_filepath=roughness_filepath
    roughness_var_name=roughness_var
)

#--------params------------------------------------
batch_size = 1
ini_wl = 0.0
#--------------------------------------------------
model = model.to(device)
ny,nx = model.bed.shape[-2:]
U0 = torch.zeros(batch_size, 3, ny, nx, dtype=torch.float64, device=device)
U0[:,0:1] = torch.clamp(ini_wl - model.bed, min=0.0)

# --- Set up multiple gauge locations ---
# Add as many (X, Y) coordinate pairs
gauge_coords = [
    (500.0, 300.0),   # Gauge 0
    (300.0, 700.0),  # Gauge 1
    (1200.0, 150.0)   # Gauge 2
]

gauge_indices = []
bed_elevations = []

# Convert all coordinates to grid indices and extract their bed elevations
for target_x, target_y in gauge_coords:
    # Find the index with the minimum absolute distance to the target coordinate
    ix = torch.argmin(torch.abs(model_grid.x - target_x)).item()
    iy = torch.argmin(torch.abs(model_grid.y - target_y)).item()
    
    gauge_indices.append((iy, ix))
    bed_elevations.append(model.bed[0, 0, iy, ix].item())

# Initialize a dictionary to store the lists of h and z for each gauge
# It will look like: {0: {'h': [], 'z': []}, 1: {'h': [], 'z': []}, ...}
gauge_data = {i: {'h': [], 'z': []} for i in range(len(gauge_coords))}
times = []

t = 0.0
dt_out = 10.0  # Record data every 10 simulation seconds
t_end = 3600.0

# Run inference mode to avoid gradient tracking and reduce memory usage in forward pass
model.eval()
with torch.inference_mode():
    U = U0.clone()    
    # --- The Time Loop ---
    while t <= t_end:
        times.append(t)
        
        # Loop through every gauge and record its state
        for i, (iy, ix) in enumerate(gauge_indices):
            # Extract depth (h) at this specific gauge
            h_point = U[0, 0, iy, ix].item()            
            # Calculate elevation (z)
            z_point = h_point + bed_elevations[i]
            
            # Store the values in our dictionary
            gauge_data[i]['h'].append(h_point)
            gauge_data[i]['z'].append(z_point)

        if use_txt_rainfall:
            current_rain = np.interp(t, rain_times, rain_amounts)
            U[:, 0, :, :] += (current_rain * dt_out)

        # Step the physics model forward by dt_out. note t_end in model is the duration of each run
        U, hmax_tensor = model(U, t_end=dt_out, start_time=t)
        t += dt_out
        
    #U, hmax_tensor  = U # Save final state

# Output section-------------------------------------------------
print("cells:",ny,nx,"square resolution:",model.dx)
print("depth range:",float(U[:,0].min()),float(U[:,0].max()))
# --- Save Time Series to TXT files ---
for i in range(len(gauge_coords)):
    # Combine the time list with this specific gauge's h and z lists
    out_data = np.column_stack((times, gauge_data[i]['h'], gauge_data[i]['z']))
    
    # Create a dynamic filename (gauge_0.txt, gauge_1.txt, etc.)
    filename = f"gauge_{i}.txt"
    
    # Include the real-world coordinates in the header so you know which file is which
    custom_header = f"Gauge {i} Location: X={gauge_coords[i][0]}, Y={gauge_coords[i][1]}\nTime(s) Depth(m) Elevation(m)"
    
    np.savetxt(
        filename, 
        out_data, 
        fmt="%.3f", 
        header=custom_header, 
        comments="" 
    )

print(f"Successfully saved {len(gauge_coords)} gauge text files!")


# Output the maximum water depth to a NetCDF file
# Extract and save max water depth to NetCDF
max_depth = hmax_tensor[0].cpu().numpy()  # Remove batch dimension and move to CPU

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
