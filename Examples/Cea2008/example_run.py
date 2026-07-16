"""Eaxmple run on a square-cell custom grid."""
import torch
from diffswe2d import SWE2D, SWEConfig, ModelGrid, io_ascii
from diffswe2d.io_ascii import ascii_to_tensor, load_rainfall_txt
from diffswe2d.model_loader import load_dynamic_model
from diffswe2d.logger import setup_logger
import xarray as xr
import numpy as np
import time
from datetime import datetime

torch.set_default_dtype(torch.float64)
device="cuda" if torch.cuda.is_available() else "cpu"

# --- Define your inputs here ---
# The script will automatically adapt based on the extension you type here!
dem_filepath = "topo.asc"  # Change this to .nc to use the NetCDF method
rain_filepath = "cstRain-test1.txt" # Choose None if no rain
roughness_filepath = None
roughness_var = None

# Set boundary conditions here: Options: "wall", "constant", "water_level", "transmissive", or "periodic"
bnd_left = "transmissive"
bnd_right = "transmissive"
bnd_top = "transmissive"
bnd_bottom = "transmissive"

# Select friction model ("roughness_length", "manning")
frictionmodel = "manning"
default_roughness_value = 0.009 # not used if spatial roughness is provided

# model domain
model_grid=ModelGrid.from_bounds(
    xmin=0.2,
    xmax=2.02,
    ymin=0.025,
    ymax=2.525,
    resolution=0.01,
    dtype=torch.float64,
    device=device)

# --- Dynamically Load the Model ---
model, use_txt_rainfall, rain_times, rain_rate_ms = load_dynamic_model(
    dem_filepath=dem_filepath, 
    rain_filepath=rain_filepath, 
    model_grid=model_grid, 
    device=device,
    boundary_left=bnd_left,
    boundary_right=bnd_right,
    boundary_top=bnd_top,
    boundary_bottom=bnd_bottom,
    frictionmodel=frictionmodel,
    roughness_filepath=roughness_filepath,
    roughness_var_name=roughness_var,
    default_roughness_value=default_roughness_value
)

#--------params------------------------------------
batch_size = 1
ini_wl = 0.0
#--------------------------------------------------
model = model.to(device)
ny,nx = model.bed.shape[-2:]
U0 = torch.zeros(batch_size, 3, ny, nx, dtype=torch.float64, device=device)
# Only use below line for a flat water surface
# U0[:,0:1] = torch.clamp(ini_wl - model.bed, min=0.0)

# --- Set up multiple gauge locations ---
# Add as many (X, Y) coordinate pairs
gauge_coords = [
    (1.02, 0.03)   # Gauge 0
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
dt_out = 1.0  # Record data every {} simulation seconds
t_end = 120.0

boundary_discharge = []

# Run inference mode to avoid gradient tracking and reduce memory usage in forward pass
model.eval()
with torch.inference_mode():
    U = U0.clone()    
    # --- The Time Loop ---
    # ---------------------------------------------------------
    # START LOGGING & TIMERS
    # ---------------------------------------------------------
    logger = setup_logger("diffswe_run.log")
    start_wall_time = time.time()
    start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"=== Simulation Started at: {start_datetime} ===")
    logger.info(f"cells: {ny}, {nx}, square resolution: {model.dx}")
    step_counter = 0
    hmax = U0[:, 0, :, :].clone()

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
            current_rain_rate = np.interp(t, rain_times, rain_rate_ms)
            U[:, 0, :, :] += (current_rain_rate * dt_out)

        # ---------------------------------------------------------
        # CALCULATE BOUNDARY DISCHARGE (RIGHT BOUNDARY)
        # ---------------------------------------------------------
        # Extract 'hv' (index 2) for all 'x' cells along the first 'y' column (index 2)
        hv_bottom = -U[0, 2, 0, :] 
        
        # Sum the unit discharges and multiply by cell width (dx) to get m³/s
        Q_right = torch.sum(hv_bottom).item() * model.dx
        boundary_discharge.append(Q_right)

        # Step the physics model forward by dt_out. note t_end in model is the duration of each run
        U, hmax_tensor = model(U, t_end=dt_out, start_time=t)
        hmax = torch.maximum(hmax, hmax_tensor)
        t += dt_out

        # ---------------------------------------------------------
        # WRITE TO LOG (EVERY {} STEPS)
        # ---------------------------------------------------------
        step_counter += 1
        if step_counter % 1 == 0:
            logger.info(f"Step: {step_counter:5d} | Sim Clock Time: {t:8.2f}s | Max Depth: {float(U[:,0].max()):.6f}m | Min Depth: {float(U[:,0].min()):.6f}m")
        
    #U, hmax_tensor  = U # Save final state
    
# Output section-------------------------------------------------
logger.info(f"depth range: {float(U[:,0].min()):.6f}, {float(U[:,0].max()):.6f}")
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

logger.info(f"Successfully saved {len(gauge_coords)} gauge text files!")

# Save Boundary Discharge Time Series
discharge_out_data = np.column_stack((times, boundary_discharge))
np.savetxt("bottom_boundary_discharge.txt", 
           discharge_out_data, 
           fmt="%.8f", 
           header="Time(s) Discharge(m3/s)", 
           comments="")
logger.info("Saved bottom_boundary_discharge.txt")


# Output the maximum water depth to a NetCDF file
# Extract and save max water depth to NetCDF
max_depth = hmax[0].cpu().numpy()  # Remove batch dimension and move to CPU

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
ds.to_netcdf("hmax_test.nc")
logger.info("Saved maximum water depth to hmax.nc")


# ---------------------------------------------------------
# END LOGGING & TIMERS
# ---------------------------------------------------------
end_wall_time = time.time()
end_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
total_computational_time = end_wall_time - start_wall_time
logger.info(f"=== Simulation Ended at: {end_datetime} ===")
logger.info(f"Total Physical Time Simulated: {t:.2f} seconds")
hours = total_computational_time/3600.0
logger.info(f"Total Computational Time: {float(hours):.4f}h")