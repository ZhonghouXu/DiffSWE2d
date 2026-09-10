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
bnd_left = "wall"
bnd_right = "wall"
bnd_top = "wall"
bnd_bottom = "transmissive"

# Select friction model ("roughness_length", "manning")
frictionmodel = "manning"
default_roughness_value = 0.012 # not used if spatial roughness is provided

# model domain
model_grid=ModelGrid.from_bounds(
    xmin=0.02,
    xmax=2.02,
    ymin=-0.005,
    ymax=2.535,
    resolution=0.01,
    dtype=torch.float64,
    device=device)

# --- Set up multiple gauge locations ---
# Add as many (X, Y) coordinate pairs
track_gauges = True
gauge_coords = {
    "gauge_0": (1.02, 0.0), 
    "gauge_1": (1.02, 0.03),
    "gauge_2": (1.02, 0.1),
    "gauge_3": (1.02, 1.0),   
}

# --- Time keeping variables ---
t = 0.0
dt_out = 0.1  # Record data every {} simulation seconds
t_end = 120.0

#--------params------------------------------------
batch_size = 1
ini_wl = 0.0
#--------------------------------------------------
#---END OF INPUTS----------------------------------
# ---Below lines call the model and can be modified for custom outputs

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

model = model.to(device)
ny,nx = model.bed.shape[-2:]
U0 = torch.zeros(batch_size, 3, ny, nx, dtype=torch.float64, device=device)
# Only use below line for a flat water surface
# U0[:,0:1] = torch.clamp(ini_wl - model.bed, min=0.0)

# Activate tracking inside the model
model.track_gauges = track_gauges

gauge_indices = []
bed_elevations = []

# Convert all coordinates to grid indices and load them into the model
for name, (target_x, target_y) in gauge_coords.items():
    ix = torch.argmin(torch.abs(model_grid.x - target_x)).item()
    iy = torch.argmin(torch.abs(model_grid.y - target_y)).item()
    
    # Store indices and elevations directly in the model
    model.gauge_indices[name] = (iy, ix)
    model.bed_elevations[name] = model.bed[0, 0, iy, ix].item()
    
    # Create empty lists for this specific gauge's data
    model.ts_h[name] = []
    model.ts_z[name] = []

times = []


boundary_discharge = []

# Run inference mode to avoid gradient tracking and reduce memory usage in forward pass
model.eval()
# Speed up PyTorch code by just-in-time (JIT) compiling it into optimized C++ kernels
model = torch.compile(model)
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

    while t < (t_end - 1e-6):
        times.append(t)        

        if use_txt_rainfall:
            current_rain_rate = np.interp(t, rain_times, rain_rate_ms)
            U[:, 0, :, :] += (current_rain_rate * dt_out)

        # ---------------------------------------------------------
        # CALCULATE BOUNDARY DISCHARGE (RIGHT BOUNDARY)
        # ---------------------------------------------------------
        # Extract 'hv' (index 2) for 'x' cells (40:160) along the fourth 'y' column (index 3)
        hv_bottom = -U[0, 2, 5, 50:150] 
        
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
for name in model.gauge_indices.keys():
    # Stack the micro-step lists: time, dt, depth, and elevation
    out_data = np.column_stack((
        model.ts_time, 
        model.ts_dt, 
        model.ts_h[name], 
        model.ts_z[name]
    ))
    
    # Name the file exactly what you called it in the dictionary (e.g., Gauge_0.txt)
    filename = f"{name}.txt"
    target_x, target_y = gauge_coords[name]
    
    custom_header = f"Gauge {name} Location: X={target_x}, Y={target_y}\nTime(s) dt(s) Depth(m) Elevation(m)"
    
    np.savetxt(filename, out_data, fmt="%.6f", header=custom_header, comments="")
    
logger.info(f"Successfully saved {len(gauge_coords)} micro-step gauge text files!")

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
ds.to_netcdf("hmax_n_0009.nc")
logger.info("Saved maximum water depth to netcdf file")


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