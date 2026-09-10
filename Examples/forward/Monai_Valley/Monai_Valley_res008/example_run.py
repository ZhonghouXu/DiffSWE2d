"""Example run on a square-cell custom grid."""
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
# -------------------------------
# The script will automatically adapt based on the extension you type here!
dem_filepath = "Monai_Bathy.nc"  # DEM format: .ASC OR .NC
rain_filepath = None # Choose None if no rain
roughness_filepath = None
roughness_var = None
tide_filepath="MonaiValley_InputWave.txt" # Optional (a path or None)

# Set boundary conditions here: Options: "wall", "constant", "water_level", "transmissive", or "periodic"
bnd_left = "water_level"
bnd_right = "wall"
bnd_top = "wall"
bnd_bottom = "wall"

# Select friction model ("roughness_length", "manning")
frictionmodel = "manning"
default_roughness_value = 0.01 # not used if spatial roughness is provided

# model domain
model_grid=ModelGrid.from_bounds(
    xmin=0.,
    xmax=5.44,
    ymin=0.,
    ymax=3.36,
    resolution=0.08,
    dtype=torch.float64,
    device=device)

# --- Set up multiple gauge locations ---
# Add (X, Y) coordinate pairs
track_gauges = True
gauge_coords = {
    "gauge_0": (4.521, 1.196),   
    "gauge_1": (4.521, 1.696),   
    "gauge_2": (4.521, 2.196),
    "gauge_3": (1.000, 1.696), 
    "gauge_4": (2.000, 1.696), 
    "gauge_5": (3.000, 1.696),    
}

# --- Time keeping variables ---
t = 0.0
dt_out = 30.0  # Record data every {} simulation seconds
t_end = 30.0

#--------params------------------------------------
batch_size = 1
ini_wl = 0.0 # Optional (a value or None), only applied with flat water surface initialization
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
    default_roughness_value=default_roughness_value,
    tide_filepath=tide_filepath
)

model = model.to(device)
ny,nx = model.bed.shape[-2:]
U0 = torch.zeros(batch_size, 3, ny, nx, dtype=torch.float64, device=device)
# If ini_wl is not None, initialise a flat water surface
if ini_wl is not None:
    U0[:,0:1] = torch.clamp(ini_wl - model.bed, min=0.0)

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
ds.to_netcdf("hmax.nc")
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