import os
import time
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

# Adjust these imports according to the actual structure of DiffSWE2d
# based on the original example_run.py imports.
# e.g., from src.utils import load_dynamic_model

# =====================================================================
# UTILITIES
# =====================================================================
def setup_logger(name):
    import logging
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger

# =====================================================================
# MAIN SCRIPT
# =====================================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = setup_logger("diffswe_train_bc")
    logger.info(f"Using device: {device}")

    # -----------------------------------------------------------------
    # 1. DIRECTORIES AND FILES (Monai Valley Example)
    # -----------------------------------------------------------------
    data_dir = "data"  # Update this to match repository structure
    dem_filepath = os.path.join(data_dir, "Monai_Valley_DEM.asc")
    roughness_filepath = os.path.join(data_dir, "Monai_Valley_Roughness.asc")
    rain_filepath = None  
    
    # CRITICAL: tide_filepath must be None so we can control the BC dynamically
    tide_filepath = None  
    
    # -----------------------------------------------------------------
    # 2. BOUNDARY CONDITIONS & CONFIGURATION
    # -----------------------------------------------------------------
    bnd_left = "water_level"  # This is the boundary we will optimize
    bnd_right = "wall"
    bnd_top = "wall"
    bnd_bottom = "wall"
    
    frictionmodel = "manning"
    default_roughness_value = 0.025
    roughness_var = "n"

    # Gauge locations (X, Y coordinates for Monai Valley)
    track_gauges = True
    gauge_coords = {
        "gauge_0": (4.521, 1.196),
        "gauge_1": (4.521, 1.696),
        "gauge_2": (4.521, 2.196),
    }

    # Simulation time variables
    dt_out = 1.0  # Extract state every 1 second
    t_end = 30.0  # Total simulation time in seconds
    num_steps = int(t_end / dt_out)
    batch_size = 1

    # -----------------------------------------------------------------
    # 3. DYNAMICALLY LOAD THE MODEL
    # -----------------------------------------------------------------
    logger.info("Loading model and grid...")
    
    # NOTE: Adjust the arguments below to perfectly match your repository's 
    # load_dynamic_model function signature.
    model, use_txt_rainfall, rain_times, rain_rate_ms = load_dynamic_model(
        dem_filepath=dem_filepath,
        rain_filepath=rain_filepath,
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
    ny, nx = model.bed.shape[-2:]

    # Map real-world coordinates to grid indices
    model.track_gauges = track_gauges
    model.gauge_indices = {}
    model.bed_elevations = {}
    
    # Assuming the grid object is attached to the model (e.g., model.grid)
    for name, (target_x, target_y) in gauge_coords.items():
        ix = torch.argmin(torch.abs(model.grid.x - target_x)).item()
        iy = torch.argmin(torch.abs(model.grid.y - target_y)).item()
        model.gauge_indices[name] = (iy, ix)
        model.bed_elevations[name] = model.bed[0, 0, iy, ix].item()

    # -----------------------------------------------------------------
    # 4. OPTIMIZATION SETUP
    # -----------------------------------------------------------------
    logger.info("Setting up optimization parameters...")
    
    # 1D tensor representing the left boundary water level for each dt_out interval
    initial_bc_series = torch.ones(num_steps, dtype=torch.float64, device=device) * 0.5
    left_bc_param = torch.nn.Parameter(initial_bc_series, requires_grad=True)

    optimizer = optim.Adam([left_bc_param], lr=0.01)

    # Define target observations (Replace with your actual measured data arrays)
    # Shape: [num_steps] (30 observations for 30 seconds)
    target_wls = {
        "gauge_0": torch.ones(num_steps, dtype=torch.float64, device=device) * 0.8,
        "gauge_1": torch.ones(num_steps, dtype=torch.float64, device=device) * 0.8,
        "gauge_2": torch.ones(num_steps, dtype=torch.float64, device=device) * 0.8,
    }

    # -----------------------------------------------------------------
    # 5. TRAINING LOOP
    # -----------------------------------------------------------------
    epochs = 20

    # CRITICAL: Ensure the model is in train mode so gradients can flow
    # Do NOT run torch.inference_mode() or torch.compile() here
    model.train() 

    logger.info("Starting BPTT optimization...")

    for epoch in range(epochs):
        epoch_start_time = time.time()
        optimizer.zero_grad()
        
        # Initialize standard water depth (Start dry or with a base water level)
        U0 = torch.zeros(batch_size, 3, ny, nx, dtype=torch.float64, device=device)
        
        U = U0.clone()
        t = 0.0
        step_idx = 0
        
        simulated_wls = {name: [] for name in gauge_coords.keys()}

        # The Time Loop (Forward Pass)
        while t < (t_end - 1e-6) and step_idx < num_steps:
            
            # Inject the trainable boundary value for the current timestep
            current_bc_wl = left_bc_param[step_idx]
            
            # --------------------------------------------------------------
            # IMPORTANT: Assign to the correct boundary attribute!
            # Replace 'tide_level' with whatever attribute DiffSWE2d reads 
            # for the left water_level boundary in its forward pass.
            # --------------------------------------------------------------
            model.tide_level = current_bc_wl 
            
            if use_txt_rainfall:
                current_rain_rate = np.interp(t, rain_times, rain_rate_ms)
                U[:, 0, :, :] += (current_rain_rate * dt_out)

            # Step the model forward by 1 second (dt_out)
            U, _ = model(U, t_end=dt_out, start_time=t)
            
            # Extract water level at gauges differentiably
            for name in gauge_coords.keys():
                iy, ix = model.gauge_indices[name]
                depth = U[0, 0, iy, ix]
                bed_z = model.bed_elevations[name]
                wl = depth + bed_z
                simulated_wls[name].append(wl)

            t += dt_out
            step_idx += 1

        # Calculate MSE Loss over the entire 30-second time series
        loss = 0.0
        for name in target_wls.keys():
            sim_tensor = torch.stack(simulated_wls[name])
            loss += F.mse_loss(sim_tensor, target_wls[name])

        # Backpropagation Through Time (BPTT)
        loss.backward()
        
        # Update parameters
        optimizer.step()

        # Constrain boundary levels to physically realistic bounds
        with torch.no_grad():
            left_bc_param.clamp_(min=0.0)

        elapsed = time.time() - epoch_start_time
        logger.info(f"Epoch {epoch+1:03d} | Loss: {loss.item():.6f} | "
                    f"BC at t=0: {left_bc_param[0].item():.4f}m | Time: {elapsed:.2f}s")

if __name__ == "__main__":
    main()