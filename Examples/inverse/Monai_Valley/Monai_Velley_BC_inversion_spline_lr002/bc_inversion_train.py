from __future__ import annotations

"""
Boundary-condition inversion for the Monai Valley DiffSWE2D model.

Purpose
-------
This script infers the incident water-level boundary hydrograph from remote
wave-gauge observations. Instead of learning one independent boundary value
for every 0.2-second model interval, it learns a small set of bounded cubic
B-spline control values. A precomputed spline basis maps those controls to the
full boundary time series while preserving PyTorch autograd.

Data split
----------
* TRAIN_GAUGES contribute to the data-misfit term and therefore influence the
  optimized spline controls.
* TEST_GAUGES are simulated during every epoch but never enter the objective,
  learning-rate scheduler, early stopping criterion, or best-state selection.
  They provide an independent check on spatial generalization.

Important modelling conventions
--------------------------------
1. Gauge water levels and model elevations are both in metres.
2. The solver's ``t_end`` argument is a duration for one call. Each outer
   interval therefore calls the solver with ``t_end=DT_OUT``.
3. The adaptive CFL timestep is detached inside the updated solver. The state
   update and boundary forcing remain differentiable, while the selected time
   grid is treated as numerical control flow.
4. ``boundary_level`` is supplied explicitly to the solver, avoiding mutation
   of ``model.cfg.constant_level`` during checkpoint recomputation.
5. The held-out test loss is reported only; it is not used to tune parameters.

Spline parameterization
-----------------------
The optimization variable is an unconstrained vector ``raw_controls``. A
sigmoid maps it smoothly into [BC_MINIMUM, BC_MAXIMUM]. A clamped B-spline
basis then reconstructs the piecewise-smooth boundary value used during each
model output interval:

    raw controls -> bounded controls -> B-spline basis -> dense BC series

The basis matrix is constant and is built once. Gradients flow from the gauge
loss through the SWE model, dense boundary series, spline matrix, bounded
control values, and finally to the raw controls.

Outputs
-------
The script saves the optimized dense hydrograph, optimized spline controls,
training histories, train/test gauge comparisons, and restartable PyTorch
checkpoints under OUTPUT_DIRECTORY.
"""

import logging
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.checkpoint import checkpoint

from diffswe2d import ModelGrid
from diffswe2d.model_loader import load_dynamic_model

# =====================================================================
# Configuration is intentionally centralized so that experiments can be
# reproduced using only the script and the saved checkpoint metadata.
# Input gauge values are assumed to already be in metres.
# FILES, MODEL, TIME, AND GAUGES
# =====================================================================
DEM_FILEPATH = "Monai_Bathy.nc"
GAUGE_FILEPATH = "MonaiValley_WaveGages_30s_02s.txt"
ROUGHNESS_FILEPATH = None
ROUGHNESS_VARIABLE = None
RAIN_FILEPATH = None
TIDE_FILEPATH = None
OUTPUT_DIRECTORY = Path("inverse_results_spline")

MODEL_DTYPE = torch.float64
GRID_X_MIN, GRID_X_MAX = 0.0, 5.44
GRID_Y_MIN, GRID_Y_MAX = 0.0, 3.36
GRID_RESOLUTION = 0.08
BOUNDARY_LEFT, BOUNDARY_RIGHT = "water_level", "wall"
BOUNDARY_TOP, BOUNDARY_BOTTOM = "wall", "wall"
FRICTION_MODEL = "manning"
DEFAULT_ROUGHNESS_VALUE = 0.01
BATCH_SIZE = 1
INITIAL_WATER_LEVEL = 0.0
MINIMUM_INITIAL_DEPTH = 0.0

DT_OUT = 0.2
T_END = 29.8

GAUGE_COORDINATES = {
    "gauge_0": (4.521, 1.196),
    "gauge_1": (4.521, 1.696),
    "gauge_2": (4.521, 2.196),
}
GAUGE_COLUMNS = {"gauge_0": 1, "gauge_1": 2, "gauge_2": 3}
GAUGE_FILE_SKIP_ROWS = 1
OBSERVATION_TYPE = "water_level"

# Two gauges train the inversion; the held-out gauge is reported only.
TRAIN_GAUGES = ("gauge_0", "gauge_1")
TEST_GAUGES = ("gauge_2",)

# =====================================================================
# The spline controls are the only trainable physical parameters. Twenty
# controls provide substantial dimension reduction relative to 149 independent
# boundary values while retaining enough flexibility for a transient wave.
# SPLINE AND OPTIMIZATION
# =====================================================================
# A clamped cubic B-spline with 20 control values.
NUM_SPLINE_CONTROLS = 20
SPLINE_DEGREE = 3
# Guidance:
#   12-16 controls -> stronger smoothing and lower overfitting risk.
#   20 controls    -> recommended initial balance for this 29.8 s record.
#   25-30 controls -> more temporal detail but greater ill-conditioning risk.
# A cubic spline requires at least SPLINE_DEGREE + 1 controls.
BC_MINIMUM, BC_MAXIMUM = -0.02, 0.05
INITIAL_BC_VALUE = 0.0

MAX_EPOCHS = 500
LEARNING_RATE = 0.02
MINIMUM_LEARNING_RATE = 1.0e-5
WEIGHT_DECAY = 0.0
MAX_GRADIENT_NORM = 1.0
FIRST_DERIVATIVE_WEIGHT = 1.0e-3
SECOND_DERIVATIVE_WEIGHT = 1.0e-2
AMPLITUDE_WEIGHT = 1.0e-6
EARLY_STOPPING_PATIENCE = 10
EARLY_STOPPING_MIN_DELTA = 1.0e-8
LR_SCHEDULER_PATIENCE = 3
LR_SCHEDULER_FACTOR = 0.5

# =====================================================================
# Checkpointing trades memory for extra forward recomputation. On an A100 with
# this small grid it is usually faster to leave checkpointing disabled.
# Anomaly detection should only be enabled while diagnosing invalid gradients.
# PERFORMANCE AND DIAGNOSTICS
# =====================================================================
USE_CHECKPOINTING = False
USE_TORCH_COMPILE = False
ENABLE_AUTOGRAD_ANOMALY_DETECTION = False
STATE_CHECK_INTERVAL = 10
LOG_FIRST_EPOCH_FORWARD_PROGRESS = True
FORWARD_LOG_INTERVAL = 25
CHECKPOINT_SAVE_INTERVAL = 1
STATE_GRADIENT_CLIP = 100.0


def setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(handler)
    return logger


# =====================================================================
# DIFFERENTIABLE CLAMPED B-SPLINE
# =====================================================================
def inverse_bounded_sigmoid(value: float, minimum: float, maximum: float, *, dtype, device):
    if not minimum < value < maximum:
        raise ValueError(f"Initial value {value} must lie inside ({minimum}, {maximum}).")
    fraction = torch.tensor((value - minimum) / (maximum - minimum), dtype=dtype, device=device)
    eps = torch.finfo(dtype).eps
    return torch.logit(fraction.clamp(eps, 1.0 - eps))


def bounded_control_values(raw_controls: torch.Tensor) -> torch.Tensor:
    return BC_MINIMUM + (BC_MAXIMUM - BC_MINIMUM) * torch.sigmoid(raw_controls)


# This routine implements the Cox-de Boor recursion directly with Torch tensor
# operations. The resulting basis does not depend on trainable parameters, so
# it is computed once before optimization and reused during every epoch.

def build_clamped_bspline_basis(
    sample_times: torch.Tensor,
    number_of_controls: int,
    degree: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return basis matrix, knot vector, and Greville control-point times."""
    if degree < 1 or number_of_controls < degree + 1:
        raise ValueError("Require degree >= 1 and number_of_controls >= degree + 1.")
    if sample_times.ndim != 1 or sample_times.numel() < 2:
        raise ValueError("sample_times must be a one-dimensional tensor with at least two values.")

    t0, t1 = sample_times[0], sample_times[-1]
    interior_count = number_of_controls - degree - 1
    interior = (
        torch.linspace(t0, t1, interior_count + 2, dtype=sample_times.dtype, device=sample_times.device)[1:-1]
        if interior_count > 0 else sample_times.new_empty(0)
    )
    knots = torch.cat((t0.repeat(degree + 1), interior, t1.repeat(degree + 1)))

    x = sample_times[:, None]
    basis = ((x >= knots[:-1]) & (x < knots[1:])).to(sample_times.dtype)
    for order in range(1, degree + 1):
        columns = knots.numel() - order - 1
        left_den = knots[order:order + columns] - knots[:columns]
        right_den = knots[order + 1:order + 1 + columns] - knots[1:1 + columns]
        left = torch.where(left_den > 0, (x - knots[:columns]) / left_den.clamp_min(torch.finfo(x.dtype).eps), 0.0)
        right = torch.where(right_den > 0, (knots[order + 1:order + 1 + columns] - x) / right_den.clamp_min(torch.finfo(x.dtype).eps), 0.0)
        basis = left * basis[:, :columns] + right * basis[:, 1:columns + 1]

    # Include the closed right endpoint in the final basis function.
    # Half-open knot intervals exclude the exact right endpoint. Explicitly assign
    # that endpoint to the final basis function so the final row still sums to 1.
    basis[-1].zero_()
    basis[-1, -1] = 1.0
    row_sums = basis.sum(dim=1)
    if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-10, rtol=1e-10):
        raise RuntimeError("B-spline basis does not form a partition of unity.")

    greville_times = torch.stack([
        knots[index + 1:index + degree + 1].mean()
        for index in range(number_of_controls)
    ])
    return basis, knots, greville_times


# Matrix multiplication is both efficient and fully differentiable. Bounds are
# imposed on the controls rather than clipping the dense series after spline
# evaluation, avoiding a hard post-interpolation gradient cutoff.

def spline_boundary_series(raw_controls: torch.Tensor, basis: torch.Tensor):
    controls = bounded_control_values(raw_controls)
    # Each row of basis sums to one, so the dense boundary is a convex combination
    # of nearby bounded controls and therefore remains inside the same bounds.
    return basis @ controls, controls


# =====================================================================
# GRID, GAUGES, AND OBSERVATIONS
# =====================================================================
def extract_coordinate_vectors(model_grid: ModelGrid) -> Tuple[torch.Tensor, torch.Tensor]:
    x = model_grid.x[0, :] if model_grid.x.ndim == 2 else model_grid.x
    y = model_grid.y[:, 0] if model_grid.y.ndim == 2 else model_grid.y
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("Could not extract one-dimensional grid coordinates.")
    return x, y


# Gauge coordinates are mapped to the nearest cell centre. For higher-resolution
# applications, bilinear spatial sampling may reduce grid-location error.

def configure_gauges(model, model_grid: ModelGrid, logger: logging.Logger) -> None:
    x, y = extract_coordinate_vectors(model_grid)
    model.track_gauges = False
    model.gauge_indices, model.bed_elevations = {}, {}
    bed = model.bed.detach()
    for name, (target_x, target_y) in GAUGE_COORDINATES.items():
        ix = int(torch.argmin(torch.abs(x - target_x)).item())
        iy = int(torch.argmin(torch.abs(y - target_y)).item())
        model.gauge_indices[name] = (iy, ix)
        bed_z = float(bed[0, 0, iy, ix].cpu())
        model.bed_elevations[name] = bed_z
        logger.info(
            "%s mapped to row=%d, column=%d, x=%.4f, y=%.4f, bed=%.4f m",
            name, iy, ix, float(x[ix].cpu()), float(y[iy].cpu()), bed_z,
        )


# The loader supports either a file with a monotonic time column or a row-aligned
# series. It interpolates observations only within the available time range and
# never extrapolates synthetic targets beyond the measurements.

def load_gauge_observations(filepath: str, model_times: np.ndarray, *, dtype, device):
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Gauge file not found: {path}")
    data = np.loadtxt(path, skiprows=GAUGE_FILE_SKIP_ROWS)
    if data.ndim == 1:
        data = data[None, :]
    if data.shape[1] <= max(GAUGE_COLUMNS.values()):
        raise ValueError("Gauge file has too few columns.")

    times = np.asarray(data[:, 0], dtype=np.float64)
    has_times = times.size >= 2 and np.all(np.isfinite(times)) and np.all(np.diff(times) > 0)
    targets: Dict[str, torch.Tensor] = {}
    for name, column in GAUGE_COLUMNS.items():
        values = np.asarray(data[:, column], dtype=np.float64)  # already metres
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Gauge {name} contains non-finite values.")
        if has_times:
            if model_times[0] < times[0] - 1e-9 or model_times[-1] > times[-1] + 1e-9:
                raise ValueError(f"Model times fall outside the observation period for {name}.")
            values = np.interp(model_times, times, values)
        elif values.size >= model_times.size:
            values = values[:model_times.size]
        else:
            raise ValueError(f"Gauge {name} has too few observations.")
        targets[name] = torch.as_tensor(values, dtype=dtype, device=device)
    return targets


# =====================================================================
# MODEL STATE AND FORWARD SIMULATION
# =====================================================================
# Momentum starts at zero. Water depth is obtained from the still-water surface
# and bed elevation, preserving the model convention eta = h + z_b.

def create_initial_state(model, *, dtype, device) -> torch.Tensor:
    bed = model.bed.detach().to(dtype=dtype, device=device)
    depth = torch.clamp(INITIAL_WATER_LEVEL - bed, min=MINIMUM_INITIAL_DEPTH)
    if depth.shape[0] == 1 and BATCH_SIZE > 1:
        depth = depth.expand(BATCH_SIZE, -1, -1, -1)
    zero = torch.zeros_like(depth)
    return torch.cat((depth, zero, zero), dim=1)


def apply_external_rainfall(state: torch.Tensor, rain_depth: torch.Tensor) -> torch.Tensor:
    return torch.cat((state[:, 0:1] + rain_depth, state[:, 1:2], state[:, 2:3]), dim=1)


# One dense spline value is held constant over each DT_OUT interval. The closure
# accepts both tensors explicitly so non-reentrant checkpointing can reconstruct
# the same computation graph safely during backward.

def run_model_chunk(model, state, boundary_series, index, start_time, duration, use_checkpointing):
    index, start_time, duration = int(index), float(start_time), float(duration)

    def step_function(state_in, series_in):
        state_out, _ = model(
            state_in,
            t_end=duration,
            start_time=start_time,
            boundary_level=series_in[index],
        )
        return state_out

    if use_checkpointing:
        return checkpoint(
            step_function, state, boundary_series,
            use_reentrant=False, preserve_rng_state=False,
        )
    return step_function(state, boundary_series)


# All three gauges are extracted in one forward run. Only the subsequent loss
# function decides which gauges are used for training and which are held out.

def simulate_boundary_series(
    *, model, boundary_series, gauge_names, initial_state, use_txt_rainfall,
    rain_times, rain_rate_ms, use_checkpointing, logger=None, log_progress=False,
):
    number_of_steps = boundary_series.numel()
    state = initial_state
    output = {name: [] for name in gauge_names}

    for step_index in range(number_of_steps):
        start_time = step_index * DT_OUT
        if use_txt_rainfall:
            if rain_times is None or rain_rate_ms is None:
                raise RuntimeError("Text rainfall enabled without rainfall arrays.")
            state = apply_external_rainfall(
                state, state.new_tensor(np.interp(start_time, rain_times, rain_rate_ms) * DT_OUT)
            )

        if log_progress and logger and (
            step_index % FORWARD_LOG_INTERVAL == 0 or step_index == number_of_steps - 1
        ):
            logger.info(
                "Forward output %d/%d: %.2f s to %.2f s",
                step_index + 1, number_of_steps, start_time, start_time + DT_OUT,
            )

        state = run_model_chunk(
            model, state, boundary_series, step_index, start_time, DT_OUT, use_checkpointing
        )

        if STATE_GRADIENT_CLIP is not None and state.requires_grad:
            limit, chunk = float(STATE_GRADIENT_CLIP), step_index + 1

            def clip_gradient(gradient, chunk_number=chunk):
                if not torch.isfinite(gradient).all():
                    raise FloatingPointError(f"Non-finite adjoint at output chunk {chunk_number}.")
                return gradient.clamp(-limit, limit)

            # Hooks operate only during backward and do not alter the validated forward
            # solution. They limit adjoint growth between outer time chunks.
            state.register_hook(clip_gradient)

        if STATE_CHECK_INTERVAL > 0 and (
            step_index % STATE_CHECK_INTERVAL == 0 or step_index == number_of_steps - 1
        ) and not torch.isfinite(state).all():
            raise FloatingPointError(f"Non-finite SWE state at output step {step_index + 1}.")

        for name in gauge_names:
            iy, ix = model.gauge_indices[name]
            depth = state[0, 0, iy, ix]
            if OBSERVATION_TYPE == "water_level":
                value = depth + model.bed[0, 0, iy, ix]
            elif OBSERVATION_TYPE == "depth":
                value = depth
            else:
                raise ValueError("OBSERVATION_TYPE must be 'water_level' or 'depth'.")
            output[name].append(value)

    return {name: torch.stack(values) for name, values in output.items()}


# =====================================================================
# LOSSES AND METRICS
# =====================================================================
# These penalties act on the dense reconstructed hydrograph, not solely on the
# controls. This makes their meaning independent of the chosen control count.

def boundary_regularization(series: torch.Tensor):
    d1 = series[1:] - series[:-1]
    d2 = series[2:] - 2.0 * series[1:-1] + series[:-2]
    return d1.square().mean(), d2.square().mean(), series.square().mean()


def gauge_mse(simulated: Dict[str, torch.Tensor], observed: Dict[str, torch.Tensor], names):
    losses = {name: F.mse_loss(simulated[name], observed[name]) for name in names}
    return torch.stack(tuple(losses.values())).mean(), losses


# The held-out gauge is deliberately absent here. Regularization is added to the
# mean training-gauge MSE to form the scalar differentiated objective.

def calculate_training_objective(simulated, observed, boundary_series):
    # Only TRAIN_GAUGES appears in this data loss. TEST_GAUGES therefore has no
    # influence on gradients or optimizer updates.
    train_loss, train_gauge_losses = gauge_mse(simulated, observed, TRAIN_GAUGES)
    first, second, amplitude = boundary_regularization(boundary_series)
    total = (
        train_loss
        + FIRST_DERIVATIVE_WEIGHT * first
        + SECOND_DERIVATIVE_WEIGHT * second
        + AMPLITUDE_WEIGHT * amplitude
    )
    return total, {
        "train_data": train_loss,
        "first_derivative": first,
        "second_derivative": second,
        "amplitude": amplitude,
        "train_gauge_losses": train_gauge_losses,
    }


# Test loss is detached only when converted for logging; computing it in the
# graph is harmless, but it is never added to the differentiated objective.

def evaluate_test_loss(simulated, observed):
    return gauge_mse(simulated, observed, TEST_GAUGES)


# =====================================================================
# OUTPUT
# =====================================================================
# Checkpoints contain both representations of the boundary: compact controls for
# restarting and the dense series for immediate inspection and reproducibility.

def save_checkpoint(
    filepath: Path, *, epoch, raw_controls, control_values, boundary_series,
    optimizer, scheduler, train_loss, test_loss, best_train_loss, knots,
    control_times,
):
    torch.save({
        "epoch": epoch,
        "raw_spline_controls": raw_controls.detach().cpu().clone(),
        "spline_control_values": control_values.detach().cpu().clone(),
        "spline_control_times": control_times.detach().cpu().clone(),
        "spline_knots": knots.detach().cpu().clone(),
        "boundary_series": boundary_series.detach().cpu().clone(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "train_loss": train_loss,
        "test_loss": test_loss,
        "best_train_loss": best_train_loss,
        "train_gauges": TRAIN_GAUGES,
        "test_gauges": TEST_GAUGES,
        "dt_out": DT_OUT,
        "t_end": T_END,
        "spline_degree": SPLINE_DEGREE,
        "bc_minimum": BC_MINIMUM,
        "bc_maximum": BC_MAXIMUM,
    }, filepath)


# Text files are human-readable; NumPy history arrays preserve all epochs for
# plotting convergence and the evolution of both controls and dense hydrographs.

def save_final_results(
    *, model_times, best_boundary, best_controls, control_times,
    boundary_history, control_history, loss_history, simulated, observed,
):
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        OUTPUT_DIRECTORY / "optimized_boundary_condition.txt",
        np.column_stack((model_times, best_boundary.cpu().numpy())),
        header="time_s boundary_water_level_m", fmt="%.10e",
    )
    np.savetxt(
        OUTPUT_DIRECTORY / "optimized_spline_controls.txt",
        np.column_stack((control_times.cpu().numpy(), best_controls.cpu().numpy())),
        header="greville_time_s control_water_level_m", fmt="%.10e",
    )
    if boundary_history:
        np.save(OUTPUT_DIRECTORY / "boundary_evolution_history.npy", np.stack(boundary_history))
    if control_history:
        np.save(OUTPUT_DIRECTORY / "control_evolution_history.npy", np.stack(control_history))

    columns, headers = [model_times], ["time_s"]
    for name in observed:
        columns.extend((simulated[name].detach().cpu().numpy(), observed[name].cpu().numpy()))
        split = "train" if name in TRAIN_GAUGES else "test"
        headers.extend((f"{name}_{split}_simulated", f"{name}_{split}_observed"))
    np.savetxt(
        OUTPUT_DIRECTORY / "best_gauge_comparison.txt",
        np.column_stack(columns), header=" ".join(headers), fmt="%.10e",
    )

    fields = (
        "epoch", "total_loss", "train_loss", "test_loss", "first_derivative",
        "second_derivative", "amplitude", "gradient_norm", "learning_rate",
        "elapsed_seconds",
    )
    matrix = np.asarray([[record[field] for field in fields] for record in loss_history])
    np.savetxt(
        OUTPUT_DIRECTORY / "loss_history.txt", matrix,
        header=" ".join(fields), fmt="%.10e",
    )


def log_gauge_metrics(logger, simulated, observed):
    for name in observed:
        residual = simulated[name] - observed[name]
        rmse = residual.square().mean().sqrt()
        split = "TRAIN" if name in TRAIN_GAUGES else "TEST"
        logger.info(
            "%s %s | RMSE=%.6f m (%.4f cm) | bias=%.6f m | max=%.6f m",
            split, name, rmse.item(), 100.0 * rmse.item(),
            residual.mean().item(), residual.abs().max().item(),
        )


# =====================================================================
# MAIN
# =====================================================================
# The main routine follows a strict sequence: validate configuration, build the
# spline basis, load the model and observations, optimize on training gauges,
# restore the best training state, evaluate the held-out gauge, and save output.

def main() -> None:
    logger = setup_logger("bc_spline_inversion")
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    if not Path(DEM_FILEPATH).exists() or not Path(GAUGE_FILEPATH).exists():
        raise FileNotFoundError("DEM or gauge observation file is missing.")

    all_gauges = tuple(GAUGE_COORDINATES)
    if set(TRAIN_GAUGES) & set(TEST_GAUGES):
        raise ValueError("Training and test gauges must not overlap.")
    if set(TRAIN_GAUGES) | set(TEST_GAUGES) != set(all_gauges):
        raise ValueError("TRAIN_GAUGES and TEST_GAUGES must partition all configured gauges.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)
    logger.info("Using dtype: %s", MODEL_DTYPE)
    logger.info("Training gauges: %s | Test gauges: %s", TRAIN_GAUGES, TEST_GAUGES)
    if device.type == "cuda":
        logger.info("GPU: %s", torch.cuda.get_device_name(device))

    number_of_steps = int(round(T_END / DT_OUT))
    if not np.isclose(number_of_steps * DT_OUT, T_END, atol=1e-10, rtol=0.0):
        raise ValueError("T_END must be an integer multiple of DT_OUT.")
    model_times = np.arange(1, number_of_steps + 1, dtype=np.float64) * DT_OUT
    boundary_times = torch.arange(number_of_steps, dtype=MODEL_DTYPE, device=device) * DT_OUT
    basis, knots, control_times = build_clamped_bspline_basis(
        boundary_times, NUM_SPLINE_CONTROLS, SPLINE_DEGREE
    )
    logger.info(
        "Using clamped degree-%d B-spline with %d controls for %d BC intervals.",
        SPLINE_DEGREE, NUM_SPLINE_CONTROLS, number_of_steps,
    )

    model_grid = ModelGrid.from_bounds(
        xmin=GRID_X_MIN, xmax=GRID_X_MAX, ymin=GRID_Y_MIN, ymax=GRID_Y_MAX,
        resolution=GRID_RESOLUTION, dtype=MODEL_DTYPE, device=device,
    )
    model, use_txt_rainfall, rain_times, rain_rate_ms = load_dynamic_model(
        dem_filepath=DEM_FILEPATH,
        rain_filepath=RAIN_FILEPATH,
        model_grid=model_grid,
        device=device,
        boundary_left=BOUNDARY_LEFT,
        boundary_right=BOUNDARY_RIGHT,
        boundary_top=BOUNDARY_TOP,
        boundary_bottom=BOUNDARY_BOTTOM,
        frictionmodel=FRICTION_MODEL,
        roughness_filepath=ROUGHNESS_FILEPATH,
        roughness_var_name=ROUGHNESS_VARIABLE,
        default_roughness_value=DEFAULT_ROUGHNESS_VALUE,
        tide_filepath=TIDE_FILEPATH,
    )
    model = model.to(device=device, dtype=MODEL_DTYPE)
    if hasattr(model, "dem_correction_parameter"):
        model.dem_correction_parameter.requires_grad_(False)
    if hasattr(model, "clear_dynamic_boundary"):
        model.clear_dynamic_boundary()
    else:
        model.use_dynamic_bc, model.bc_times, model.bc_wls = False, None, None
    model.train()
    if USE_TORCH_COMPILE:
        model = torch.compile(model, dynamic=False)

    configure_gauges(model, model_grid, logger)
    observed = load_gauge_observations(
        GAUGE_FILEPATH, model_times, dtype=MODEL_DTYPE, device=device
    )
    initial_state = create_initial_state(model, dtype=MODEL_DTYPE, device=device)
    if not torch.isfinite(initial_state).all():
        raise FloatingPointError("Initial state contains non-finite values.")

    wet_fraction = (INITIAL_BC_VALUE > model.bed[0, 0, :, 0].detach()).float().mean().item()
    logger.info("Initial wet fraction of left boundary: %.2f%%", 100.0 * wet_fraction)

    initial_raw = inverse_bounded_sigmoid(
        INITIAL_BC_VALUE, BC_MINIMUM, BC_MAXIMUM,
        dtype=MODEL_DTYPE, device=device,
    )
    raw_controls = torch.nn.Parameter(initial_raw.repeat(NUM_SPLINE_CONTROLS))
    optimizer = optim.Adam([raw_controls], lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=LR_SCHEDULER_FACTOR,
        patience=LR_SCHEDULER_PATIENCE, threshold=EARLY_STOPPING_MIN_DELTA,
        threshold_mode="abs", min_lr=MINIMUM_LEARNING_RATE,
    )

    best_train_loss, best_epoch, best_raw_controls = float("inf"), 0, None
    patience_counter = 0
    boundary_history, control_history, loss_history = [], [], []

    for epoch_index in range(MAX_EPOCHS):
        epoch, started = epoch_index + 1, time.perf_counter()
        optimizer.zero_grad(set_to_none=True)

        def forward_and_loss():
            boundary, controls = spline_boundary_series(raw_controls, basis)
            simulation = simulate_boundary_series(
                model=model,
                boundary_series=boundary,
                gauge_names=all_gauges,
                initial_state=initial_state,
                use_txt_rainfall=use_txt_rainfall,
                rain_times=rain_times,
                rain_rate_ms=rain_rate_ms,
                use_checkpointing=USE_CHECKPOINTING,
                logger=logger,
                log_progress=LOG_FIRST_EPOCH_FORWARD_PROGRESS and epoch_index == 0,
            )
            objective, parts = calculate_training_objective(simulation, observed, boundary)
            test_loss, test_parts = evaluate_test_loss(simulation, observed)
            return boundary, controls, simulation, objective, parts, test_loss, test_parts

        if ENABLE_AUTOGRAD_ANOMALY_DETECTION:
            with torch.autograd.detect_anomaly(check_nan=True):
                boundary, controls, simulated, total_loss, parts, test_loss, _ = forward_and_loss()
                total_loss.backward()
        else:
            boundary, controls, simulated, total_loss, parts, test_loss, _ = forward_and_loss()
            total_loss.backward()

        if not torch.isfinite(total_loss) or raw_controls.grad is None:
            raise FloatingPointError("Non-finite loss or missing spline-control gradient.")
        if not torch.isfinite(raw_controls.grad).all():
            raise FloatingPointError("Spline-control gradient contains non-finite values.")

        train_loss_value = float(parts["train_data"].detach().cpu())
        test_loss_value = float(test_loss.detach().cpu())
        total_loss_value = float(total_loss.detach().cpu())
        gradient_norm = float(torch.linalg.vector_norm(raw_controls.grad).detach().cpu())
        max_gradient = float(raw_controls.grad.detach().abs().max().cpu())
        if gradient_norm == 0.0:
            raise RuntimeError("All spline-control gradients are zero.")

        # Save the best parameter state that produced the evaluated loss.
        # Select the best state using training loss only. The test gauge remains a true
        # held-out diagnostic rather than becoming an implicit tuning target.
        improved = train_loss_value < best_train_loss - EARLY_STOPPING_MIN_DELTA
        if improved:
            best_train_loss, best_epoch = train_loss_value, epoch
            best_raw_controls = raw_controls.detach().clone()
            patience_counter = 0
            save_checkpoint(
                OUTPUT_DIRECTORY / "best_inverse_checkpoint.pth",
                epoch=epoch,
                raw_controls=best_raw_controls,
                control_values=controls,
                boundary_series=boundary,
                optimizer=optimizer,
                scheduler=scheduler,
                train_loss=train_loss_value,
                test_loss=test_loss_value,
                best_train_loss=best_train_loss,
                knots=knots,
                control_times=control_times,
            )
        else:
            patience_counter += 1

        # Parameter-gradient clipping is applied after best-state evaluation and before
        # Adam updates. It protects against rare long-horizon adjoint spikes.
        torch.nn.utils.clip_grad_norm_([raw_controls], MAX_GRADIENT_NORM, error_if_nonfinite=True)
        optimizer.step()
        scheduler.step(train_loss_value)

        updated_boundary, updated_controls = spline_boundary_series(raw_controls, basis)
        boundary_history.append(updated_boundary.detach().cpu().numpy().copy())
        control_history.append(updated_controls.detach().cpu().numpy().copy())
        elapsed = time.perf_counter() - started
        record = {
            "epoch": epoch,
            "total_loss": total_loss_value,
            "train_loss": train_loss_value,
            "test_loss": test_loss_value,
            "first_derivative": float(parts["first_derivative"].detach().cpu()),
            "second_derivative": float(parts["second_derivative"].detach().cpu()),
            "amplitude": float(parts["amplitude"].detach().cpu()),
            "gradient_norm": gradient_norm,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "elapsed_seconds": elapsed,
        }
        loss_history.append(record)

        if epoch % CHECKPOINT_SAVE_INTERVAL == 0:
            save_checkpoint(
                OUTPUT_DIRECTORY / "latest_inverse_checkpoint.pth",
                epoch=epoch,
                raw_controls=raw_controls,
                control_values=updated_controls,
                boundary_series=updated_boundary,
                optimizer=optimizer,
                scheduler=scheduler,
                train_loss=train_loss_value,
                test_loss=test_loss_value,
                best_train_loss=best_train_loss,
                knots=knots,
                control_times=control_times,
            )

        logger.info(
            "Epoch %03d | train=%.6e | test=%.6e | total=%.6e | "
            "grad=%.3e | max|grad|=%.3e | lr=%.3e | controls=[%.4f, %.4f] m | "
            "BC=[%.4f, %.4f] m | patience=%d/%d | %.2f s",
            epoch, train_loss_value, test_loss_value, total_loss_value,
            gradient_norm, max_gradient, optimizer.param_groups[0]["lr"],
            updated_controls.min().item(), updated_controls.max().item(),
            updated_boundary.min().item(), updated_boundary.max().item(),
            patience_counter, EARLY_STOPPING_PATIENCE, elapsed,
        )
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            logger.info("Early stopping at epoch %d; best epoch=%d.", epoch, best_epoch)
            break

    if best_raw_controls is None:
        raise RuntimeError("Training did not produce a valid best state.")
    with torch.no_grad():
        raw_controls.copy_(best_raw_controls)
    best_boundary, best_controls = spline_boundary_series(raw_controls, basis)
    best_boundary, best_controls = best_boundary.detach(), best_controls.detach()

    logger.info("Running final simulation using best training epoch %d.", best_epoch)
    final_simulated = simulate_boundary_series(
        model=model,
        boundary_series=best_boundary,
        gauge_names=all_gauges,
        initial_state=initial_state,
        use_txt_rainfall=use_txt_rainfall,
        rain_times=rain_times,
        rain_rate_ms=rain_rate_ms,
        use_checkpointing=False,
    )
    final_objective, final_parts = calculate_training_objective(
        final_simulated, observed, best_boundary
    )
    final_test_loss, _ = evaluate_test_loss(final_simulated, observed)
    log_gauge_metrics(logger, final_simulated, observed)
    logger.info(
        "Final train MSE=%.8e | held-out test MSE=%.8e | objective=%.8e",
        float(final_parts["train_data"].cpu()),
        float(final_test_loss.cpu()),
        float(final_objective.cpu()),
    )

    save_final_results(
        model_times=model_times,
        best_boundary=best_boundary,
        best_controls=best_controls,
        control_times=control_times,
        boundary_history=boundary_history,
        control_history=control_history,
        loss_history=loss_history,
        simulated=final_simulated,
        observed=observed,
    )
    save_checkpoint(
        OUTPUT_DIRECTORY / "final_inverse_checkpoint.pth",
        epoch=best_epoch,
        raw_controls=raw_controls,
        control_values=best_controls,
        boundary_series=best_boundary,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loss=float(final_parts["train_data"].cpu()),
        test_loss=float(final_test_loss.cpu()),
        best_train_loss=best_train_loss,
        knots=knots,
        control_times=control_times,
    )
    logger.info("Inversion complete. Results saved to %s", OUTPUT_DIRECTORY.resolve())


if __name__ == "__main__":
    main()
