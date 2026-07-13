# diffswe2d_conservative

A PyTorch-based, piecewise-differentiable two-dimensional shallow-water
equations solver for flood simulation and DEM inversion. The solver uses a
structured square-cell model grid, B-Flood-inspired finite-volume numerics,
and precipitation-volume-conserving rainfall remapping.

## Overview

**diffswe2d_conservative** implements a vectorized 2-D shallow-water solver
on a uniform Cartesian grid with square cells (`dx == dy`). It is compatible
with PyTorch automatic differentiation and can run on either CPUs or
CUDA-capable GPUs.

The discrete solver is piecewise differentiable rather than smoothly
differentiable everywhere. Gradients can propagate through the forward
simulation to the DEM correction parameter, but limiters, wet/dry decisions,
flux branches, positivity treatment, and other hard numerical switches can
introduce discontinuous or zero-gradient regions.

The package is intended for research, forward hydraulic simulations,
sensitivity analysis, and gradient-based DEM inversion. It should be
validated against analytical solutions and established hydraulic benchmarks
before operational use.

## Key Features

### Numerical methods

- **B-Flood-inspired finite-volume solver**
  - Basilisk/B-Flood HLLC-named (Harten-Lax-van Leer-Contact) approximate Riemann solver (Godunov-type upwind scheme)
  - B-Flood-style reconstruction: MUSCL-type second-order spatial reconstruction scheme (using minmod slope limiter, theta=1.3), modified by Audusse/CN hydrostatic reconstruction for abrupt topography
  - water-surface slope reset near abrupt topography
  - Bouchut-style velocity reconstruction near wet/dry fronts
  - asymmetric face-based topographic pressure corrections  
  - dry-front wave-speed estimates
  - transverse-momentum upwinding based on mass-flux direction
  - midpoint predictor-corrector time integration
  - adaptive CFL timestep based on Riemann wave speeds
  - practical depth-positivity safeguard and dry-cell momentum cleanup

### Bottom friction

- **BG_Flood-style roughness-length friction**
  - spatially varying roughness length `z0`
  - roughness length read from the same NetCDF file and grid as the DEM
  - piecewise shallow and logarithmic-law drag-coefficient relations
  - semi-implicit velocity update
  - roughness length remains fixed during DEM inversion

The current package does not implement Manning's `n` as an alternative
friction model.

### Rainfall forcing

- Rainfall read from a separate NetCDF file
- Rainfall source grid may differ from both the DEM grid and model grid
- Linear temporal interpolation between rainfall records
- Predictor-stage rainfall evaluated at the beginning of each timestep
- Corrector-stage rainfall evaluated at the midpoint of each timestep
- Area-overlap conservative spatial remapping to the model grid
- Explicit handling of model cells outside rainfall coverage

### Custom model grid

- Independent of both the DEM and rainfall source grids
- Uniform square cells with `dx == dy`
- Defined using outer domain bounds and one model resolution
- Domain dimensions must be exactly divisible by the requested resolution
- Model coordinates and input files must use the same projected CRS

### NetCDF input

- DEM and roughness length loaded from one NetCDF file
- Rainfall loaded from a separate NetCDF file
- DEM and roughness must share coordinates, resolution, and extent
- Rainfall may have different rows, columns, resolution, alignment, and extent
- Linear DEM remapping to the model grid
- Log-space roughness-length remapping to preserve positivity
- Conservative area-overlap rainfall remapping

### Differentiable DEM

- Reference DEM retained as a fixed model buffer
- Trainable DEM correction defined on the custom model grid
- Optional bounded correction using a hyperbolic tangent transformation
- Built-in prior, slope, and curvature regularization
- Compatible with PyTorch optimizers and gradient clipping

## Differentiability

The solver is end-to-end autograd-compatible with respect to the model-grid
DEM correction. It is piecewise differentiable because it contains hard
numerical operations such as:

- `minimum` and `maximum`
- `clamp`
- `where`
- minmod limiting
- wet/dry classification
- positivity limiting
- HLLC branch selection

The adaptive CFL timestep and resulting timestep count are intentionally
detached from the autograd graph. Gradients therefore describe the discrete
forward simulation for the selected timestep sequence, but do not include the
derivative of timestep selection itself.

## Conservative rainfall remapping

Rainfall values are treated as cell-average intensities. For every target
model cell \(T\),

\[
R_T =
\frac{\sum_S R_S A_{S \cap T}}{A_T},
\]

where \(R_S\) is the source-cell rainfall rate and \(A_{S \cap T}\) is the
overlap area between source cell \(S\) and target cell \(T\).

This conserves

\[
\sum R A
\]

over the spatial overlap between the rainfall and model grids for each
rainfall time record, up to floating-point precision.

- If the model grid covers the complete rainfall grid, the total source
  precipitation rate-volume is conserved.
- If the model domain is smaller, rainfall outside the model domain is
  intentionally excluded.
- If the model domain is larger and `rainfall_outside_domain="zero"`,
  uncovered model cells receive zero rainfall.
- If `rainfall_outside_domain="error"`, loading fails when the model grid is
  not fully covered by the rainfall grid.

## Core components

### `SWE2D`

The main `torch.nn.Module` simulation engine:

- computes B-Flood-style finite-volume tendencies
- performs midpoint predictor-corrector integration
- applies rainfall and roughness-length friction
- exposes a differentiable DEM correction
- supports forward simulations and DEM inversion

### `SWEConfig`

Controls model parameters and runtime options:

- `gravity`: gravitational acceleration
- `dry_depth`: wet/dry depth threshold
- `cfl`: Courant number
- `limiter_theta`: slope-limiter parameter
- `boundary`: `"wall"`, `"transmissive"`, or `"periodic"`
- `water_slope_reset`: enable B-Flood water-slope reset
- `positivity_limiter`: enable depth-positivity safeguard
- `apply_friction`: enable roughness-length bottom friction
- `rainfall_outside_domain`: `"zero"` or `"error"`

### `ModelGrid`

Defines the independent model grid:

- outer-edge domain bounds
- one square-cell resolution
- cell-centre coordinate arrays
- an exact `dx == dy` requirement

### `RainfallForcing`

Stores rainfall remapped to the model grid:

- rates with shape `[time, 1, ny, nx]`
- units of metres per second
- linear temporal interpolation
- constant endpoint values outside the input time range

## Scope and limitations

The code is a research implementation rather than a validated operational
flood model. Current limitations include:

- piecewise rather than globally smooth differentiation
- uniform Cartesian square cells only
- fixed roughness length during inversion
- no adaptive mesh refinement
- no conservative DEM remapping
- no CRS reprojection
- no rainfall infiltration model beyond an optional prescribed rate
- no built-in uncertainty-quantification workflow
- no demonstrated performance benchmark against B-Flood or BG_Flood