"""DEM inversion while roughness length and rainfall remain fixed."""
import torch
from diffswe2d_final import SWE2D, SWEConfig


torch.set_default_dtype(torch.float64)
device = "cuda" if torch.cuda.is_available() else "cpu"

model = SWE2D.from_netcdf(
    "dem_and_roughness.nc",
    "rainfall.nc",
    dem_variable="elevation",
    roughness_variable="z0",
    rainfall_variable="rainfall",
    config=SWEConfig(cfl=0.25, rainfall_outside_domain="zero"),
    device=device,
    train_dem=True,
    maximum_dem_correction=1.0,
)

ny, nx = model.bed.shape[-2:]
U0 = torch.zeros(1, 3, ny, nx, device=device)
U0[:, 0:1] = torch.clamp(1.0 - model.bed.detach(), min=0.0)

# Replace these placeholders with observations on the DEM grid.
observed_depth = torch.zeros(1, 1, ny, nx, device=device)
observation_mask = model.valid_mask.to(dtype=observed_depth.dtype)

optimizer = torch.optim.Adam([model.dem_correction_parameter], lr=1.0e-3)
for epoch in range(100):
    optimizer.zero_grad(set_to_none=True)
    prediction = model(U0, t_end=1800.0)
    residual = (prediction[:, 0:1] - observed_depth) * observation_mask
    data_loss = residual.square().sum() / observation_mask.sum().clamp_min(1.0)
    regularization = model.dem_regularization(
        prior_weight=1.0e-3,
        slope_weight=1.0e-4,
        curvature_weight=1.0e-4,
    )
    loss = data_loss + regularization
    loss.backward()
    torch.nn.utils.clip_grad_norm_([model.dem_correction_parameter], 10.0)
    optimizer.step()
    print(f"{epoch:04d} loss={loss.item():.6e} data={data_loss.item():.6e}")
