import math
import torch

def _floor(x, epsilon, dry_depth=0.0):
    """Positive floor suitable for the tensor precision."""
    precision = 1e-5 if x.dtype in (torch.float16, torch.bfloat16) else (
        1e-6 if x.dtype == torch.float32 else 1e-10
    )
    return max(float(epsilon), float(dry_depth), precision)


def _velocity_and_speed(h, hu, hv, dry_depth, epsilon):
    """
    Calculate wet-cell velocity and speed without sqrt(0).

    Momentum is masked before division, and sqrt receives a strictly
    positive argument. Dry or stationary cells retain zero speed.
    """
    wet = h > dry_depth
    safe_h = torch.clamp(h, min=_floor(h, epsilon, dry_depth))
    zero = torch.zeros_like(h)

    u = torch.where(wet, hu, zero) / safe_h
    v = torch.where(wet, hv, zero) / safe_h

    speed_squared = u.square() + v.square()
    moving = wet & (speed_squared > 0.0)
    safe_speed_squared = torch.clamp(
        speed_squared,
        min=_floor(h, epsilon) ** 2,
    )
    speed = torch.where(
        moving,
        torch.sqrt(safe_speed_squared),
        zero,
    )

    return wet, safe_h, speed


def smart_friction_coefficient(h, roughness_length, epsilon=1e-12):
    """
    BG_Flood smart-friction coefficient based on h/z0.

    The shallow branch avoids the singular logarithmic-law region.
    """
    floor = _floor(h, epsilon)
    z0 = torch.clamp(roughness_length, min=floor)
    ratio = torch.clamp(h, min=floor) / z0

    shallow = 1.0 / torch.clamp(0.46 * ratio, min=floor)
    log_term = 2.5 * (
        torch.log(torch.clamp(ratio, min=floor))
        - 1.0
        + 1.359 / torch.clamp(ratio, min=floor)
    )
    deep = 1.0 / torch.clamp(log_term.square(), min=floor)

    return torch.where(ratio < math.e, shallow, deep)


def manning_friction_coefficient(h, n, epsilon=1e-12):
    """Manning drag coefficient: Cf = g n² / h^(1/3)."""
    safe_h = torch.clamp(h, min=_floor(h, epsilon))
    safe_n = torch.clamp(n, min=0.0)
    return 9.81 * safe_n.square() / safe_h.pow(1.0 / 3.0)


def apply_roughness_length_friction(
    U,
    roughness_length,
    dt,
    dry_depth=1e-6,
    epsilon=1e-12,
):
    """
    Semi-implicit BG_Flood roughness-length friction update.

    Depth is unchanged. Momentum is scaled by:
        1 / (1 + Cf |u| dt / h)
    """
    h = torch.clamp(U[:, 0:1], min=0.0)
    hu, hv = U[:, 1:2], U[:, 2:3]

    wet, safe_h, speed = _velocity_and_speed(
        h, hu, hv, dry_depth, epsilon
    )
    cf = smart_friction_coefficient(h, roughness_length, epsilon)
    factor = 1.0 / (1.0 + cf * speed * dt / safe_h)

    zero = torch.zeros_like(h)
    return torch.cat(
        (
            h,
            torch.where(wet, hu * factor, zero),
            torch.where(wet, hv * factor, zero),
        ),
        dim=1,
    )


def apply_manning_friction(
    U,
    n,
    dt,
    dry_depth=1e-6,
    epsilon=1e-12,
):
    """
    Semi-implicit Manning-friction update.

    Depth is unchanged. Momentum is scaled by:
        1 / (1 + Cf |u| dt / h)
    """
    h = torch.clamp(U[:, 0:1], min=0.0)
    hu, hv = U[:, 1:2], U[:, 2:3]

    wet, safe_h, speed = _velocity_and_speed(
        h, hu, hv, dry_depth, epsilon
    )
    cf = manning_friction_coefficient(h, n, epsilon)
    factor = 1.0 / (1.0 + cf * speed * dt / safe_h)

    zero = torch.zeros_like(h)
    return torch.cat(
        (
            h,
            torch.where(wet, hu * factor, zero),
            torch.where(wet, hv * factor, zero),
        ),
        dim=1,
    )