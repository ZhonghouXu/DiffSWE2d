import torch

from .numerics import limited_increment
from .riemann import primitive


def _next(q, dim):
    """Neighbouring value on the positive side of each face."""
    return torch.roll(q, shifts=-1, dims=dim)


def _depth_floor(U, epsilon, dry_depth):
    """Positive floor for stable wet/dry gradients."""
    precision_floor = 1.0e-6 if U.dtype == torch.float32 else 1.0e-10
    return max(float(epsilon), float(dry_depth), precision_floor)


def bflood_cn_reconstruction(
    U, bed, dim, gravity, epsilon, dry_depth,
    theta=1.3, water_slope_reset=True,
):
    """B-Flood combined CN hydrostatic reconstruction."""
    h, u, v, _ = primitive(U, gravity, epsilon, dry_depth)
    eta = h + bed

    dh = limited_increment(h, dim, theta)
    deta = limited_increment(eta, dim, theta)
    dz = limited_increment(bed, dim, theta)
    du = limited_increment(u, dim, theta)
    dv = limited_increment(v, dim, theta)

    if water_slope_reset:
        eta_minus = torch.roll(eta, 1, dim)
        eta_plus = torch.roll(eta, -1, dim)
        bed_minus = eta - h - 0.5 * (deta - dh)
        bed_plus = eta - h + 0.5 * (deta - dh)

        reset = (
            (deta.square() > dz.square())
            & ((bed_minus > eta_minus) | (bed_plus > eta_plus))
        )
        deta = torch.where(reset, dh + dz, deta)

    next_h = _next(h, dim)
    next_bed = _next(bed, dim)
    next_dh = _next(dh, dim)
    next_deta = _next(deta, dim)
    next_dz = _next(dz, dim)

    h_linear_L = h + 0.5 * dh
    h_linear_R = next_h - 0.5 * next_dh

    eta_L = eta + 0.5 * deta
    eta_R = _next(eta, dim) - 0.5 * next_deta

    bed_L = bed + 0.5 * dz
    bed_R = next_bed - 0.5 * next_dz

    bed_audusse = torch.maximum(bed_L, bed_R)
    bed_cn = torch.minimum(bed_audusse, torch.minimum(eta_L, eta_R))

    h_cn_L = torch.clamp(
        torch.minimum(eta_L - bed_cn, h_linear_L),
        min=0.0,
    )
    h_cn_R = torch.clamp(
        torch.minimum(eta_R - bed_cn, h_linear_R),
        min=0.0,
    )

    floor = _depth_floor(U, epsilon, dry_depth)
    safe_h_L = torch.clamp(h, min=floor)
    safe_h_R = torch.clamp(next_h, min=floor)

    factor_L_raw = 1.0 - 0.5 * dh / safe_h_L
    factor_R_raw = 1.0 + 0.5 * next_dh / safe_h_R

    factor_L = torch.where(
        h > dry_depth,
        factor_L_raw,
        torch.ones_like(h),
    )
    factor_R = torch.where(
        next_h > dry_depth,
        factor_R_raw,
        torch.ones_like(next_h),
    )

    # Prevent extreme adjoints near moving wet/dry fronts.
    factor_L = torch.clamp(factor_L, -10.0, 10.0)
    factor_R = torch.clamp(factor_R, -10.0, 10.0)

    u_L = u + 0.5 * factor_L * du
    v_L = v + 0.5 * factor_L * dv
    u_R = _next(u, dim) - 0.5 * factor_R * _next(du, dim)
    v_R = _next(v, dim) - 0.5 * factor_R * _next(dv, dim)

    UL = torch.cat((h_cn_L, h_cn_L * u_L, h_cn_L * v_L), dim=1)
    UR = torch.cat((h_cn_R, h_cn_R * u_R, h_cn_R * v_R), dim=1)

    source_L = (
        0.5 * gravity * (h + h_cn_L) * (bed - bed_cn)
    )
    source_R = (
        0.5 * gravity * (h_cn_R + next_h) * (next_bed - bed_cn)
    )

    return UL, UR, source_L, source_R