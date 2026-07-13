import torch
from .numerics import limited_increment
from .riemann import primitive


def _next(q, dim):
    """Value in the neighbouring cell on the positive side of each face."""
    return torch.roll(q, shifts=-1, dims=dim)


def bflood_cn_reconstruction(U, bed, dim: int, gravity: float,
                             epsilon: float, dry_depth: float,
                             theta: float = 1.3,
                             water_slope_reset: bool = True):
    """B-Flood combined CN hydrostatic reconstruction.

    For every positive-direction face, the current cell is left/bottom and the
    rolled cell is right/top. The function returns reconstructed conservative
    states plus the two asymmetric bed-pressure corrections used by the cells
    sharing that face.
    """
    h, u, v, _ = primitive(U, gravity, epsilon, dry_depth)
    eta = h + bed

    dh = limited_increment(h, dim, theta)
    deta = limited_increment(eta, dim, theta)
    dz = limited_increment(bed, dim, theta)
    du = limited_increment(u, dim, theta)
    dv = limited_increment(v, dim, theta)

    if water_slope_reset:
        eta_minus = torch.roll(eta, shifts=1, dims=dim)
        eta_plus = torch.roll(eta, shifts=-1, dims=dim)
        implied_bed_minus = eta - h - 0.5 * (deta - dh)
        implied_bed_plus = eta - h + 0.5 * (deta - dh)
        reset = (
            (deta.square() > dz.square())
            & ((implied_bed_minus > eta_minus) | (implied_bed_plus > eta_plus))
        )
        deta = torch.where(reset, dh + dz, deta)

    h_linear_L = h + 0.5 * dh
    eta_L = eta + 0.5 * deta
    bed_L = bed + 0.5 * dz

    h_cell_R = _next(h, dim)
    bed_cell_R = _next(bed, dim)
    h_linear_R = h_cell_R - 0.5 * _next(dh, dim)
    eta_R = _next(eta, dim) - 0.5 * _next(deta, dim)
    bed_R = bed_cell_R - 0.5 * _next(dz, dim)

    bed_audusse = torch.maximum(bed_L, bed_R)
    bed_cn = torch.minimum(bed_audusse, torch.minimum(eta_L, eta_R))
    h_cn_L = torch.clamp(torch.minimum(eta_L - bed_cn, h_linear_L), min=0.0)
    h_cn_R = torch.clamp(torch.minimum(eta_R - bed_cn, h_linear_R), min=0.0)

    # Bouchut-type velocity reconstruction near wet/dry interfaces.
    factor_L = torch.where(
        h > dry_depth,
        1.0 - 0.5 * dh / torch.clamp(h, min=epsilon),
        torch.ones_like(h),
    )
    factor_R = torch.where(
        h_cell_R > dry_depth,
        1.0 + 0.5 * _next(dh, dim) / torch.clamp(h_cell_R, min=epsilon),
        torch.ones_like(h_cell_R),
    )

    u_L = u + 0.5 * factor_L * du
    v_L = v + 0.5 * factor_L * dv
    u_R = _next(u, dim) - 0.5 * factor_R * _next(du, dim)
    v_R = _next(v, dim) - 0.5 * factor_R * _next(dv, dim)

    UL = torch.cat((h_cn_L, h_cn_L * u_L, h_cn_L * v_L), dim=1)
    UR = torch.cat((h_cn_R, h_cn_R * u_R, h_cn_R * v_R), dim=1)

    source_L = 0.5 * gravity * (h + h_cn_L) * (bed - bed_cn)
    source_R = 0.5 * gravity * (h_cn_R + h_cell_R) * (bed_cell_R - bed_cn)
    return UL, UR, source_L, source_R
