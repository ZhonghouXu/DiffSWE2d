import torch


def primitive(U, gravity: float, epsilon: float, dry_depth: float):
    """Convert [h, hu, hv] into h, u, v, and gravity-wave celerity."""
    h = torch.clamp(U[:, 0:1], min=0.0)
    wet = h > dry_depth
    denominator = torch.clamp(h, min=epsilon)
    u = torch.where(wet, U[:, 1:2] / denominator, torch.zeros_like(h))
    v = torch.where(wet, U[:, 2:3] / denominator, torch.zeros_like(h))
    c = torch.sqrt(gravity * h)
    return h, u, v, c


def bflood_hllc_flux(UL, UR, normal: str, gravity: float,
                      epsilon: float, dry_depth: float):
    """Basilisk/B-Flood routine named HLLC, including dry-front wave speeds.

    The upstream routine uses two bounding waves and an HLL expression in the
    star region. We keep its original HLLC name for numerical fidelity.
    Transverse momentum is upwinded using the sign of the mass flux.
    """
    hL, uL, vL, cL = primitive(UL, gravity, epsilon, dry_depth)
    hR, uR, vR, cR = primitive(UR, gravity, epsilon, dry_depth)

    unL, utL = (uL, vL) if normal == "x" else (vL, uL)
    unR, utR = (uR, vR) if normal == "x" else (vR, uR)

    u_star = 0.5 * (unL + unR) + cL - cR
    c_star = 0.5 * (cL + cR) + 0.25 * (unL - unR)

    SL_wet = torch.minimum(unL - cL, u_star - c_star)
    SR_wet = torch.maximum(unR + cR, u_star + c_star)
    SL = torch.where(hL <= dry_depth, unR - 2.0 * cR, SL_wet)
    SR = torch.where(hR <= dry_depth, unL + 2.0 * cL, SR_wet)

    fhL = hL * unL
    fhR = hR * unR
    fqL = hL * (unL.square() + 0.5 * gravity * hL)
    fqR = hR * (unR.square() + 0.5 * gravity * hR)

    denominator = torch.clamp(SR - SL, min=epsilon)
    fh_star = (SR * fhL - SL * fhR + SL * SR * (hR - hL)) / denominator
    fq_star = (
        SR * fqL - SL * fqR
        + SL * SR * (hR * unR - hL * unL)
    ) / denominator

    fh = torch.where(SL >= 0.0, fhL, torch.where(SR <= 0.0, fhR, fh_star))
    fq = torch.where(SL >= 0.0, fqL, torch.where(SR <= 0.0, fqR, fq_star))
    ft = torch.where(fh > 0.0, utL, utR) * fh

    if normal == "x":
        flux = torch.cat((fh, fq, ft), dim=1)
    elif normal == "y":
        flux = torch.cat((fh, ft, fq), dim=1)
    else:
        raise ValueError("normal must be 'x' or 'y'")

    both_dry = (hL <= dry_depth) & (hR <= dry_depth)
    flux = torch.where(both_dry.expand_as(flux), torch.zeros_like(flux), flux)
    maximum_wave_speed = torch.maximum(torch.abs(SL), torch.abs(SR))
    return flux, maximum_wave_speed
