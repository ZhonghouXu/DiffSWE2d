from __future__ import annotations

import torch


def _depth_floor(x, epsilon, dry_depth):
    """Strictly positive depth floor for stable divisions and sqrt."""
    precision = 1e-5 if x.dtype in (torch.float16, torch.bfloat16) else (
        1e-6 if x.dtype == torch.float32 else 1e-10
    )
    return max(float(epsilon), float(dry_depth), precision)


def _safe_denominator(x, epsilon):
    """Positive floor for the non-negative HLL denominator SR - SL."""
    precision = 1e-6 if x.dtype == torch.float32 else 1e-10
    return torch.clamp(x, min=max(float(epsilon), precision))


def primitive(U, gravity: float, epsilon: float, dry_depth: float):
    """Convert [h, hu, hv] to depth, velocities, and wave celerity."""
    if U.ndim != 4 or U.shape[1] != 3:
        raise ValueError(
            f"U must have shape [batch, 3, ny, nx], got {tuple(U.shape)}."
        )
    if gravity <= 0 or epsilon <= 0 or dry_depth < 0:
        raise ValueError("Require gravity > 0, epsilon > 0, dry_depth >= 0.")

    h = torch.clamp(U[:, 0:1], min=0.0)
    wet = h > dry_depth
    safe_h = torch.clamp(h, min=_depth_floor(U, epsilon, dry_depth))
    zero = torch.zeros_like(h)

    u = torch.where(wet, U[:, 1:2] / safe_h, zero)
    v = torch.where(wet, U[:, 2:3] / safe_h, zero)
    c = torch.where(wet, torch.sqrt(float(gravity) * safe_h), zero)

    return h, u, v, c


def bflood_hllc_flux(
    UL, UR, normal: str, gravity: float, epsilon: float, dry_depth: float
):
    """B-Flood/Basilisk two-wave HLL flux with dry-front treatment."""
    if normal not in ("x", "y"):
        raise ValueError("normal must be 'x' or 'y'.")
    if UL.shape != UR.shape:
        raise ValueError(f"UL and UR shapes differ: {UL.shape} != {UR.shape}.")

    hL, uL, vL, cL = primitive(UL, gravity, epsilon, dry_depth)
    hR, uR, vR, cR = primitive(UR, gravity, epsilon, dry_depth)

    unL, utL = (uL, vL) if normal == "x" else (vL, uL)
    unR, utR = (uR, vR) if normal == "x" else (vR, uR)

    u_star = 0.5 * (unL + unR) + cL - cR
    c_star = 0.5 * (cL + cR) + 0.25 * (unL - unR)

    SL_wet = torch.minimum(unL - cL, u_star - c_star)
    SR_wet = torch.maximum(unR + cR, u_star + c_star)

    left_dry = hL <= dry_depth
    right_dry = hR <= dry_depth

    SL = torch.where(left_dry, unR - 2.0 * cR, SL_wet)
    SR = torch.where(right_dry, unL + 2.0 * cL, SR_wet)

    fhL, fhR = hL * unL, hR * unR
    fqL = hL * (unL.square() + 0.5 * float(gravity) * hL)
    fqR = hR * (unR.square() + 0.5 * float(gravity) * hR)

    denominator = _safe_denominator(SR - SL, epsilon)
    fh_star = (
        SR * fhL - SL * fhR + SL * SR * (hR - hL)
    ) / denominator
    fq_star = (
        SR * fqL - SL * fqR
        + SL * SR * (hR * unR - hL * unL)
    ) / denominator

    left_flux = SL >= 0.0
    right_flux = SR <= 0.0

    fh = torch.where(
        left_flux, fhL, torch.where(right_flux, fhR, fh_star)
    )
    fq = torch.where(
        left_flux, fqL, torch.where(right_flux, fqR, fq_star)
    )
    ft = torch.where(fh > 0.0, utL, utR) * fh

    flux = (
        torch.cat((fh, fq, ft), dim=1)
        if normal == "x"
        else torch.cat((fh, ft, fq), dim=1)
    )

    both_dry = left_dry & right_dry
    flux = torch.where(both_dry.expand_as(flux), torch.zeros_like(flux), flux)

    max_wave_speed = torch.maximum(SL.abs(), SR.abs())
    max_wave_speed = torch.where(
        both_dry, torch.zeros_like(max_wave_speed), max_wave_speed
    )

    return flux, max_wave_speed