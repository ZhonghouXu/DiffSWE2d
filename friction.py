import math
import torch


def smart_friction_coefficient(h, roughness_length, epsilon=1.0e-12):
    """BG_Flood smart-friction drag coefficient from h/z0.

    roughness_length is z0 in metres and must be positive. The piecewise shallow
    branch avoids the singular logarithmic-law region.
    """
    z0 = torch.clamp(roughness_length, min=epsilon)
    depth_ratio = torch.clamp(h, min=epsilon) / z0

    shallow = 1.0 / torch.clamp(0.46 * depth_ratio, min=epsilon)
    logarithmic_term = 2.5 * (
        torch.log(torch.clamp(depth_ratio, min=epsilon))
        - 1.0
        + 1.359 / torch.clamp(depth_ratio, min=epsilon)
    )
    deep = 1.0 / torch.clamp(logarithmic_term.square(), min=epsilon)
    return torch.where(depth_ratio < math.e, shallow, deep)


def apply_roughness_length_friction(U, roughness_length, dt,
                                    dry_depth=1.0e-6,
                                    epsilon=1.0e-12):
    """Semi-implicit BG_Flood bottom-friction velocity update.

    u_new = u_old / (1 + Cf*|u_old|*dt/h). Conserved momentum is scaled by the
    same factor while depth is unchanged.
    """
    h = torch.clamp(U[:, 0:1], min=0.0)
    hu, hv = U[:, 1:2], U[:, 2:3]
    wet = h > dry_depth
    denominator = torch.clamp(h, min=epsilon)
    u = torch.where(wet, hu / denominator, torch.zeros_like(h))
    v = torch.where(wet, hv / denominator, torch.zeros_like(h))
    speed = torch.sqrt(u.square() + v.square())
    cf = smart_friction_coefficient(h, roughness_length, epsilon)
    factor = 1.0 / (1.0 + cf * speed * dt / denominator)
    return torch.cat(
        (
            h,
            torch.where(wet, hu * factor, torch.zeros_like(hu)),
            torch.where(wet, hv * factor, torch.zeros_like(hv)),
        ),
        dim=1,
    )
