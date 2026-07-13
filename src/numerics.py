import torch


def minmod3(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Three-argument minmod limiter.

    The limiter is piecewise differentiable. At branch boundaries PyTorch uses
    the derivative of the selected branch, which is standard for differentiable
    finite-volume implementations with hard TVD limiting.
    """
    same_sign = (torch.sign(a) == torch.sign(b)) & (torch.sign(b) == torch.sign(c))
    magnitude = torch.minimum(torch.abs(a), torch.minimum(torch.abs(b), torch.abs(c)))
    return torch.where(same_sign, torch.sign(a) * magnitude, torch.zeros_like(a))


def limited_increment(q: torch.Tensor, dim: int, theta: float = 1.3) -> torch.Tensor:
    """Return the limited cell increment used for face extrapolation.

    The returned value is Delta*q rather than dq/dx. Therefore face values are
    reconstructed as q +/- 0.5*Delta*q on a uniform grid.
    """
    q_minus = torch.roll(q, shifts=1, dims=dim)
    q_plus = torch.roll(q, shifts=-1, dims=dim)
    backward = q - q_minus
    centred = 0.5 * (q_plus - q_minus)
    forward = q_plus - q
    return minmod3(theta * backward, centred, theta * forward)
