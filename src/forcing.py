import torch
import torch.nn as nn


class RainfallForcing(nn.Module):
    """Rainfall already remapped to the DEM grid.

    times has shape [nt] in seconds and rates has shape [nt,1,ny,nx] in m/s.
    Linear time interpolation is used. Values outside the time range are held
    at the first or last record.
    """

    def __init__(self, times: torch.Tensor, rates: torch.Tensor):
        super().__init__()
        if times.ndim != 1 or rates.ndim != 4 or rates.shape[0] != times.numel():
            raise ValueError("Expected times=[nt] and rates=[nt,1,ny,nx]")
        if times.numel() < 1:
            raise ValueError("Rainfall forcing must contain at least one time")
        if times.numel() > 1 and not torch.all(times[1:] > times[:-1]):
            raise ValueError("Rainfall times must be strictly increasing")
        self.register_buffer("times", times)
        self.register_buffer("rates", rates)

    def at(self, time) -> torch.Tensor:
        """Return rainfall [1,1,ny,nx] at model time in seconds."""
        if self.times.numel() == 1:
            return self.rates[0:1]

        t = torch.as_tensor(time, dtype=self.times.dtype, device=self.times.device)
        right = torch.searchsorted(self.times, t).clamp(1, self.times.numel() - 1)
        left = right - 1
        width = torch.clamp(self.times[right] - self.times[left], min=1.0e-12)
        weight = (t - self.times[left]) / width
        interpolated = (1.0 - weight) * self.rates[left] + weight * self.rates[right]
        interpolated = torch.where(t <= self.times[0], self.rates[0], interpolated)
        interpolated = torch.where(t >= self.times[-1], self.rates[-1], interpolated)
        return interpolated.unsqueeze(0)
