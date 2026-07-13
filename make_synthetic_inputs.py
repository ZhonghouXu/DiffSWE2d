"""Create small NetCDF files with deliberately different spatial grids."""
import numpy as np
import xarray as xr

# DEM/roughness grid: 40 x 80 at 20 m.
x = np.arange(80) * 20.0
y = np.arange(40) * 20.0
X, Y = np.meshgrid(x, y)
bed = 0.2 * np.exp(-((X-800.0)**2 + (Y-400.0)**2) / 80000.0)
z0 = np.full_like(bed, 0.003)
xr.Dataset(
    data_vars={
        "elevation": (("y", "x"), bed, {"units": "m"}),
        "z0": (("y", "x"), z0, {"units": "m"}),
    },
    coords={"x": x, "y": y},
).to_netcdf("dem_and_roughness.nc")

# Rainfall grid: 18 x 32 at 50 m, covering a somewhat different extent.
rx = np.arange(32) * 50.0 - 50.0
ry = np.arange(18) * 50.0 - 25.0
time = np.array([0.0, 900.0, 1800.0, 3600.0])
rain = np.zeros((time.size, ry.size, rx.size))
rain[1] = 20.0
rain[2] = 10.0
xr.Dataset(
    data_vars={"rainfall": (("time", "y", "x"), rain, {"units": "mm/h"})},
    coords={"time": ("time", time, {"units": "seconds"}), "x": rx, "y": ry},
).to_netcdf("rainfall.nc")
