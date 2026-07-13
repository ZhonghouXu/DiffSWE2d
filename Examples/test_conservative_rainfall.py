import numpy as np
import torch
from diffswe2d_conservative.io_netcdf import ModelGrid,conservative_rectilinear_remap


def test_precipitation_volume_conservation_same_extent():
    # Source: 2x3 cells, each 20 m square. Target: 4x6 cells, each 10 m square.
    sx=np.array([10.,30.,50.]); sy=np.array([10.,30.])
    target=ModelGrid.from_bounds(0.,60.,0.,40.,resolution=10.)
    source=np.array([[[1.,2.,3.],[4.,5.,6.]]])
    target_rate=conservative_rectilinear_remap(source,sx,sy,target)
    source_volume=(source*20.*20.).sum()
    target_volume=(target_rate*10.*10.).sum()
    assert np.isclose(source_volume,target_volume,rtol=1e-12,atol=1e-12)


def test_square_grid_rejects_different_dx_dy():
    try:
        ModelGrid.from_cell_centres(torch.arange(4.)*10.,torch.arange(4.)*20.)
    except ValueError as error:
        assert "dx == dy" in str(error)
    else:
        raise AssertionError("Expected non-square grid rejection")
