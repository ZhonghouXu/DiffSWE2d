"""
io_ascii.py

Read ESRI ASCII rasters and interpolate them onto an existing SWE2D
ModelGrid created with, for example:

    model_grid = ModelGrid.from_bounds(
        xmin=0.0,
        xmax=1580.0,
        ymin=0.0,
        ymax=780.0,
        resolution=10.0,
        dtype=torch.float64,
        device=device,
    )

The module does not define or replace the package's ModelGrid class.
Instead, it converts the supplied ModelGrid into an internal GridSpec
for NumPy/SciPy interpolation.

Grid convention
---------------
The model-grid bounds are assumed to represent the OUTER EDGES of the
computational domain.

For example:

    xmin = 0
    xmax = 1580
    resolution = 10

produces:

    nx = 158
    x cell centres = 5, 15, ..., 1575

Similarly:

    ymin = 0
    ymax = 780
    resolution = 10

produces:

    ny = 78
    y cell centres = 5, 15, ..., 775

Array orientation
-----------------
ESRI ASCII files normally store the northern/top row first.

This module flips the ASCII array after reading so that:

    array[0, :]  = southern row
    array[-1, :] = northern row

Therefore, array row index increases with y.

Dependencies
------------
numpy
scipy
torch
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Union

import numpy as np
import torch
from scipy.interpolate import NearestNDInterpolator
from scipy.interpolate import RegularGridInterpolator
import pandas as pd


PathLike = Union[str, Path]
GridInput = Any


# =============================================================================
# Internal regular-grid representation
# =============================================================================


@dataclass(frozen=True)
class GridSpec:
    """
    Lightweight CPU representation of an SWE2D model grid.

    This class is used only for ASCII reading and SciPy interpolation.
    The original SWE2D ModelGrid should still be retained by the model.

    Parameters
    ----------
    xmin, xmax
        Outer-edge coordinates in the x-direction.

    ymin, ymax
        Outer-edge coordinates in the y-direction.

    resolution
        Width and height of each square model cell.
    """

    xmin: float
    xmax: float
    ymin: float
    ymax: float
    resolution: float

    def __post_init__(self) -> None:
        values = {
            "xmin": self.xmin,
            "xmax": self.xmax,
            "ymin": self.ymin,
            "ymax": self.ymax,
            "resolution": self.resolution,
        }

        for name, value in values.items():
            if not np.isfinite(value):
                raise ValueError(
                    f"{name} must be finite, got {value}."
                )

        if self.xmax <= self.xmin:
            raise ValueError(
                "xmax must be greater than xmin: "
                f"xmin={self.xmin}, xmax={self.xmax}."
            )

        if self.ymax <= self.ymin:
            raise ValueError(
                "ymax must be greater than ymin: "
                f"ymin={self.ymin}, ymax={self.ymax}."
            )

        if self.resolution <= 0.0:
            raise ValueError(
                "resolution must be positive, got "
                f"{self.resolution}."
            )

        # Validate that both extents contain an integer number of cells.
        _ = self.nx
        _ = self.ny

    @staticmethod
    def _number_of_cells(
        lower: float,
        upper: float,
        resolution: float,
        axis: str,
    ) -> int:
        cell_count_float = (upper - lower) / resolution
        cell_count = int(round(cell_count_float))

        tolerance = max(
            1.0e-9,
            abs(cell_count_float) * 1.0e-10,
        )

        if not np.isclose(
            cell_count_float,
            cell_count,
            rtol=0.0,
            atol=tolerance,
        ):
            raise ValueError(
                f"The {axis}-extent is not divisible by resolution:\n"
                f"    lower      = {lower}\n"
                f"    upper      = {upper}\n"
                f"    resolution = {resolution}\n"
                f"    cell count = {cell_count_float}"
            )

        if cell_count < 1:
            raise ValueError(
                f"The grid must contain at least one {axis}-cell."
            )

        return cell_count

    @property
    def nx(self) -> int:
        """Number of model cells in the x-direction."""
        return self._number_of_cells(
            lower=self.xmin,
            upper=self.xmax,
            resolution=self.resolution,
            axis="x",
        )

    @property
    def ny(self) -> int:
        """Number of model cells in the y-direction."""
        return self._number_of_cells(
            lower=self.ymin,
            upper=self.ymax,
            resolution=self.resolution,
            axis="y",
        )

    @property
    def shape(self) -> tuple[int, int]:
        """Grid shape in array order: (ny, nx)."""
        return self.ny, self.nx

    @property
    def x(self) -> np.ndarray:
        """X-coordinate of each model-cell centre."""
        return (
            self.xmin
            + (
                np.arange(
                    self.nx,
                    dtype=np.float64,
                )
                + 0.5
            )
            * self.resolution
        )

    @property
    def y(self) -> np.ndarray:
        """Y-coordinate of each model-cell centre."""
        return (
            self.ymin
            + (
                np.arange(
                    self.ny,
                    dtype=np.float64,
                )
                + 0.5
            )
            * self.resolution
        )

    @property
    def x_edges(self) -> np.ndarray:
        """X-coordinate of model-cell edges."""
        return (
            self.xmin
            + np.arange(
                self.nx + 1,
                dtype=np.float64,
            )
            * self.resolution
        )

    @property
    def y_edges(self) -> np.ndarray:
        """Y-coordinate of model-cell edges."""
        return (
            self.ymin
            + np.arange(
                self.ny + 1,
                dtype=np.float64,
            )
            * self.resolution
        )

    def as_dict(self) -> dict[str, Any]:
        """Return the grid definition as a dictionary."""
        return {
            "xmin": self.xmin,
            "xmax": self.xmax,
            "ymin": self.ymin,
            "ymax": self.ymax,
            "resolution": self.resolution,
            "nx": self.nx,
            "ny": self.ny,
            "shape": self.shape,
        }


def _to_float(value: Any, name: str) -> float:
    """
    Convert a Python, NumPy, or scalar PyTorch value to float.

    CUDA tensors are moved to CPU before conversion.
    """
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(
                f"{name} must contain one value, but its shape is "
                f"{tuple(value.shape)}."
            )

        value = value.detach().cpu().item()

    elif isinstance(value, np.ndarray):
        if value.size != 1:
            raise ValueError(
                f"{name} must contain one value, but its shape is "
                f"{value.shape}."
            )

        value = value.item()

    result = float(value)

    if not np.isfinite(result):
        raise ValueError(
            f"{name} must be finite, got {result}."
        )

    return result


def _extract_resolution(model_grid: Any) -> Optional[float]:
    """
    Extract scalar resolution from a model-grid object.

    Supported attribute names are:

        resolution
        dx
        cellsize
        cell_size

    The function also accepts a two-component resolution if dx == dy.
    """
    for name in (
        "resolution",
        "dx",
        "cellsize",
        "cell_size",
    ):
        if not hasattr(model_grid, name):
            continue

        value = getattr(model_grid, name)

        if isinstance(value, torch.Tensor):
            values = (
                value.detach()
                .cpu()
                .numpy()
                .reshape(-1)
            )
        else:
            values = np.asarray(value).reshape(-1)

        if values.size == 1:
            return _to_float(values[0], name)

        if values.size == 2:
            dx = _to_float(values[0], f"{name}[0]")
            dy = _to_float(values[1], f"{name}[1]")

            if not np.isclose(
                dx,
                dy,
                rtol=0.0,
                atol=max(
                    1.0e-12,
                    abs(dx) * 1.0e-10,
                ),
            ):
                raise ValueError(
                    "This ASCII reader currently requires square "
                    f"model cells, but dx={dx} and dy={dy}."
                )

            return dx

        raise ValueError(
            f"{name} must be scalar or contain dx and dy, "
            f"but it contains {values.size} values."
        )

    return None


def _to_1d_numpy(
    value: Any,
    name: str,
) -> np.ndarray:
    """
    Convert coordinate data to a one-dimensional NumPy array.
    """
    if isinstance(value, torch.Tensor):
        array = (
            value.detach()
            .cpu()
            .numpy()
        )
    else:
        array = np.asarray(value)

    array = np.asarray(
        array,
        dtype=np.float64,
    ).squeeze()

    if array.ndim != 1:
        raise ValueError(
            f"{name} must be a one-dimensional coordinate vector, "
            f"but its shape is {array.shape}."
        )

    if array.size < 1:
        raise ValueError(
            f"{name} cannot be empty."
        )

    if not np.isfinite(array).all():
        raise ValueError(
            f"{name} contains NaN or infinite values."
        )

    return array


def parse_model_grid(
    model_grid: GridInput,
) -> GridSpec:
    """
    Convert an existing SWE2D ModelGrid into a GridSpec.

    Accepted formats
    ----------------
    1. GridSpec.

    2. Dictionary:

       {
           "xmin": ...,
           "xmax": ...,
           "ymin": ...,
           "ymax": ...,
           "resolution": ...
       }

    3. Existing ModelGrid containing attributes:

       model_grid.xmin
       model_grid.xmax
       model_grid.ymin
       model_grid.ymax
       model_grid.resolution

    4. Existing ModelGrid containing cell-centre vectors:

       model_grid.x
       model_grid.y
       model_grid.resolution

    Returns
    -------
    GridSpec
        CPU-based grid description suitable for SciPy interpolation.
    """
    if isinstance(model_grid, GridSpec):
        return model_grid

    required_bounds = (
        "xmin",
        "xmax",
        "ymin",
        "ymax",
    )

    # Dictionary or Mapping input.
    if isinstance(model_grid, Mapping):
        missing = [
            name
            for name in (
                "xmin",
                "xmax",
                "ymin",
                "ymax",
                "resolution",
            )
            if name not in model_grid
        ]

        if missing:
            raise KeyError(
                "model_grid is missing required entries: "
                f"{missing}"
            )

        return GridSpec(
            xmin=_to_float(
                model_grid["xmin"],
                "xmin",
            ),
            xmax=_to_float(
                model_grid["xmax"],
                "xmax",
            ),
            ymin=_to_float(
                model_grid["ymin"],
                "ymin",
            ),
            ymax=_to_float(
                model_grid["ymax"],
                "ymax",
            ),
            resolution=_to_float(
                model_grid["resolution"],
                "resolution",
            ),
        )

    resolution = _extract_resolution(model_grid)

    # Existing ModelGrid with explicit bounds.
    if all(
        hasattr(model_grid, name)
        for name in required_bounds
    ):
        if resolution is None:
            raise AttributeError(
                "The supplied model_grid contains bounds but does not "
                "provide resolution, dx, cellsize, or cell_size."
            )

        return GridSpec(
            xmin=_to_float(
                model_grid.xmin,
                "xmin",
            ),
            xmax=_to_float(
                model_grid.xmax,
                "xmax",
            ),
            ymin=_to_float(
                model_grid.ymin,
                "ymin",
            ),
            ymax=_to_float(
                model_grid.ymax,
                "ymax",
            ),
            resolution=resolution,
        )

    # Alternative: infer outer bounds from cell-centre coordinate vectors.
    if (
        hasattr(model_grid, "x")
        and hasattr(model_grid, "y")
    ):
        if resolution is None:
            raise AttributeError(
                "The supplied model_grid contains x and y coordinates "
                "but does not provide resolution, dx, cellsize, or "
                "cell_size."
            )

        x = _to_1d_numpy(
            model_grid.x,
            "model_grid.x",
        )

        y = _to_1d_numpy(
            model_grid.y,
            "model_grid.y",
        )

        # The existing coordinate vectors are assumed to represent
        # model-cell centres.
        xmin = float(np.min(x) - 0.5 * resolution)
        xmax = float(np.max(x) + 0.5 * resolution)
        ymin = float(np.min(y) - 0.5 * resolution)
        ymax = float(np.max(y) + 0.5 * resolution)

        grid = GridSpec(
            xmin=xmin,
            xmax=xmax,
            ymin=ymin,
            ymax=ymax,
            resolution=resolution,
        )

        if grid.nx != x.size:
            raise ValueError(
                "The number of x coordinates does not agree with "
                "the inferred grid bounds and resolution:\n"
                f"    coordinate count = {x.size}\n"
                f"    inferred nx      = {grid.nx}"
            )

        if grid.ny != y.size:
            raise ValueError(
                "The number of y coordinates does not agree with "
                "the inferred grid bounds and resolution:\n"
                f"    coordinate count = {y.size}\n"
                f"    inferred ny      = {grid.ny}"
            )

        return grid

    raise TypeError(
        "Unsupported model_grid object. The supplied grid must be:\n"
        "  1. A GridSpec\n"
        "  2. A dictionary with xmin, xmax, ymin, ymax, resolution\n"
        "  3. An object with xmin, xmax, ymin, ymax and resolution\n"
        "  4. An object with x, y and resolution"
    )


# =============================================================================
# ESRI ASCII raster representation
# =============================================================================


@dataclass
class AsciiRaster:
    """
    ESRI ASCII raster and its spatial metadata.

    The values array uses ascending-y orientation:

        values[0, :]  = southern row
        values[-1, :] = northern row
    """

    values: np.ndarray

    xmin: float
    xmax: float
    ymin: float
    ymax: float

    resolution: float
    ncols: int
    nrows: int

    nodata_value: Optional[float] = None
    path: Optional[Path] = None

    @property
    def shape(self) -> tuple[int, int]:
        """Raster shape in array order: (nrows, ncols)."""
        return self.nrows, self.ncols

    @property
    def x(self) -> np.ndarray:
        """Source raster cell-centre x-coordinates."""
        return (
            self.xmin
            + (
                np.arange(
                    self.ncols,
                    dtype=np.float64,
                )
                + 0.5
            )
            * self.resolution
        )

    @property
    def y(self) -> np.ndarray:
        """Source raster cell-centre y-coordinates."""
        return (
            self.ymin
            + (
                np.arange(
                    self.nrows,
                    dtype=np.float64,
                )
                + 0.5
            )
            * self.resolution
        )

    @property
    def x_edges(self) -> np.ndarray:
        """Source raster cell-edge x-coordinates."""
        return (
            self.xmin
            + np.arange(
                self.ncols + 1,
                dtype=np.float64,
            )
            * self.resolution
        )

    @property
    def y_edges(self) -> np.ndarray:
        """Source raster cell-edge y-coordinates."""
        return (
            self.ymin
            + np.arange(
                self.nrows + 1,
                dtype=np.float64,
            )
            * self.resolution
        )

    def as_dict(self) -> dict[str, Any]:
        """Return raster metadata as a dictionary."""
        return {
            "path": (
                None
                if self.path is None
                else str(self.path)
            ),
            "xmin": self.xmin,
            "xmax": self.xmax,
            "ymin": self.ymin,
            "ymax": self.ymax,
            "resolution": self.resolution,
            "ncols": self.ncols,
            "nrows": self.nrows,
            "shape": self.shape,
            "nodata_value": self.nodata_value,
        }


# =============================================================================
# ASCII header reading
# =============================================================================


def _read_ascii_header(
    filepath: PathLike,
) -> tuple[dict[str, float], int]:
    """
    Read an ESRI ASCII header.

    Returns
    -------
    header
        Dictionary using lower-case header keys.

    header_line_count
        Number of header lines to skip when reading raster values.
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(
            f"ASCII raster was not found: {filepath}"
        )

    if not filepath.is_file():
        raise ValueError(
            f"ASCII raster path is not a file: {filepath}"
        )

    recognised_keys = {
        "ncols",
        "nrows",
        "xllcorner",
        "yllcorner",
        "xllcenter",
        "yllcenter",
        "cellsize",
        "nodata_value",
    }

    header: dict[str, float] = {}
    header_line_count = 0

    with filepath.open(
        "r",
        encoding="utf-8",
    ) as stream:
        while True:
            position = stream.tell()
            line = stream.readline()

            if line == "":
                break

            stripped = line.strip()

            if not stripped:
                # Retain the position logic in case there is an empty
                # line between the header and values.
                continue

            components = stripped.split()

            if len(components) < 2:
                stream.seek(position)
                break

            key = components[0].lower()

            if key not in recognised_keys:
                stream.seek(position)
                break

            try:
                value = float(components[1])
            except ValueError as exc:
                raise ValueError(
                    f"Invalid ASCII header line in {filepath}:\n"
                    f"    {stripped}"
                ) from exc

            header[key] = value
            header_line_count += 1

    required_keys = {
        "ncols",
        "nrows",
        "cellsize",
    }

    missing = required_keys.difference(header)

    if missing:
        raise ValueError(
            f"ASCII header in {filepath} is missing entries: "
            f"{sorted(missing)}"
        )

    has_corner_origin = (
        "xllcorner" in header
        and "yllcorner" in header
    )

    has_centre_origin = (
        "xllcenter" in header
        and "yllcenter" in header
    )

    if not has_corner_origin and not has_centre_origin:
        raise ValueError(
            f"ASCII header in {filepath} must contain either:\n"
            "    xllcorner and yllcorner\n"
            "or:\n"
            "    xllcenter and yllcenter"
        )

    return header, header_line_count


# =============================================================================
# ASCII raster reading
# =============================================================================


def read_ascii(
    filepath: PathLike,
    *,
    convert_nodata_to_nan: bool = True,
    flip_y: bool = True,
    dtype: np.dtype = np.float64,
) -> AsciiRaster:
    """
    Read an ESRI ASCII raster.

    Parameters
    ----------
    filepath
        Input ESRI ASCII file.

    convert_nodata_to_nan
        Replace NODATA_value with NaN.

    flip_y
        Flip the original top-to-bottom ESRI orientation into ascending
        y orientation. This should normally remain True.

    dtype
        NumPy dtype used when reading values.

    Returns
    -------
    AsciiRaster
        Raster values and spatial metadata.
    """
    filepath = Path(filepath)

    header, header_line_count = _read_ascii_header(
        filepath
    )

    ncols_float = header["ncols"]
    nrows_float = header["nrows"]

    ncols = int(round(ncols_float))
    nrows = int(round(nrows_float))

    if not np.isclose(ncols_float, ncols):
        raise ValueError(
            f"ncols must be an integer, got {ncols_float}."
        )

    if not np.isclose(nrows_float, nrows):
        raise ValueError(
            f"nrows must be an integer, got {nrows_float}."
        )

    if ncols < 1 or nrows < 1:
        raise ValueError(
            "ncols and nrows must be positive, got "
            f"ncols={ncols}, nrows={nrows}."
        )

    resolution = float(header["cellsize"])

    if (
        not np.isfinite(resolution)
        or resolution <= 0.0
    ):
        raise ValueError(
            "ASCII cellsize must be finite and positive, "
            f"got {resolution}."
        )

    # Pandas parses text blocks written in C and is about 10x faster for large ASCIIs
    values = pd.read_csv(filepath, sep=r'\s+', skiprows=header_line_count, header=None).values

    # returns a one-dimensional array for a raster with one
    # row or one column.
    if values.ndim == 1:
        if nrows == 1:
            values = values.reshape(1, ncols)
        elif ncols == 1:
            values = values.reshape(nrows, 1)

    expected_shape = (nrows, ncols)

    if values.shape != expected_shape:
        raise ValueError(
            "ASCII raster dimensions do not match the header:\n"
            f"    file           = {filepath}\n"
            f"    loaded shape   = {values.shape}\n"
            f"    expected shape = {expected_shape}"
        )

    if "xllcorner" in header:
        xmin = float(header["xllcorner"])
    else:
        # Convert lower-left cell centre to lower-left outer edge.
        xmin = (
            float(header["xllcenter"])
            - 0.5 * resolution
        )

    if "yllcorner" in header:
        ymin = float(header["yllcorner"])
    else:
        # Convert lower-left cell centre to lower-left outer edge.
        ymin = (
            float(header["yllcenter"])
            - 0.5 * resolution
        )

    xmax = xmin + ncols * resolution
    ymax = ymin + nrows * resolution

    nodata_value = header.get("nodata_value")

    values = np.asarray(
        values,
        dtype=dtype,
    )

    if (
        convert_nodata_to_nan
        and nodata_value is not None
    ):
        values = np.array(
            values,
            copy=True,
        )

        tolerance = max(
            1.0e-12,
            abs(nodata_value) * 1.0e-12,
        )

        nodata_mask = np.isclose(
            values,
            nodata_value,
            rtol=0.0,
            atol=tolerance,
        )

        values[nodata_mask] = np.nan

    if flip_y:
        # ESRI ASCII stores the northern row first.
        values = np.flipud(values).copy()
    else:
        values = np.ascontiguousarray(values)

    return AsciiRaster(
        values=values,
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
        resolution=resolution,
        ncols=ncols,
        nrows=nrows,
        nodata_value=nodata_value,
        path=filepath,
    )


# =============================================================================
# NoData handling
# =============================================================================


def _fill_internal_nodata_nearest(
    raster: AsciiRaster,
) -> np.ndarray:
    """
    Fill missing source cells using nearest valid source cells.

    This fills NoData holes within the source raster. It does not control
    how target cells outside the source domain are handled.
    """
    values = np.asarray(
        raster.values,
        dtype=np.float64,
    ).copy()

    invalid = ~np.isfinite(values)

    if not invalid.any():
        return values

    valid = ~invalid

    if not valid.any():
        raise ValueError(
            "The ASCII raster contains no valid finite cells: "
            f"{raster.path}"
        )

    source_xx, source_yy = np.meshgrid(
        raster.x,
        raster.y,
        indexing="xy",
    )

    valid_points = np.column_stack(
        (
            source_yy[valid],
            source_xx[valid],
        )
    )

    valid_values = values[valid]

    interpolator = NearestNDInterpolator(
        valid_points,
        valid_values,
    )

    values[invalid] = interpolator(
        source_yy[invalid],
        source_xx[invalid],
    )

    return values


# =============================================================================
# Regridding
# =============================================================================


def regrid_ascii(
    raster: AsciiRaster,
    model_grid: GridInput,
    *,
    method: str = "linear",
    outside_domain: str = "error",
    fill_internal_nodata: bool = True,
) -> np.ndarray:
    """
    Interpolate an ASCII raster onto an existing SWE2D ModelGrid.

    Parameters
    ----------
    raster
        Source raster returned by read_ascii().

    model_grid
        Existing ModelGrid, GridSpec, or bounds dictionary.

    method
        Interpolation method:

        - "linear"
        - "nearest"

    outside_domain
        Behaviour when target cell centres are outside the source raster:

        - "error":
          Raise an exception.

        - "nearest":
          Use nearest-neighbour extrapolation.

        - "nan":
          Leave target cells as NaN.

    fill_internal_nodata
        Fill NoData holes in the source raster before interpolation.

    Returns
    -------
    np.ndarray
        Regridded values with shape (ny, nx).
    """
    grid = parse_model_grid(model_grid)

    method = str(method).lower()
    outside_domain = str(outside_domain).lower()

    if method not in {
        "linear",
        "nearest",
    }:
        raise ValueError(
            "method must be 'linear' or 'nearest', got "
            f"{method!r}."
        )

    if outside_domain not in {
        "error",
        "nearest",
        "nan",
    }:
        raise ValueError(
            "outside_domain must be 'error', 'nearest', or "
            f"'nan', got {outside_domain!r}."
        )

    # Linear RegularGridInterpolator requires at least two points along
    # each interpolated dimension.
    if (
        method == "linear"
        and (
            raster.ncols < 2
            or raster.nrows < 2
        )
    ):
        raise ValueError(
            "Linear interpolation requires at least two source cells "
            "in both x and y. Use method='nearest' for a one-row or "
            "one-column raster."
        )

    if fill_internal_nodata:
        source_values = _fill_internal_nodata_nearest(
            raster
        )
    else:
        source_values = np.asarray(
            raster.values,
            dtype=np.float64,
        ).copy()

    if (
        method == "linear"
        and not np.isfinite(source_values).all()
    ):
        raise ValueError(
            "The source raster contains NaN or infinite values. "
            "Use fill_internal_nodata=True before linear "
            "interpolation."
        )

    target_xx, target_yy = np.meshgrid(
        grid.x,
        grid.y,
        indexing="xy",
    )

    # The source values are ordered as (y, x), so interpolation points
    # must also be supplied as (y, x).
    target_points = np.column_stack(
        (
            target_yy.ravel(),
            target_xx.ravel(),
        )
    )

    interpolator = RegularGridInterpolator(
        points=(
            raster.y,
            raster.x,
        ),
        values=source_values,
        method=method,
        bounds_error=False,
        fill_value=np.nan,
    )

    result = interpolator(
        target_points
    ).reshape(grid.shape)

    invalid = ~np.isfinite(result)

    if invalid.any() and outside_domain == "nearest":
        nearest_interpolator = RegularGridInterpolator(
            points=(
                raster.y,
                raster.x,
            ),
            values=source_values,
            method="nearest",
            bounds_error=False,

            # None enables extrapolation outside the source range.
            fill_value=None,
        )

        nearest_result = nearest_interpolator(
            target_points
        ).reshape(grid.shape)

        result[invalid] = nearest_result[invalid]
        invalid = ~np.isfinite(result)

    if invalid.any() and outside_domain == "error":
        invalid_count = int(
            np.count_nonzero(invalid)
        )

        invalid_x = target_xx[invalid]
        invalid_y = target_yy[invalid]

        raise ValueError(
            f"{invalid_count} target cells could not be assigned a "
            "valid raster value.\n"
            "\n"
            "Source ASCII outer bounds:\n"
            f"    xmin = {raster.xmin}\n"
            f"    xmax = {raster.xmax}\n"
            f"    ymin = {raster.ymin}\n"
            f"    ymax = {raster.ymax}\n"
            "\n"
            "Target model-grid outer bounds:\n"
            f"    xmin = {grid.xmin}\n"
            f"    xmax = {grid.xmax}\n"
            f"    ymin = {grid.ymin}\n"
            f"    ymax = {grid.ymax}\n"
            "\n"
            "Invalid target cell-centre range:\n"
            f"    x = [{invalid_x.min()}, {invalid_x.max()}]\n"
            f"    y = [{invalid_y.min()}, {invalid_y.max()}]\n"
            "\n"
            "Use outside_domain='nearest' if nearest-neighbour "
            "extrapolation is acceptable."
        )

    return np.ascontiguousarray(result)


# =============================================================================
# High-level loading functions
# =============================================================================


def read_ascii_on_grid(
    filepath: PathLike,
    model_grid: GridInput,
    *,
    method: str = "linear",
    outside_domain: str = "error",
    fill_internal_nodata: bool = True,
    dtype: np.dtype = np.float32,
) -> tuple[np.ndarray, GridSpec]:
    """
    Read an ASCII raster and interpolate it onto an SWE2D ModelGrid.

    Returns
    -------
    values
        NumPy array with shape (ny, nx).

    grid_spec
        CPU representation of the supplied ModelGrid.
    """
    grid_spec = parse_model_grid(model_grid)

    source_raster = read_ascii(
        filepath=filepath,
        convert_nodata_to_nan=True,
        flip_y=True,
        dtype=np.float64,
    )

    values = regrid_ascii(
        raster=source_raster,
        model_grid=grid_spec,
        method=method,
        outside_domain=outside_domain,
        fill_internal_nodata=fill_internal_nodata,
    )

    values = np.asarray(
        values,
        dtype=dtype,
    )

    return np.ascontiguousarray(values), grid_spec


def ascii_to_tensor(
    filepath: PathLike,
    model_grid: GridInput,
    *,
    method: str = "linear",
    outside_domain: str = "error",
    fill_internal_nodata: bool = True,
    device: Optional[
        Union[str, torch.device]
    ] = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, GridSpec]:
    """
    Read, regrid, and convert an ASCII raster to a PyTorch tensor.

    Parameters
    ----------
    filepath
        Input ESRI ASCII file.

    model_grid
        Existing SWE2D ModelGrid.

    method
        "linear" or "nearest".

    outside_domain
        "error", "nearest", or "nan".

    fill_internal_nodata
        Fill source NoData holes before interpolation.

    device
        Destination PyTorch device.

    dtype
        Destination PyTorch dtype.

    Returns
    -------
    tensor
        Regridded tensor with shape (ny, nx).

    grid_spec
        CPU representation of the supplied ModelGrid.
    """
    # Read through float64 to preserve interpolation precision before
    # converting to the requested PyTorch dtype.
    array, grid_spec = read_ascii_on_grid(
        filepath=filepath,
        model_grid=model_grid,
        method=method,
        outside_domain=outside_domain,
        fill_internal_nodata=fill_internal_nodata,
        dtype=np.float64,
    )

    tensor = torch.as_tensor(
        np.ascontiguousarray(array),
        dtype=dtype,
        device=device,
    )

    expected_shape = (
        grid_spec.ny,
        grid_spec.nx,
    )

    if tuple(tensor.shape) != expected_shape:
        raise RuntimeError(
            "Unexpected output tensor shape:\n"
            f"    actual   = {tuple(tensor.shape)}\n"
            f"    expected = {expected_shape}"
        )

    return tensor, grid_spec


def read_ascii_native_tensor(
    filepath: PathLike,
    *,
    convert_nodata_to_nan: bool = True,
    device: Optional[
        Union[str, torch.device]
    ] = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, GridSpec]:
    """
    Read an ASCII raster on its native grid without interpolation.

    Returns
    -------
    tensor
        Native raster tensor in ascending-y orientation.

    grid_spec
        GridSpec corresponding to the native ASCII raster.
    """
    raster = read_ascii(
        filepath=filepath,
        convert_nodata_to_nan=convert_nodata_to_nan,
        flip_y=True,
        dtype=np.float64,
    )

    grid_spec = GridSpec(
        xmin=raster.xmin,
        xmax=raster.xmax,
        ymin=raster.ymin,
        ymax=raster.ymax,
        resolution=raster.resolution,
    )

    tensor = torch.as_tensor(
        np.ascontiguousarray(raster.values),
        dtype=dtype,
        device=device,
    )

    return tensor, grid_spec



# =============================================================================
# Diagnostic functions
# =============================================================================


def grids_are_equal(
    raster: AsciiRaster,
    model_grid: GridInput,
    *,
    atol: float = 1.0e-8,
) -> bool:
    """
    Check whether an ASCII raster and ModelGrid have identical bounds,
    resolution, and cell dimensions.
    """
    grid = parse_model_grid(model_grid)

    return bool(
        np.isclose(
            raster.xmin,
            grid.xmin,
            rtol=0.0,
            atol=atol,
        )
        and np.isclose(
            raster.xmax,
            grid.xmax,
            rtol=0.0,
            atol=atol,
        )
        and np.isclose(
            raster.ymin,
            grid.ymin,
            rtol=0.0,
            atol=atol,
        )
        and np.isclose(
            raster.ymax,
            grid.ymax,
            rtol=0.0,
            atol=atol,
        )
        and np.isclose(
            raster.resolution,
            grid.resolution,
            rtol=0.0,
            atol=atol,
        )
        and raster.ncols == grid.nx
        and raster.nrows == grid.ny
    )


def describe_ascii(
    filepath: PathLike,
) -> dict[str, Any]:
    """
    Return metadata and value statistics for an ASCII raster.
    """
    raster = read_ascii(
        filepath=filepath,
        convert_nodata_to_nan=True,
        flip_y=True,
        dtype=np.float64,
    )

    valid = np.isfinite(raster.values)

    description = raster.as_dict()

    description.update(
        {
            "valid_cell_count": int(
                np.count_nonzero(valid)
            ),
            "invalid_cell_count": int(
                np.count_nonzero(~valid)
            ),
            "minimum": (
                float(np.nanmin(raster.values))
                if valid.any()
                else None
            ),
            "maximum": (
                float(np.nanmax(raster.values))
                if valid.any()
                else None
            ),
            "mean": (
                float(np.nanmean(raster.values))
                if valid.any()
                else None
            ),
        }
    )

    return description


def describe_model_grid(
    model_grid: GridInput,
) -> dict[str, Any]:
    """
    Return a standard dictionary describing an existing ModelGrid.
    """
    grid = parse_model_grid(model_grid)
    return grid.as_dict()


def load_rainfall_txt(filepath):
    """Reads rainfall txt (Col 1: Time, Col 2: Rain in mm) and converts to SI units."""
    data = np.loadtxt(filepath)
    times = data[:, 0]
    rain_meters = data[:, 1] / 1000.0 
    
    return times, rain_meters
# =============================================================================
# Public exports
# =============================================================================


__all__ = [
    "AsciiRaster",
    "GridSpec",
    "ascii_to_tensor",
    "describe_ascii",
    "describe_model_grid",
    "grids_are_equal",
    "parse_model_grid",
    "read_ascii",
    "read_ascii_native_tensor",
    "read_ascii_on_grid",
    "regrid_ascii",
    "load_rainfall_txt"
]