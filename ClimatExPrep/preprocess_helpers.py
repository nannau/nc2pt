import xarray as xr

import xesmf as xe
import pandas as pd

import logging
from datetime import datetime
from typing import Tuple
import glob
import numpy as np


class MissingKeys(Exception):
    pass


class MultipleKeys(Exception):
    pass


def regrid_align(ds: xr.Dataset, grid: xr.Dataset) -> xr.Dataset:
    """Regrid and interpolate input dataset to the given grid.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to regrid and align (i.e. ERA5).
    grid : xarray.Dataset
        Grid to regrid to (i.e. wrf).

    Returns
    -------
    ds : xarray.Dataset
        Dataset regridded and aligned to the given grid.
    """

    # Regrid to the given grid.
    regridder = xe.Regridder(ds, grid, "bilinear")
    ds = regridder(ds)
    return ds
    # ds.interp_like(grid)
    # return ds.interp(coords={"lat": grid.lat, "lon": grid.lon}, method="linear")
    # return


def load_grid(path: str, engine: str = "netcdf4", chunks: int = 100) -> xr.Dataset:
    """Load the grid to regrid to.

    Parameters
    ----------
    path : str
        Path to the grid to regrid to.

    Returns
    -------
    grid : xarray.Dataset
        Grid to regrid to.
    """

    chunky = {"time": chunks}

    if "*" in path:
        return xr.open_mfdataset(
            glob.glob(path, recursive=True),
            engine=engine,
            chunks=chunky,
            parallel=True,
        )
    else:
        return xr.open_dataset(path, engine=engine, chunks=chunky)


def slice_time(ds: xr.Dataset, start: str, end: str) -> xr.Dataset:
    """Slice the time dimension of the dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to slice.
    start : str
        Start time.
    end : str
        End time.

    Returns
    -------
    ds : xarray.Dataset
        Sliced dataset.
    """
    ds = ds.sel(time=slice(start, end), drop=True)

    return ds


def train_test_split(ds: xr.Dataset, years: list) -> Tuple[xr.Dataset, xr.Dataset]:
    """Split the dataset into a training and test set.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to split.
    years : list
        List of years to use for the test set.

    Returns
    -------
    train : xarray.Dataset
        Training dataset.
    test : xarray.Dataset
        Test dataset.
    """
    train = ds.isel(time=~ds.time.dt.year.isin(years), drop=True)
    test = ds.isel(time=ds.time.dt.year.isin(years), drop=True)

    return train, test


def match_longitudes(ds: xr.Dataset) -> xr.Dataset:
    """Match the longitudes of the dataset to the range [-180, 180].

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to match the longitudes of.

    Returns
    -------
    ds : xarray.Dataset
        Dataset with longitudes in the range [-180, 180].
    """
    ds = ds.assign_coords(lon=(ds.lon + 360))
    return ds


def crop_field(ds, scale_factor, x, y):
    """Crop the field to the given size.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to crop.
    scale_factor : int
        Scale factor of the dataset.
    x : int
        Size of the x dimension.
    y : int
        Size of the y dimension.

    Returns
    -------
    ds : xarray.Dataset
        Cropped dataset.
    """
    assert "rlon" in ds.dims, "rlon not in dims, check dataset"
    assert "rlat" in ds.dims, "rlat not in dims, check dataset"

    ds = ds.isel(
        rlon=slice(x.first_index, x.last_index),
        rlat=slice(y.first_index, y.last_index),
        drop=True,
    )

    assert (
        x.last_index - x.first_index
    ) % scale_factor == 0, "x dimension not divisible by scale factor, check config"
    assert (
        y.last_index - y.first_index
    ) % scale_factor == 0, "y dimension not divisible by scale factor, check config"

    assert (
        ds.rlon.size == ds.rlat.size
    ), "rlon and rlat not the same size, check dataset"

    return ds


def coarsen_lr(ds, scale_factor):
    """Coarsen the low resolution dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to coarsen.
    scale_factor : int
        Scale factor of the dataset.

    Returns
    -------
    ds : xarray.Dataset
        Coarsened dataset.
    """

    ds = ds.coarsen(rlon=scale_factor, rlat=scale_factor).mean()

    return ds


def homogenize_names(ds: xr.Dataset, keymaps: dict, key_attr: str) -> xr.Dataset:
    """Homogenize the names of the variables in the dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to homogenize the names of.
    var_info : dict
        Dictionary containing the variable names.

    Returns
    -------
    ds : xarray.Dataset
        Dataset with homogenized variable names.
    """
    keys = getattr(ds, key_attr)
    for name in keymaps:
        keymatch = [i for i in keymaps[name]["alternative_names"] if i in keys]
        if name not in keys:
            # Check if it is listed as an alternative name.
            if len(keymatch) == 1:
                ds = ds.rename({keymatch[0]: name})
                logging.info(f"Renamed {keymatch[0]} to {name}")
            elif len(keymatch) > 1:
                raise MultipleKeys(f"{name} has multiple alternatives in dataset.")
            # Check if the keyname is only in one of the datasets
            elif "hr_only" in keymaps[name] and keymaps[name]["hr_only"]:
                continue
            elif "lr_only" in keymaps[name] and keymaps[name]["lr_only"]:
                continue
            else:
                continue
    return ds


def standardize(x, mean, std):
    return (x - mean) / std


def compute_standardization(
    ds: xr.Dataset, var: str, precomputed: xr.Dataset = None
) -> xr.Dataset:  # sourcery skip: avoid-builtin-shadow
    """Standardize the statistics of the dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to standardize the statistics of.
    precomputed : xarray.Dataset, optional
        Dataset containing precomputed statistics, by default None
    Returns
    -------
    ds : xarray.Dataset
        Dataset with standardized statistics.
    """
    logging.info("Computing mean and standard deviation...")
    if precomputed is not None:
        mean = precomputed[var].attrs["mean"]
        std = precomputed[var].attrs["std"]
    else:
        logging.info("Calculation mean...")
        mean = ds[var].sum().compute() / ds[var].size
        logging.info("Calculation std...")
        std = ds[var].std(axis=0).compute().std()

    logging.info("Applying function...")
    ds[var] = xr.apply_ufunc(
        standardize,
        ds[var],
        mean,
        std,
        dask="parallelized",
    )
    ds[var] = ds[var].assign_attrs({"mean": float(mean), "std": float(std)})

    return ds


def write_to_zarr(ds: xr.Dataset, path: str, chunks: int = 500) -> None:
    """Write the output to disk.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to write to disk.
    path : str
        Path to write the dataset to.
    """
    ds = ds.assign_attrs(
        {
            "history": f"Created by {__file__} on {datetime.now()}",
        }
    )
    ds.chunk({"time": chunks}).to_zarr(f"{path}.zarr", mode="a")


def unit_change(x):
    return x * 3600


def log_transform(x):
    return np.log10(x + 1)
