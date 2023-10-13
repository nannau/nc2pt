import logging
import glob
from typing import Tuple
from datetime import datetime

import xarray as xr
import xesmf as xe
import numpy as np


class MultipleKeys(Exception):
    """Raised when a variable has multiple keys in the dataset."""

    pass


class MissingKey(Exception):
    """Raised when a variable is missing from the dataset."""

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

    regridder = xe.Regridder(ds, grid, "bilinear")
    ds = regridder(ds)
    return ds


def load_grid(path: str, engine: str = "netcdf4", chunks: int = 250) -> xr.Dataset:
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
    """Homogenize variable names in the dataset based on a key mapping.

    1. If the variable name is not in the dataset but has one and only one
    alternative name in the dataset, it renames the variable to match the
    alternative name.
    2. If the variable name is not in the dataset but has multiple alternative
    names in the dataset, it raises a MultipleKeys exception.
    3. If the variable name is not in the dataset and doesn't have an
    alternative name, it checks if it has specific flags in the keymaps dictionary (e.g., "hr_only" or "lr_only"). If these flags are present and set to True, it continues to the next variable name; otherwise, it continues to the next variable name.

        Parameters
        ----------
        ds : xarray.Dataset
            Dataset to homogenize variable names.
        keymaps : dict
            Dictionary containing variable name mappings.
        key_attr : str
            Attribute in the dataset containing variable names.

        Returns
        -------
        ds : xarray.Dataset
            Dataset with homogenized variable names.
    """
    keys = getattr(ds, key_attr)

    for name in keymaps:
        keymatch = [i for i in keymaps[name]["alternative_names"] if i in keys]

        if name not in keys:
            # Check if it is listed as an alternative name and rename it.
            if len(keymatch) == 1:
                new_name = keymatch[0]
                ds = ds.rename({new_name: name})
                logging.info(f"Renamed {new_name} to {name}")
            elif len(keymatch) > 1:
                raise MultipleKeys(f"{name} has multiple alternatives in dataset.")
            # Check if the keyname is only in one of the datasets
            elif keymaps[name].get("hr_only") or keymaps[name].get("lr_only"):
                continue
            else:
                continue  # If none of the above conditions are met, skip this name.

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
    if precomputed:
        logging.info("Calculation mean...")
        mean = ds[var].mean().compute()
        logging.info("Calculation std...")
        std = ds[var].std().compute()
    else:
        mean = precomputed[var].attrs["mean"]
        std = precomputed[var].attrs["std"]

    logging.info("Applying function...")
    ds[var] = xr.apply_ufunc(
        standardize,
        ds[var],
        mean,
        std,
        dask="parallelized",
    )

    ds[var].attrs["mean"] = float(mean)
    ds[var].attrs["std"] = float(std)

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


def maxnorm(x, max):
    return x / max
