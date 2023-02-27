import xarray as xr
import xesmf as xe

import logging
from datetime import datetime
from typing import Tuple


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


def load_grid(path: str, engine: str = "netcdf4") -> xr.Dataset:
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

    chunky = {"time": "auto"}

    return xr.open_mfdataset(
        path, engine=engine, data_vars="minimal", chunks=chunky, parallel=True
    )


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

    ds = ds.isel(rlon=slice(x.first_index, x.last_index), rlat=slice(y.first_index, y.last_index), drop=True)

    assert (x.last_index - x.first_index) % scale_factor == 0, "x dimension not divisible by scale factor, check config"
    assert (y.last_index - y.first_index) % scale_factor == 0, "y dimension not divisible by scale factor, check config"

    assert ds.rlon.size == ds.rlat.size, "rlon and rlat not the same size, check dataset"

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
                raise MissingKeys(f"{name} or alternative not found in dataset.")

    return ds


def compute_standardization(ds: xr.Dataset, precomputed: xr.Dataset = None) -> xr.Dataset:
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
    for var in ds.data_vars:
        if precomputed is not None:
            mean = precomputed[var].attrs["mean"]
            std = precomputed[var].attrs["std"]
        else:
            mean = ds[var].mean()
            std = ds[var].std()
        ds[var] = (ds[var] - mean) / std
        ds[var] = ds[var].assign_attrs(
            {
                "mean": float(mean),
                "std": float(std)
            }
        )
    return ds


def write_to_zarr(ds: xr.Dataset, path: str) -> None:
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
    ds.chunk({"time": "auto"}).to_zarr(path, mode="a")
