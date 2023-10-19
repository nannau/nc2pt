import logging
from typing import Dict

import pandas as pd
import xarray as xr
import xesmf as xe

from nc2pt.climatedata import ClimateData, ClimateModel


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
    # Check that end is not before start
    end = pd.to_datetime(end)
    start = pd.to_datetime(start)

    if end < start:
        raise ValueError("End date is before start date.")

    ds = ds.sel(time=slice(start, end), drop=True)

    return ds


def train_test_split(ds: xr.Dataset, years: list) -> Dict[str, xr.Dataset]:
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
    # check that time is in the dataset
    if "time" not in ds.dims:
        raise ValueError("Time dimension not in dataset.")

    # check that list of years is not empty
    if not years:
        raise ValueError("List of years is empty.")

    # check if years is a list
    if not isinstance(years, list):
        raise ValueError("Years is not a list.") 

    train = ds.isel(time=~ds.time.dt.year.isin(years), drop=True)
    test = ds.isel(time=ds.time.dt.year.isin(years), drop=True)

    return {"train": train, "test": test}


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


def interpolate(ds: xr.Dataset, grid: xr.Dataset) -> xr.Dataset:
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


def align_grid(ds: xr.DataArray, hr_ref: xr.DataArray, climdata: ClimateData) -> xr.DataArray:
    """Processes low resolution dataset for machine learning.

    Regrids, aligns, crops, and coarsens the low resolution dataset for use in machine learning.

    Args:
    ds: Low resolution xarray dataset.
    climdata: Climate data configuration object.

    Returns:
    Processed low resolution xarray dataset.
    """
    # Regrid and align the dataset.
    logging.info("Regridding and interpolating...")
    ds = interpolate(ds, hr_ref)

    # Crop the field to the given size.
    logging.info("Cropping field...")
    ds = crop_field(
        ds,
        climdata.select.spatial.scale_factor,
        climdata.select.spatial.x,
        climdata.select.spatial.y,
    )
    ds = ds.drop(["lat", "lon"])

    return ds


def align_with_lr(ds, hr_ref, climdata) -> xr.Dataset:
    """Processes low resolution dataset for machine learning.

    Regrids, aligns, crops, and coarsens the low resolution dataset for use in machine learning.

    Args:
    ds: Low resolution xarray dataset.
    climdata: Climate data configuration object.

    Returns:
    Processed low resolution xarray dataset.
    """
    # Regrid and align the dataset.
    ds = align_grid(ds, hr_ref, climdata)

    # Coarsen the low resolution dataset.
    logging.info("Coarsening low resolution dataset...")
    ds = coarsen_lr(ds, climdata.select.spatial.scale_factor)

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
