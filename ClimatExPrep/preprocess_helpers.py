import xarray as xr
# import xesmf as xe
import pandas as pd

import logging
from datetime import datetime
from typing import Tuple
import glob

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
    # regridder = xe.Regridder(ds, grid, "bilinear")
    # ds = regridder(ds)
    # ds.interp_like(grid)
    return ds.interp(coords={"lat": grid.lat, "lon": grid.lon}, method="linear")
    # return 


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
        glob.glob(path, recursive=True), engine=engine, data_vars="minimal", chunks=chunky, parallel=True
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
    for var in ds.data_vars:
        if "time" in ds[var].dims:
            ds[var] = ds[var].sel(time=slice(start, end), drop=True)
        elif "forecast_initial_time" in ds[var].dims:
            ds[var] = ds[var].sel(forecast_initial_time=slice(start, end), drop=True)
        else:
            raise ValueError("Time dimension not found in dataset.")

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

def convert_time(ds: xr.Dataset, cfg: dict) -> xr.Dataset:
    """Flattens the time dimensions to a single dimension."""

    for var in ds.data_vars:
        timelist = []
        fieldlist = []
        if cfg.vars[var]["slice_on_forecast_initial_time"]:
            logging.info(f"Converting time for {var}...")
            da_stacked = ds[var].stack(time=('forecast_initial_time', 'forecast_hour'))
            print(da_stacked.indexes["time"][0][0], da_stacked.indexes["time"][0][1])
            for t, (init_time, hour) in enumerate(da_stacked.indexes["time"]):
                time_delta = pd.Timedelta(hours=int(hour)-1)
                time = pd.to_datetime(init_time) + time_delta
                fieldlist.append(da_stacked.sel({"time": init_time}).isel({"forecast_hour": int(hour)-1}))
                timelist.append(time)
            
            ds[var] = xr.concat(fieldlist, dim="time")
            ds[var] = ds[var].assign_coords(time=timelist)
            if ["forecast_hour", "forecast_initial_time"] in ds[var].dims:
                ds[var] = ds[var].drop_vars(["forecast_hour", "forecast_initial_time"])
        else:
            continue

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
    ds.chunk({"time": "auto"}).to_zarr(f"{path}.zarr", mode="a")
