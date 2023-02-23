import xarray as xr
import xesmf as xe
import glob

import logging
import os
from timeit import default_timer as timer
from datetime import timedelta, datetime
import hydra
from dask.distributed import Client
import multiprocessing
from typing import Tuple


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


def load_grid(path: str, engine: str) -> xr.Dataset:
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

    chunky = {"time": 50}

    if len(path) > 1:
        grid = xr.open_mfdataset(path, engine=engine, data_vars="minimal", chunks=chunky)
    else:
        grid = xr.open_dataset(path[0], engine=engine, chunk=chunky)

    return grid


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
    test = ds.isel(time=~ds.time.dt.year.isin(years), drop=True)
    train = ds.isel(time=ds.time.dt.year.isin(years), drop=True)

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
    ds = ds.assign_coords(lon=(ds.lon  + 360))

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
                raise KeyError(f"{name} has multiple alternatives in dataset.")
            # Check if the keyname is only in one of the datasets
            elif "hr_only" in keymaps[name] and keymaps[name]["hr_only"]:
                continue
            elif "lr_only" in keymaps[name] and keymaps[name]["lr_only"]:
                continue
            else:
                raise KeyError(f"{name} or alternative not found in dataset.")

    return ds


def compute_standardization(ds: xr.Dataset, precomputed: xr.Dataset=None) -> xr.Dataset:
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
    ds.chunk({"time": 200}).to_zarr(path, mode="a")


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg) -> None:

    cores = int(multiprocessing.cpu_count()/2)
    print(f"Using {cores} cores")
    client = Client(
        n_workers=cores,
        threads_per_worker=cfg.compute.threads_per_worker,
        memory_limit=cfg.compute.memory_limit
    )

    start = timer()
    # Load the input dataset.
    logging.info("Loading input dataset...")
    lr_path_list = glob.glob(cfg.lr.path)
    hr_path_list = glob.glob(cfg.hr.path)

    lr = load_grid(lr_path_list, cfg.engine)
    hr = load_grid(hr_path_list, cfg.engine)

    # rename the variables to match
    logging.info("Homogenizing dataset keys...")
    keys = {"data_vars": cfg.vars, "coords": cfg.coords, "dims": cfg.dims}
    for key, config_value in keys.items():
        lr = homogenize_names(lr, config_value, key)
        hr = homogenize_names(hr, config_value, key)

    # Check if keys are already in the dimensions
    logging.info("Matching longitudes...")
    lr = match_longitudes(lr) if cfg.lr.is_west_negative else lr
    hr = match_longitudes(hr) if cfg.hr.is_west_negative else hr

    # Slice the time dimension.
    logging.info("Slicing time dimension...")
    lr = slice_time(lr, cfg.time.full.start, cfg.time.full.end)
    hr = slice_time(hr, cfg.time.full.start, cfg.time.full.end)

    # Regrid and align the dataset.
    logging.info("Regridding and interpolating...")
    lr = regrid_align(lr, hr)

    # Crop the field to the given size.
    logging.info("Cropping field...")
    lr = crop_field(lr, cfg.spatial.scale_factor, cfg.spatial.x, cfg.spatial.y)
    hr = crop_field(hr, cfg.spatial.scale_factor, cfg.spatial.x, cfg.spatial.y)

    lr = lr.drop(["lat", "lon"])
    hr = hr.drop(["lat", "lon"])

    # Coarsen the low resolution dataset.
    logging.info("Coarsening low resolution dataset...")
    lr = coarsen_lr(lr, cfg.spatial.scale_factor)

    # Train test split
    logging.info("Splitting dataset...")
    test_lr, train_lr = train_test_split(lr, cfg.time.test_years)
    test_hr, train_hr = train_test_split(hr, cfg.time.test_years)

    # Standardize the dataset.
    logging.info("Standardizing dataset...")
    train_lr = compute_standardization(train_lr)
    train_hr = compute_standardization(train_hr)
    test_lr = compute_standardization(test_lr, train_lr)
    test_hr = compute_standardization(test_hr, train_hr)

    # Write the output to disk.
    logging.info("Writing test output...")
    write_to_zarr(test_lr, f"{cfg.lr.output_path}/test_lr.zarr")
    write_to_zarr(test_hr, f"{cfg.hr.output_path}/test_hr.zarr")
    logging.info("Writing train output...")
    write_to_zarr(train_lr, f"{cfg.lr.output_path}/train_lr.zarr")
    write_to_zarr(train_hr, f"{cfg.hr.output_path}/train_hr.zarr")

    end = timer()
    logging.info("Done!")
    logging.info(f"Time elapsed: {timedelta(seconds=end-start)}")


if __name__ == '__main__':
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
    main()