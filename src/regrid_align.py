import xarray as xr
from tqdm import tqdm
import glob

import logging
import os
from timeit import default_timer as timer
from datetime import timedelta
import hydra
from dask.distributed import Client
import dask
import multiprocessing


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
    ds = ds.interp(coords = {'longitude': grid.longitude, 'latitude': grid.latitude})

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

    ds = ds.assign_coords(longitude=(ds.longitude  + 360))

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

def write_to_zarr(ds: xr.Dataset, path: str) -> None:
    """Write the output to disk.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to write to disk.
    path : str
        Path to write the dataset to.
    """

    # ds.chunk(chunksize)
    logging.info("Creating Zarr store...")
    # ds.chunk({"time": 1}).isel(time=slice(0)).to_zarr(path, mode='a', region={'time': slice(0)})
    logging.info("Interating through Zarr store...")
    # for t in tqdm(range(1, len(ds.time))):
        # ds = ds.drop_vars(['longitude', 'latitude'])
        # ds.chunk({"time": 1}).isel(time=slice(t)).to_zarr(path, mode='a', region={'time': slice(t)})
    ds.chunk({"time": 500}).to_zarr(path, mode="w")

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg) -> None:
    start = timer()
    # Load the input dataset.
    logging.info("Loading input dataset...")
    lr_path_list = glob.glob(cfg.lr.path)
    hr_path_list = glob.glob(cfg.hr.path)

    lr = load_grid(lr_path_list, cfg.engine)
    hr = load_grid(hr_path_list, cfg.engine)

    # rename the variables to match
    logging.info("Renaming variables...")
    lr = lr.rename(cfg.lr.rename_vars).swap_dims(cfg.lr.rename_dims)
    hr = hr.rename(cfg.hr.rename_vars).swap_dims(cfg.hr.rename_dims)

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

    lr = lr.drop(["latitude", "longitude"])
    hr = hr.drop(["latitude", "longitude"])

    # Coarsen the low resolution dataset.
    logging.info("Coarsening low resolution dataset...")
    lr = coarsen_lr(lr, cfg.spatial.scale_factor)

    # Write the output to disk.
    logging.info("Writing output...")
    write_to_zarr(lr, cfg.lr.output_path)
    write_to_zarr(hr, cfg.hr.output_path)

    end = timer()
    logging.info("Done!")
    logging.info(f"Time elapsed: {timedelta(seconds=end-start)}")

if __name__ == '__main__':
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
    # with dask.config.set(**{'array.slicing.split_large_chunks': False}):
    # client = Client()
    cores = int(multiprocessing.cpu_count()/2)
    print(f"Using {cores} cores")
    client = Client(n_workers = cores, threads_per_worker = 2, memory_limit='24GB')
    main()