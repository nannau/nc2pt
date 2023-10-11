import logging
import os
from timeit import default_timer as timer
import hydra
from dask.distributed import Client
import multiprocessing
import xarray as xr
import numpy as np
from datetime import timedelta

import dask

from ClimatExPrep.preprocess_helpers import (
    load_grid,
    regrid_align,
    slice_time,
    crop_field,
    coarsen_lr,
    train_test_split,
    homogenize_names,
    match_longitudes,
    compute_standardization,
    write_to_zarr,
    unit_change,
    log_transform,
)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def start(cfg) -> None:
    cores = int(multiprocessing.cpu_count())
    print(f"Using {cores} cores")

    with Client(
        # n_workers=16,
        # threads_per_worker=2,
        processes=False,
        # memory_limit="16GB",
        dashboard_address=cfg.compute.dashboard_address,
    ):
        logging.info("Now processing the following variables:")
        logging.info(cfg.vars.keys())
        for var in cfg.vars:
            start = timer()
            # process lr data first

            logging.info(f"Loading {var} LR input dataset...")
            lr_path_list = cfg.vars[var].lr.path
            hr_grid_ref = cfg.hr_grid_ref_path

            lr = load_grid(lr_path_list, cfg.engine, chunks="auto")
            hr_ref = load_grid(hr_grid_ref, cfg.engine)

            logging.info("Homogenizing dataset keys...")
            keys = {"data_vars": cfg.vars, "coords": cfg.coords, "dims": cfg.dims}
            for key_attr, config in keys.items():
                lr = homogenize_names(lr, config, key_attr)
                hr_ref = homogenize_names(hr_ref, config, key_attr)

            logging.info("Matching longitudes...")
            lr = match_longitudes(lr) if cfg.vars[var].lr.is_west_negative else lr
            hr_ref = match_longitudes(hr_ref)
            logging.info("Slicing time dimension...")
            lr = slice_time(lr, cfg.time.full.start, cfg.time.full.end)

            # Regrid and align the dataset.
            logging.info("Regridding and interpolating...")
            lr = regrid_align(lr, hr_ref)

            # Crop the field to the given size.
            logging.info("Cropping field...")
            lr = crop_field(lr, cfg.spatial.scale_factor, cfg.spatial.x, cfg.spatial.y)
            lr = lr.drop(["lat", "lon"])

            # Coarsen the low resolution dataset.
            logging.info("Coarsening low resolution dataset...")
            lr = coarsen_lr(lr, cfg.spatial.scale_factor)

            if var == "pr":
                logging.info("Changing units of lr...")
                # function to change units to mm/hr
                lr[var] = xr.apply_ufunc(unit_change, lr[var], dask="parallelized")
                logging.info("Apply log transform to lr...")
                lr[var] = xr.apply_ufunc(log_transform, lr[var], dask="parallelized")
                lr[var].attrs["units"] = "mm/h"
                lr[var].attrs["transform"] = "log10(1+X)"

            # Train test split
            logging.info("Splitting dataset...")
            train_lr, test_lr = train_test_split(lr, cfg.time.test_years)

            # Standardize the dataset.
            logging.info("Standardizing dataset...")
            if cfg.vars[var].standardize:
                logging.info(f"Standardizing {var}...")
                train_lr = compute_standardization(train_lr, var)
                test_lr = compute_standardization(test_lr, var, train_lr)

            # Write the output to disk.
            logging.info("Writing test output...")
            write_to_zarr(test_lr, f"{cfg.vars[var].output_path}/{var}_test_lr")
            logging.info("Writing train output...")
            write_to_zarr(train_lr, f"{cfg.vars[var].output_path}/{var}_train_lr")

            end = timer()
            logging.info("Done LR!")
            logging.info(f"Time elapsed: {timedelta(seconds=end-start)}")
