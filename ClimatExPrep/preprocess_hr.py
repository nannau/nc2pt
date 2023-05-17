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
            # Now process the hr data
            logging.info("Loading HR input dataset...")
            hr_path_list = cfg.vars[var].hr.path

            hr = load_grid(hr_path_list, cfg.engine, chunks=500)

            # rename the variables to match
            logging.info("Homogenizing dataset keys...")
            keys = {"data_vars": cfg.vars, "coords": cfg.coords, "dims": cfg.dims}
            for key_attr, config in keys.items():
                hr = homogenize_names(hr, config, key_attr)

            # Check if keys are already in the dimensions
            logging.info("Matching longitudes...")
            hr = match_longitudes(hr) if cfg.vars[var].hr.is_west_negative else hr

            # Slice the time dimension.
            logging.info("Slicing time dimension...")
            hr = slice_time(hr, cfg.time.full.start, cfg.time.full.end)

            # Crop the field to the given size.
            logging.info("Cropping field...")
            hr = crop_field(hr, cfg.spatial.scale_factor, cfg.spatial.x, cfg.spatial.y)
            hr = hr.drop(["lat", "lon"])

            if var == "pr":
                logging.info("Apply log transform to hr...")
                hr[var] = xr.apply_ufunc(log_transform, hr[var], dask="parallelized")
                hr[var] = hr[var].assign_attrs({"transform": "log10(1+X)"})

            # Train test split
            logging.info("Splitting dataset...")
            train_hr, test_hr = train_test_split(hr, cfg.time.test_years)

            # Standardize the dataset.
            logging.info("Standardizing dataset...")
            if cfg.vars[var].standardize:
                logging.info(f"Standardizing {var}...")
                train_hr = compute_standardization(train_hr, var)
                test_hr = compute_standardization(test_hr, var, train_hr)

            # Write the output to disk.
            logging.info("Writing test output...")
            write_to_zarr(
                test_hr, f"{cfg.vars[var].output_path}/{var}_test_hr", chunks=50
            )
            logging.info("Writing train output...")
            write_to_zarr(
                train_hr, f"{cfg.vars[var].output_path}/{var}_train_hr", chunks=50
            )

            end = timer()
            logging.info("Done!")
            logging.info(f"Time elapsed: {timedelta(seconds=end-start)}")
