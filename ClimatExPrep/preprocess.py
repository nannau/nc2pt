import logging
import os
from timeit import default_timer as timer
import hydra
from dask.distributed import Client
import multiprocessing
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
)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg) -> None:
    cores = int(multiprocessing.cpu_count())
    print(f"Using {cores} cores")

    with Client(
        # n_workers=16,
        # threads_per_worker=1,
        # memory_limit="2GB",
        processes=False,
        # dashboard_address=cfg.compute.dashboard_address,
    ):
        logging.info("Now processing the following variables:")
        logging.info(cfg.vars.keys())
        for var in cfg.vars:
            start = timer()
            # process lr data first

            logging.info(f"Loading {var} LR input dataset...")
            lr_path_list = cfg.vars[var].lr.path
            hr_grid_ref = cfg.hr_grid_ref_path

            lr = load_grid(lr_path_list, cfg.engine, chunks=100)
            hr_ref = load_grid(hr_grid_ref, cfg.engine)

            logging.info("Homogenizing dataset keys...")
            keys = {"data_vars": cfg.vars, "coords": cfg.coords, "dims": cfg.dims}
            for key_attr, config in keys.items():
                lr = homogenize_names(lr, config, key_attr)
                hr_ref = homogenize_names(hr_ref, config, key_attr)

            if var == "pr":
                lr[var] = lr[var] * 3600
                lr[var] = lr[var].assign_attrs({"units": "mm/h"})

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

            # Train test split
            logging.info("Splitting dataset...")
            train_lr, test_lr = train_test_split(lr, cfg.time.test_years)

            # Standardize the dataset.
            logging.info("Standardizing dataset...")
            train_lr = compute_standardization(train_lr, var)
            test_lr = compute_standardization(test_lr, var, train_lr)

            # Write the output to disk.
            logging.info("Writing test output...")
            write_to_zarr(test_lr, f"{cfg.vars[var].output_path}/{var}_test_lr")
            logging.info("Writing train output...")
            write_to_zarr(train_lr, f"{cfg.vars[var].output_path}/{var}_train_lr")

            del test_lr
            del train_lr
            del lr
            del hr_ref

            end = timer()
            logging.info("Done LR!")
            logging.info(f"Time elapsed: {timedelta(seconds=end-start)}")

            start = timer()
            # Now process the hr data
            logging.info("Loading HR input dataset...")
            hr_path_list = cfg.vars[var].hr.path

            hr = load_grid(hr_path_list, cfg.engine)

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

            # Train test split
            logging.info("Splitting dataset...")
            train_hr, test_hr = train_test_split(hr, cfg.time.test_years)

            # Standardize the dataset.
            logging.info("Standardizing dataset...")
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

            del test_hr
            del train_hr
            del hr


if __name__ == "__main__":
    with dask.config.set(**{"array.slicing.split_large_chunks": False}):
        logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
        main()
