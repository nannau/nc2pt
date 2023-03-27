import logging
import os
from timeit import default_timer as timer
import hydra
from dask.distributed import Client
import multiprocessing
from datetime import timedelta
import torch

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
    write_to_zarr
)


def make_batches(ds, steps_per_file, randomize):
    return (
        torch.split(torch.randperm(ds.time.size), steps_per_file)
        if randomize
        else torch.split(torch.arange(ds.time.size), steps_per_file)
    )


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg) -> None:
    cores = int(multiprocessing.cpu_count()/24)
    print(f"Using {cores} cores")

    with Client(
        n_workers=cores,
        threads_per_worker=cfg.compute.threads_per_worker,
        memory_limit=cfg.compute.memory_limit,
        processes=False,
        dashboard_address=cfg.compute.dashboard_address,
    ):

        start = timer()
        # Load the input dataset.
        logging.info("Loading input dataset...")
        lr_path_list = cfg.lr.path
        hr_path_list = cfg.hr.path

        lr = load_grid(lr_path_list, cfg.engine)
        hr = load_grid(hr_path_list, cfg.engine)

        # rename the variables to match
        logging.info("Homogenizing dataset keys...")
        keys = {"data_vars": cfg.vars, "coords": cfg.coords, "dims": cfg.dims}
        for key_attr, config in keys.items():
            lr = homogenize_names(lr, config, key_attr)
            hr = homogenize_names(hr, config, key_attr)

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
        lr = crop_field(
            lr, cfg.spatial.scale_factor, cfg.spatial.x, cfg.spatial.y
        )
        hr = crop_field(
            hr, cfg.spatial.scale_factor, cfg.spatial.x, cfg.spatial.y
        )

        lr = lr.drop(["lat", "lon"])
        hr = hr.drop(["lat", "lon"])

        # Coarsen the low resolution dataset.
        logging.info("Coarsening low resolution dataset...")
        lr = coarsen_lr(lr, cfg.spatial.scale_factor)

        # Train test split
        logging.info("Splitting dataset...")
        train_lr, test_lr = train_test_split(lr, cfg.time.test_years)
        train_hr, test_hr = train_test_split(hr, cfg.time.test_years)

        # Standardize the dataset.
        logging.info("Standardizing dataset...")
        train_lr = compute_standardization(train_lr)
        train_hr = compute_standardization(train_hr)
        test_lr = compute_standardization(test_lr, train_lr)
        test_hr = compute_standardization(test_hr, train_hr)

        # Write the output to disk.
        logging.info("Writing test output...")
        # batches = make_batches(test_lr, cfg.steps_per_file, cfg.randomize)
        # write_to_pt(test_lr, f"{cfg.lr.output_path}/test/", batches)
        # write_to_pt(test_hr, f"{cfg.hr.output_path}/test/", batches)
        # logging.info("Writing train output...")
        # batches = make_batches(train_lr, cfg.steps_per_file, cfg.randomize)
        # write_to_pt(train_lr, f"{cfg.lr.output_path}/train/", batches)
        # write_to_pt(train_hr, f"{cfg.hr.output_path}/train/", batches)

        write_to_zarr(test_lr, f"{cfg.lr.output_path}/test_lr")
        write_to_zarr(test_hr, f"{cfg.hr.output_path}/test_hr")
        logging.info("Writing train output...")
        write_to_zarr(train_lr, f"{cfg.lr.output_path}/train_lr")
        write_to_zarr(train_hr, f"{cfg.hr.output_path}/train_hr")

        end = timer()
        logging.info("Done!")
        logging.info(f"Time elapsed: {timedelta(seconds=end-start)}")


if __name__ == '__main__':
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
    main()
