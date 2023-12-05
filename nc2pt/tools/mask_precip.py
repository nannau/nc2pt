from multiprocessing import get_context
from datetime import timedelta
from timeit import default_timer as timer

from nc2pt.climatedata import ClimateVariable, ClimateModel

import torch
import hydra
from hydra.utils import instantiate
import logging
import os
import glob
from functools import partial
import numpy as np
from pathlib import Path

from numpy.random import RandomState


def parallel_loop(input_path, output_path):
    delta_precip = 0.0769779160618782 / 0.3726992905139923
    x = torch.load(input_path) > -delta_precip
    x = 1 * x
    assert not torch.isnan(x).any(), f"NaNs found in {input_path}"
    torch.save(x, output_path)


def make_dirs(output_path: str, s, var_name: str, res: str) -> None:
    if not os.path.exists(f"{output_path}/{s}/{var_name}/{res}"):
        os.makedirs(f"{output_path}/{s}/{var_name}/{res}")


def loop_over_variables(climate_data, model, var, s, res):
    logging.info(f"✨ Processing {var.name}...")
    climate_data = instantiate(climate_data)
    output_path_base = climate_data.output_path

    input_paths = sorted(glob.glob(f"{output_path_base}/{s}/pr/{res}/*.pt"))
    indices = [os.path.basename(path) for path in input_paths]

    make_dirs(output_path_base, s, "pr_mask", res)

    output_paths = [
        f"{output_path_base}/{s}/pr_mask/{res}/mask_{path}" for path in indices
    ]

    inputs = zip(input_paths, output_paths)

    with get_context("spawn").Pool(24) as pool:
        pool.starmap(parallel_loop, inputs)


def loop_over_sets(climate_data, model, s):
    model = instantiate(model)

    logging.info(f"👀 Loading {s} {model.name} dataset...")

    # hard code precip variable list
    for var in model.climate_variables:
        if var.name != "pr":
            raise ValueError("Only pr is supported for this script.")
        else:
            loop_over_variables(climate_data, model, var, s, model.name)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(climate_data) -> None:
    climate_data = instantiate(climate_data)
    era5 = ClimateVariable(
        name="pr",
        alternative_names=["PREC"],
        path="/home/nannau/ERA5_NCAR-RDA_North_America/proc/pr_1hr_ERA5_fc_RDA-025_1979010106-2019010106_time_sliced_cropped_merged_forecast_time.nc",
        is_west_negative=False,
        apply_standardize=False,
        invariant=False,
        transform="x * 3600",
    )
    wrf = ClimateVariable(
        name="pr",
        alternative_names=["PREC"],
        path="/home/nannau/USask-WRF-WCA/fire_vars/PREC/*.nc",
        is_west_negative=True,
        apply_standardize=False,
        invariant=False,
        transform=None,
    )

    models = [
        ClimateModel(
            name="lr",
            info="ERA5",
            climate_variables=[era5],
        ),
        ClimateModel(
            name="hr",
            info="WRF",
            climate_variables=[wrf],
        ),
    ]

    for model in models:
        start = timer()

        partial_set_loop = partial(loop_over_sets, climate_data, model)
        for s in ["train", "test", "validation"]:
            partial_set_loop(s)

        end = timer()
        logging.info(
            f"🚀🎉 Finished {model.name} dataset in {timedelta(seconds=end-start)}"
        )


if __name__ == "__main__":
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
    main()
