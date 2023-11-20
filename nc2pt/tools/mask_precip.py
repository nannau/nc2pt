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


def parallel_loop(sub_i, path, output_path):
    sub_path = [f"{path}{j}" for j in sub_i]
    stack = [torch.load(sub) for sub in sub_path]
    x = torch.stack(stack, dim=0) > 0
    x = 1 * x
    assert not torch.isnan(x).any(), f"NaNs found in {sub_path}"
    new_filename = [f"mask_{Path(x).stem}" for x in sub_i]
    torch.save(x, output_path + "__".join(new_filename) + ".pt")


def make_dirs(output_path: str, s, var_name: str, res: str) -> None:
    if not os.path.exists(f"{output_path}/batched/{s}/{var_name}/{res}"):
        os.makedirs(f"{output_path}/batched/{s}/{var_name}/{res}")


def loop_over_variables(climate_data, model, var, s, res):
    logging.info(f"✨ Processing {var.name}...")
    climate_data = instantiate(climate_data)
    output_path = climate_data.output_path

    prng = RandomState(1234567890)
    indices = prng.permutation(
        len(glob.glob(f"{climate_data.output_path}/{s}/{var.name}/{res}/*.pt"))
    ).tolist()

    permuted_paths = np.array(
        glob.glob(f"{climate_data.output_path}/{s}/{var.name}/{res}/*.pt")
    )[indices]

    permuted_paths = np.array([os.path.basename(path) for path in permuted_paths])
    indices = np.array_split(
        permuted_paths, len(indices) // climate_data.loader.batch_size, axis=0
    )

    # Create parent dir if it doesn't exist for each variable
    make_dirs(output_path, s, "pr_mask", res)
    partial_paths = [f"{output_path}/{s}/{var.name}/{res}/" for _ in indices]

    output_path = [f"{output_path}/batched/{s}/pr_mask/{res}/" for _ in indices]
    inputs = zip(indices, partial_paths, output_path)

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
