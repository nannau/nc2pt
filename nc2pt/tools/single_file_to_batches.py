from multiprocessing import get_context
from datetime import timedelta
from timeit import default_timer as timer

import torch
import hydra
from hydra.utils import instantiate
import logging
import os
import glob
from functools import partial

torch.manual_seed(0)


def parallel_loop(sub_i, path, output_path):
    sub_path = [f"{path}_{j}.pt" for j in sub_i]
    stack = [torch.load(sub) for sub in sub_path]
    x = torch.stack(stack, dim=0)
    assert not torch.isnan(x).any(), f"NaNs found in {sub_path}"
    torch.save(x, output_path)


def make_dirs(output_path: str, s, var_name: str, res: str) -> None:
    if not os.path.exists(f"{output_path}/batched/{s}/{var_name}/{res}"):
        os.makedirs(f"{output_path}/batched/{s}/{var_name}/{res}")


def loop_over_variables(climate_data, model, var, s, res):
    logging.info(f"✨ Processing {var.name}...")
    climate_data = instantiate(climate_data)
    output_path = climate_data.output_path

    indices = torch.randperm(
        len(glob.glob(f"{climate_data.output_path}/{s}/uas/lr/*.pt"))
    )
    indices = torch.split(indices, climate_data.loader.batch_size, dim=0)

    # Create parent dir if it doesn't exist for each variable
    make_dirs(output_path, s, var.name, res)
    partial_paths = [
        f"{output_path}/{s}/{var.name}/{res}/{var.name}" for _ in range(len(indices))
    ]
    # output paths with zfill
    zfill_length = len(str(len(indices)))
    output_path = [
        f"{output_path}/batched/{s}/{var.name}/{res}/{var.name}_{str(i).zfill(zfill_length)}.pt"
        for i in range(len(indices))
    ]

    inputs = zip(indices, partial_paths, output_path)

    with get_context("spawn").Pool(24) as pool:
        pool.starmap(parallel_loop, inputs)


def loop_over_sets(climate_data, model, s):
    model = instantiate(model)

    logging.info(f"👀 Loading {s} {model.name} dataset...")
    for var in model.climate_variables:
        loop_over_variables(climate_data, model, var, s, model.name)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(climate_data) -> None:
    climate_data = instantiate(climate_data)
    for model in climate_data.climate_models:
        start = timer()

        partial_set_loop = partial(loop_over_sets, climate_data, model)
        for s in ["validation"]:
            partial_set_loop(s)

        end = timer()
        logging.info(
            f"🚀🎉 Finished {model.name} dataset in {timedelta(seconds=end-start)}"
        )


if __name__ == "__main__":
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
    main()
