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
import numpy as np
from pathlib import Path

from numpy.random import RandomState

from itertools import islice


def parallel_loop(sub_i, path, output_path):
    sub_path = [f"{path}{j}" for j in sub_i]
    stack = [torch.load(sub) for sub in sub_path]
    x = torch.stack(stack, dim=0)
    assert not torch.isnan(x).any(), f"NaNs found in {sub_path}"
    new_filename = [Path(x).stem for x in sub_i]
    torch.save(x, output_path + "__".join(new_filename) + ".pt")


def make_dirs(output_path: str, s, var_name: str, res: str) -> None:
    if not os.path.exists(f"{output_path}/batched/{s}/{var_name}/{res}"):
        os.makedirs(f"{output_path}/batched/{s}/{var_name}/{res}")


def batched(iterable, n):
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch


def loop_over_variables(climate_data, model, var, s, res):
    logging.info(f"✨ Processing {var.name}...")
    climate_data = instantiate(climate_data)
    output_path = climate_data.output_path

    prng = RandomState(1234567890)
    indices = prng.permutation(
        len(glob.glob(f"{climate_data.output_path}/{s}/{var.name}/{res}/*.pt"))
    ).tolist()

    permuted_paths = np.array(
        sorted(glob.glob(f"{climate_data.output_path}/{s}/{var.name}/{res}/*.pt"))
    )[indices]

    permuted_paths = np.array([os.path.basename(path) for path in permuted_paths])
    indices = list(batched(permuted_paths, climate_data.loader.batch_size))

    # Create parent dir if it doesn't exist for each variable
    make_dirs(output_path, s, var.name, res)
    partial_paths = [f"{output_path}/{s}/{var.name}/{res}/" for _ in indices]

    output_path = [f"{output_path}/batched/{s}/{var.name}/{res}/" for _ in indices]
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
        for s in ["train", "test", "validation"]:
            partial_set_loop(s)

        end = timer()
        logging.info(
            f"🚀🎉 Finished {model.name} dataset in {timedelta(seconds=end-start)}"
        )


if __name__ == "__main__":
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
    main()
