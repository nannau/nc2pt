import torch
from datetime import timedelta
from timeit import default_timer as timer
import hydra
from hydra.utils import instantiate
import logging
import os
import glob
from parallelbar import progress_starmap
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
    climate_data = instantiate(climate_data)
    output_path = climate_data.output_path

    indices = torch.randperm(
        len(glob.glob(f"{climate_data.output_path}/train/uas/lr/*.pt"))
    )
    indices = torch.split(indices, climate_data.loader.batch_size, dim=0)

    # Create parent dir if it doesn't exist for each variable
    make_dirs(output_path, s, var.name, res)
    partial_paths = [
        f"{output_path}/{s}/{var.name}/{res}/{var.name}" for _ in range(len(indices))
    ]

    output_path = [
        f"{output_path}/batched/{s}/{var.name}/{res}/{var.name}_{i}.pt"
        for i in range(len(indices))
    ]

    pool_tuple = zip(indices, partial_paths, output_path)
    progress_starmap(parallel_loop, pool_tuple, total=len(indices), n_cpu=24)


def loop_over_sets(climate_data, model, s):
    model = instantiate(model)

    logging.info(f"Loading {s} {model.name} dataset...")
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
        logging.info(f"Finished {model.name} dataset in {timedelta(seconds=end-start)}")


if __name__ == "__main__":
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
    main()
