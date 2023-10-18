import torch
import xarray as xr
from datetime import timedelta
from timeit import default_timer as timer
import numpy as np
import hydra
from hydra.utils import instantiate
import logging
import os
import glob
from parallelbar import progress_starmap


torch.manual_seed(0)


def parallel_loop(sub_i, path, output_path):
    sub_path = [f"{path}_{j}.pt" for j in sub_i]
    stack = [torch.load(sub) for sub in sub_path]
    x = torch.stack(stack, dim=0)
    assert not torch.isnan(x).any(), f"NaNs found in {i}"
    torch.save(x, output_path)


def make_dirs(output_path: str, s, var_name: str, res: str) -> None:
    if not os.path.exists(f"{output_path}/batched/{s}/{var_name}/{res}"):
        os.makedirs(f"{output_path}/batched/{s}/{var_name}/{res}")


def loop_over_variables(climate_data, model, var, s, res):
    climate_data = instantiate(climate_data)
    output_path = climate_data.output_path
    dims = [instantiate(dim) for dim in climate_data.dims]
    chunks = {dim.name: dim.chunksize for dim in dims}

    indices = torch.randperm(
        len(glob.glob(f"{climate_data.output_path}/train/uas/lr/*.pt"))
    )
    indices = torch.split(indices, climate_data.loader.batch_size, dim=0)

    with xr.open_zarr(
        f"{output_path}/{var.name}_{s}_{model.name}.zarr/", chunks=chunks
    ) as ds:
        # Create parent dir if it doesn't exist for each variable
        make_dirs(output_path, s, var.name, res)
        indices = np.arange(ds.time.size)

        partial_paths = [
            f"{output_path}/{s}/{var.name}/{res}/{var.name}_{i}.pt" for i in indices
        ]

        pool_tuple = zip(
            indices,
            partial_paths,
            ds[var.name].transpose(*chunks.keys()),
        )

        progress_starmap(
            parallel_loop, pool_tuple, total=ds.time.size, n_cpu=16, chunk_size=1
        )


def loop_over_sets(climate_data, model, s):
    model = instantiate(model)

    logging.info(f"Loading {s} {model.name} dataset...")
    for var in model.climate_variables:
        loop_over_variables(climate_data, model, var, s)


@hydra.main(config_path="conf", config_name="config")
def main(cfg) -> None:
    climate_data = instantiate(climate_data)
    for model in climate_data.climate_models:
        start = timer()
        loop_over_sets(climate_data, model, "train")
        loop_over_sets(climate_data, model, "test")
        end = timer()
        logging.info(f"Finished {model.name} dataset in {timedelta(seconds=end-start)}")


# @hydra.main(version_base=None, config_path="conf", config_name="config")
# def main(cfg) -> None:
#     # Define for loop that iterates over sets, resolutions, and variables
#     # and saves each time step as a torch tensor to write to a pytorch file
#     # format.

#     indices = torch.randperm(
#         len(glob.glob(f"{cfg.vars['uas'].output_path}/train/uas/lr/*.pt"))
#     )
#     indices = torch.split(indices, cfg.loader.batch_size, dim=0)

#     for res in ["lr", "hr"]:
#         start = timer()
#         for s in ["train"]:
#             logging.info(f"Loading {s} {res} dataset...")
#             for var in cfg.vars:
#                 output_path = cfg.vars[var].output_path
#                 if not os.path.exists(f"{output_path}/batched/{s}/{var}/{res}"):
#                     logging.info(
#                         f"Creating directory: {output_path}/batched/{s}/{var}/{res}"
#                     )
#                     os.makedirs(f"{output_path}/batched/{s}/{var}/{res}")

#                 partial_paths = [
#                     f"{output_path}/{s}/{var}/{res}/{var}" for i in range(len(indices))
#                 ]

#                 output_path = [
#                     f"{output_path}/batched/{s}/{var}/{res}/{var}_{i}.pt"
#                     for i in range(len(indices))
#                 ]

#                 pool_tuple = zip(indices, partial_paths, output_path)

#                 # if __name__ == "__main__":
#                 with get_context("spawn").Pool() as pool:
#                     # progress_starmap(
#                     pool.starmap(
#                         parallel_loop,
#                         pool_tuple,
#                         # total=len(indices),
#                         # n_cpu=16,
#                     )
#                 logging.info(f"Completed {s} {res} {var} batching")

#         end = timer()
#         logging.info(f"Finished {res} dataset in {timedelta(seconds=end-start)}")


if __name__ == "__main__":
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
    main()
