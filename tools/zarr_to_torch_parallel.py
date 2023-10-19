import os
import logging
from datetime import timedelta
from timeit import default_timer as timer

import xarray as xr
import torch
import numpy as np
from hydra.utils import instantiate
import hydra

from parallelbar import progress_starmap


def parallel_loop(i, path, arr):
    # i, path, arr = tup
    arr = arr.values
    x = torch.tensor(np.array(arr))
    assert not torch.isnan(x).any(), f"NaNs found in {i}"
    torch.save(x, path)


def make_dirs(output_path: str, s, var_name: str, model_name: str) -> None:
    if not os.path.exists(f"{output_path}/{s}/{var_name}/{model_name}"):
        os.makedirs(f"{output_path}/{s}/{var_name}/{model_name}")


def loop_over_variables(climate_data, model, var, s):
    climate_data = instantiate(climate_data)
    output_path = climate_data.output_path
    dims = [instantiate(dim) for dim in climate_data.dims]
    chunks = {dim.name: dim.chunksize for dim in dims}

    with xr.open_zarr(
        f"{output_path}/{var.name}_{s}_{model.name}.zarr/", chunks=chunks
    ) as ds:
        # Create parent dir if it doesn't exist for each variable
        make_dirs(output_path, s, var.name, model.name)
        indices = np.arange(ds.time.size)

        partial_paths = [
            f"{output_path}/{s}/{var.name}/{model.name}/{var.name}_{i}.pt"
            for i in indices
        ]

        pool_tuple = zip(
            indices,
            partial_paths,
            ds[var.name].transpose(*chunks.keys()),
        )

        progress_starmap(
            parallel_loop, pool_tuple, total=ds.time.size, n_cpu=24, chunk_size=1
        )


def loop_over_sets(climate_data, model, s):
    model = instantiate(model)

    logging.info(f"Loading {s} {model.name} dataset...")
    for var in model.climate_variables:
        loop_over_variables(climate_data, model, var, s)


@hydra.main(version_base=None, config_path="../nc2pt/conf", config_name="config")
def main(climate_data) -> None:
    # Define for loop that iterates over sets, resolutions, and variables
    # and saves each time step as a torch tensor to write to a pytorch file
    # format.
    climate_data = instantiate(climate_data)
    for model in climate_data.climate_models:
        start = timer()
        loop_over_sets(climate_data, model, "train")
        loop_over_sets(climate_data, model, "test")
        end = timer()
        logging.info(f"Finished {model.name} dataset in {timedelta(seconds=end-start)}")


if __name__ == "__main__":
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
    main()
