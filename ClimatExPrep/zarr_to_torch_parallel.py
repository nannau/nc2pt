from dask.distributed import Client
import xarray as xr
import torch
from datetime import timedelta
from timeit import default_timer as timer
from tqdm import tqdm
import numpy as np
import hydra
import logging
import os
from multiprocessing import Pool
from tqdm.contrib.concurrent import process_map


def parallel_loop(i, path, arr):
    arr = arr.values
    x = torch.tensor(np.array(arr))
    torch.save(x, path)

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg) -> None:
    # Define for loop that iterates over sets, resolutions, and variables
    # and saves each time step as a torch tensor to write to a pytorch file
    # format.

    for res in ["lr", "hr"]:
        start = timer()
        for s in ["train", "test"]:
            logging.info(f"Loading {s} {res} dataset...")
            with xr.open_zarr(f"{cfg[res].output_path}/{s}_{res}.zarr/") as ds:
                for var in cfg.vars:
                    if "lr_only" in cfg.vars[var] and res == "hr" and cfg.vars[var].lr_only:
                        continue
                    if "hr_only" in cfg.vars[var] and res == "lr" and cfg.vars[var].hr_only:
                        continue
                    # Create parent dir if it doesn't exist for each variable
                    if not os.path.exists(f"{cfg[res].output_path}/{s}/{var}"):
                        logging.info(f"Creating directory: {cfg[res].output_path}/{s}/{var}")
                        os.makedirs(f"{cfg[res].output_path}/{s}/{var}")

                    logging.info(f"Saving {s} {res} {var} to torch tensors...")
                    logging.info(f"Writing to {cfg[res].output_path}/{s}/{var}/")
                    indices = np.arange(ds.time.size)
                    partial_paths = [f"{cfg[res].output_path}/{s}/{var}/{var}_{i}.pt" for i in indices]
                    pool_tuple = zip(indices, partial_paths, ds[var].transpose("time", "rlat", "rlon"))
                    if __name__ == '__main__':
                        with Pool() as pool:
                            pmask = pool.starmap(parallel_loop, pool_tuple, chunksize=1)

        end = timer()
        logging.info(f"Finished {res} dataset in {timedelta(seconds=end-start)}")


if __name__ == "__main__":
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
    # with Client(n_workers=8, threads_per_worker=2, processes=False, dashboard_address=8787, memory_limit='4GB'):
    main()
