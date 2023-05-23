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
from joblib import Parallel, delayed
from parallelbar import progress_imap, progress_starmap, progress_imapu


def parallel_loop(i, path, arr):
    # i, path, arr = tup
    arr = arr.values
    x = torch.tensor(np.array(arr))
    assert not torch.isnan(x).any(), f"NaNs found in {i}"
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
            for var in cfg.vars:
                output_path = cfg.vars[var].output_path
                with xr.open_zarr(f"{output_path}/{var}_{s}_{res}.zarr/", chunks={"time": 1000}) as ds:
                    # Create parent dir if it doesn't exist for each variable
                    if not os.path.exists(f"{output_path}/{s}/{var}/{res}"):
                        logging.info(
                            f"Creating directory: {output_path}/{s}/{var}/{res}"
                        )
                        os.makedirs(f"{output_path}/{s}/{var}/{res}")

                    logging.info(f"Saving {s} {res} {var} to torch tensors...")
                    logging.info(f"Writing to {output_path}/{s}/{var}/{res}/")
                    indices = np.arange(ds.time.size)
                    # partial_paths = Parallel(n_jobs=-1)(
                    #     delayed(
                    #         lambda i: f"{output_path}/{s}/{var}/{res}/{var}_{i}.pt"
                    #     )(i)
                    #     for i in indices
                    # )
                    partial_paths = [
                        f"{output_path}/{s}/{var}/{res}/{var}_{i}.pt" for i in indices
                    ]

                    pool_tuple = zip(
                        indices,
                        partial_paths,
                        ds[var].transpose("time", "rlat", "rlon"),
                    )
                    if __name__ == "__main__":
                        # process_map(parallel_loop, pool_tuple, max_workers=16)
                        progress_starmap(
                            parallel_loop,
                            pool_tuple,
                            total=ds.time.size,
                            n_cpu=16,
                            chunk_size=1,
                        )
                        # with Pool() as pool:
                        #     pmask = pool.starmap(parallel_loop, pool_tuple, chunksize=1)

        end = timer()
        logging.info(f"Finished {res} dataset in {timedelta(seconds=end-start)}")


if __name__ == "__main__":
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
    # with Client(n_workers=8, threads_per_worker=2, processes=False, dashboard_address=8787, memory_limit='4GB'):
    main()
