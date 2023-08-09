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
import xbatcher

def parallel_loop(i, path, arr):
    # i, path, arr = tup
    # arr = arr.values  # [ds_indexed[i]]
    arr = arr.values
    # print(arr.shape)
    x = torch.tensor(np.array(arr))
    assert not torch.isnan(x).any(), f"NaNs found in {i}"
    torch.save(x, path)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg) -> None:
    # Define for loop that iterates over sets, resolutions, and variables
    # and saves each time step as a torch tensor to write to a pytorch file
    # format.

    # for res in ["lr", "hr"]:
    for res in ["hr"]:
        start = timer()
        for s in ["train", "test"]:
            logging.info(f"Loading {s} {res} dataset...")
            for var in cfg.vars:
                output_path = cfg.vars[var].output_path
                with xr.open_zarr(
                    f"{output_path}/{var}_{s}_{res}.zarr/", chunks={"time": 1000}
                ) as ds:
                    # ds = ds.chunk({"time": cfg.loader.batch_size})
                    # Create parent dir if it doesn't exist for each variable
                    if not os.path.exists(f"{output_path}/{s}/{var}/{res}"):
                        logging.info(
                            f"Creating directory: {output_path}/{s}/{var}/{res}"
                        )
                        os.makedirs(f"{output_path}/{s}/{var}/{res}")

                    logging.info(f"Saving {s} {res} {var} to torch tensors...")
                    logging.info(f"Writing to {output_path}/{s}/{var}/{res}/")
                    indices = np.arange(ds.time.size)
                    indices = torch.split(
                        torch.tensor(indices), cfg.loader.batch_size, dim=0
                    )
                    partial_paths = [
                        f"{output_path}/{s}/{var}/{res}/{var}_{i}.pt"
                        for i in range(len(indices))
                    ]

                    ds_indexed = [
                        ds[var].transpose("time", "rlat", "rlon")[i, ...]
                        for i in indices
                    ]

                    bgen = xbatcher.BatchGenerator(
                        ds=ds[var],
                        input_dims={"time": cfg.loader.batch_size, "rlat": ds.rlat.size, "rlon": ds.rlon.size},
                    )

                    indices = np.arange(len(bgen))
                    partial_paths = [
                        f"{output_path}/{s}/{var}/{res}/{var}_{i}.pt"
                        for i in range(len(indices))
                    ]
                    print(len(indices), ds.time.size)
                    print(bgen[1].values.shape)

                    pool_tuple = zip(
                        indices,
                        partial_paths,
                        bgen,
                        # ds[var].transpose("time", "rlat", "rlon"),
                        # ds_indexed,
                    )
                    if __name__ == "__main__":
                        progress_starmap(
                            parallel_loop,
                            pool_tuple,
                            total=len(indices),
                            n_cpu=24,
                            chunk_size=1,
                        )

        end = timer()
        logging.info(f"Finished {res} dataset in {timedelta(seconds=end-start)}")


if __name__ == "__main__":
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
    # with Client(n_workers=8, threads_per_worker=2, processes=False, dashboard_address=8787, memory_limit='4GB'):
    main()
