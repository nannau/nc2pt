from dask.distributed import Client
import torch
from datetime import timedelta
from timeit import default_timer as timer
import numpy as np
import hydra
import logging
import os
import glob
from multiprocessing import Pool
from joblib import Parallel, delayed
from parallelbar import progress_imap, progress_starmap, progress_imapu

from multiprocessing import set_start_method
from multiprocessing import get_context

torch.manual_seed(0)

def parallel_loop(sub_i, path, output_path):
    sub_path = [f"{path}_{j}.pt" for j in sub_i]
    stack = [torch.load(sub) for sub in sub_path]
    x = torch.stack(stack, dim=0)
    assert not torch.isnan(x).any(), f"NaNs found in {i}"
    torch.save(x, output_path)

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg) -> None:
    # Define for loop that iterates over sets, resolutions, and variables
    # and saves each time step as a torch tensor to write to a pytorch file
    # format.

    indices = torch.randperm(len(glob.glob(f"{cfg.vars['uas'].output_path}/train/uas/lr/*.pt")))
    indices = torch.split(
        indices, cfg.loader.batch_size, dim=0
    )

    for res in ["lr", "hr"]:
        start = timer()
        for s in ["train"]:
            logging.info(f"Loading {s} {res} dataset...")
            for var in cfg.vars:
                output_path = cfg.vars[var].output_path
                if not os.path.exists(f"{output_path}/batched/{s}/{var}/{res}"):
                    logging.info(
                        f"Creating directory: {output_path}/batched/{s}/{var}/{res}"
                    )
                    os.makedirs(f"{output_path}/batched/{s}/{var}/{res}")

                partial_paths = [
                    f"{output_path}/{s}/{var}/{res}/{var}"
                    for _ in range(len(indices))
                ]

                output_path = [
                    f"{output_path}/batched/{s}/{var}/{res}/{var}_{i}.pt"
                    for i in range(len(indices))
                ]



                pool_tuple = zip(
                    indices,
                    partial_paths,
                    output_path
                )

                # if __name__ == "__main__":
                with get_context("spawn").Pool() as pool:
                    # progress_starmap(
                    pool.starmap(
                        parallel_loop,
                        pool_tuple,
                        # total=len(indices),
                        # n_cpu=16,
                    )
                logging.info(f"Completed {s} {res} {var} batching")

        end = timer()
        logging.info(f"Finished {res} dataset in {timedelta(seconds=end-start)}")


if __name__ == "__main__":
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
    # with Client(n_workers=8, threads_per_worker=2, processes=False, dashboard_address=8787, memory_limit='4GB'):
    main()
