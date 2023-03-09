import torch
from datetime import timedelta
from timeit import default_timer as timer
from tqdm import tqdm
import hydra
import glob
import logging
import os


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg) -> None:

    torch.manual_seed(cfg.loader.seed)

    for res in ["lr", "hr"]:
        start = timer()
        for s in ["train", "test"]:
            for var in cfg.vars:
                if "lr_only" in cfg.vars[var] and res == "hr" and cfg.vars[var].lr_only:
                    continue
                if "hr_only" in cfg.vars[var] and res == "lr" and cfg.vars[var].hr_only:
                    continue

                logging.info(f"Loading {s} {res} dataset...")
                nfiles = len(glob.glob(f"{cfg[res].output_path}/{s}/{var}/{var}_*.pt"))

                if cfg.loader.randomize:
                    # Randomize the order of the files
                    logging.info("Split the files into random batches...")
                    split = torch.split(torch.randperm(nfiles), cfg.loader.batch_size)
                else:
                    # Split the files into batches
                    logging.info("Split the files int0 non-random batches")
                    split = torch.split(torch.arange(nfiles), cfg.loader.batch_size)

                # Create parent dir if it doesn't exist for each variable
                output_dir = f"{cfg[res].output_path}/{s}/{var}_batch_{cfg.loader.batch_size}"
                if not os.path.exists(output_dir):
                    logging.info(f"Creating directory {output_dir} because it does not exist...")
                    os.makedirs(output_dir)
                logging.info(f"Batching {cfg[res].output_path}/{s}/{var}/...")
                for i, batch in enumerate(tqdm(split, desc=f"Creating batches for {s} {res} dataset")):
                    batchlist = [
                        torch.load(
                            f"{cfg[res].output_path}/{s}/{var}/{var}_{i}.pt"
                        )
                        for i in batch
                    ]
                    write_batch = torch.stack(batchlist)
                    torch.save(write_batch, f"{cfg[res].output_path}/{s}/{var}_batch_{cfg.loader.batch_size}/batch_{i}.pt")

        end = timer()
        logging.info(f"Finished {res} dataset in {timedelta(seconds=end-start)}")


if __name__ == "__main__":
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
    main()
