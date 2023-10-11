import logging
import os
import dask

from ClimatExPrep import preprocess_lr, preprocess_hr


if __name__ == "__main__":
    with dask.config.set(**{"array.slicing.split_large_chunks": False}):
        logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
        # preprocess_lr.start()
        preprocess_hr.start()
