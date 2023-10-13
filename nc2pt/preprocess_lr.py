import logging
from timeit import default_timer as timer
import hydra
from dask.distributed import Client
import multiprocessing
import xarray as xr
from datetime import timedelta
from dataclasses import dataclass, field
from hydra.utils import instantiate
from typing import Any

from nc2pt.preprocess_helpers import (
    load_grid,
    regrid_align,
    slice_time,
    crop_field,
    coarsen_lr,
    train_test_split,
    homogenize_names,
    match_longitudes,
    compute_standardization,
    write_to_zarr,
    unit_change,
    log_transform,
)


@dataclass
class ClimateVariable:
    varname: str
    alternative_name: str
    path: str
    is_west_negative: bool


# Write a dataclass that loads config data from hydra-core and
# populates the class with the config data.
@dataclass
class ClimateModel:
    cfg: Any
    # These will come from instantiating the class with hydra.
    info: str
    hr_reference_field_path: str
    vars: dict = field(init=False)

    def __post_init__(self):
        self.var_data = {var: instantiate(var) for var in self.vars}
        logging.info(f"Instantiated Model with information: {self.info}")

    @property
    def path(self):
        return self.vars[self.info].path


@dataclass
class ClimateData:
    output_path: str
    models: list
    dims: dict
    coords: dict
    time: dict
    spatial: dict
    engine: str
    randomize: bool
    compute: dict
    loader: dict

    def __post_init__(self):
        self.climmodels = [instantiate(model) for model in self.models]


def before_preprocess(var, climdata):
    """Function to be called before preprocessing.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to preprocess.

    Returns
    -------
    ds : xarray.Dataset
        Dataset after preprocessing.
    """
    ds = load_grid(var.path)

    logging.info("Homogenizing dataset keys...")

    keys = {"data_vars": var.varname, "coords": climdata.coords, "dims": climdata.dims}
    for key_attr, config in keys.items():
        ds = homogenize_names(ds, config, key_attr)

    logging.info("Matching longitudes...")
    ds = match_longitudes(ds) if var.is_west_negative else ds

    return ds


def process_steps_for_lr(ds, climdata):
    # Regrid and align the dataset.
    logging.info("Regridding and interpolating...")
    ds = regrid_align(ds, hr_ref)

    # Crop the field to the given size.
    logging.info("Cropping field...")
    ds = crop_field(
        ds,
        climdata.spatial.scale_factor,
        climdata.spatial.x,
        climdata.spatial.y
    )
    ds = ds.drop(["lat", "lon"])

    # Coarsen the low resolution dataset.
    logging.info("Coarsening low resolution dataset...")
    ds = coarsen_lr(ds, climdata.spatial.scale_factor)

    return ds

def split_and_standardize(ds, climdata, var):
    # Train test split
    logging.info("Splitting dataset...")
    train, test = train_test_split(ds, climdata.time.test_years)

    # Standardize the dataset.
    logging.info(f"Standardizing {var.varname}...")
    train = compute_standardization(train, var.varname)
    test = compute_standardization(test, var.varname, train)

    return train, test

def preprocess_variable(model, climdata):
    if "hr_reference_field" in model.var_data:
        hr_ref_var = model.var_data["hr_reference_field"]
        hr_ref = before_preprocess(hr_ref_var, climdata)
        model.var_data.pop("hr_reference_field")

    for var in model.var_data:
        start = timer()
        logging.info(f"Starting {var} from {model.info} input dataset...")
        ds = before_preprocess(var, climdata)

        logging.info("Slicing time dimension...")
        ds = slice_time(ds, climdata.time.range.start, climdata.time.range.end)

        if "hr_reference_field" in model.var_data:
                ds = process_lr(ds, climdata)

        if var.transform is not None:
            for transform in var.transform:
                func = lambda x: eval(transform)
                logging.info(f"Applying transform {transform} to {var.varname}...")
                ds[var] = xr.apply_ufunc(func, ds[var.varname], dask="parallelized")

        logging.info("Splitting dataset...")
        train, test = split_and_standardize(ds, climdata, var)

        logging.info("Writing test output...")
        write_to_zarr(test, f"{var.output_path}/{var.varname}_test_lr")
        logging.info("Writing train output...")
        write_to_zarr(train, f"{var.output_path}/{var.varname}_train_lr")

        end = timer()
        logging.info(f"Done processing {model.info}!")
        logging.info(f"Time elapsed: {timedelta(seconds=end-start)}")


def preprocess(climdata):
    for model in climdata.climmodels:
        preprocess_variable(model, climdata)



@hydra.main(version_base=None, config_path="conf", config_name="config")
def start(cfg) -> None:
    cores = int(multiprocessing.cpu_count())
    logging.info(f"Using {cores} cores")

    with Client(
        # n_workers=16,
        # threads_per_worker=2,
        processes=False,
        # memory_limit="16GB",
        dashboard_address=cfg.compute.dashboard_address,
    ):
        climdata = instantiate(cfg.data, _recursive_=True)
        preprocess(climdata)
