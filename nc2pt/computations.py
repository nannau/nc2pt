import logging
import xarray as xr
import numpy as np
from nc2pt.align import train_test_split


def user_defined_transform(ds, var) -> xr.Dataset:
    """Apply user defined transform to the dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to apply transform to.
    var : ClimateVariable
        Climate variable to apply transform to.

    Returns
    -------
    ds : xarray.Dataset
        Dataset after transform has been applied.
    """
    for transform in var.transform:
        func = lambda x: eval(transform)
        logging.info(f"Applying transform {transform} to {var.name}...")
        ds[var.name] = xr.apply_ufunc(func, ds[var.name], dask="parallelized").compute()

    return ds


def standardize(x: xr.DataArray, mean: float, std: float) -> xr.DataArray:
    """Standardize the data.

    Parameters
    ----------
    x : xarray.DataArray
        Data to standardize.
    mean : float
        Mean of the data.
    std : float
        Standard deviation of the data.

    Returns
    -------
    x : xarray.DataArray
        Standardized data.
    """
    return (x - mean) / std


def compute_standardization(
    ds: xr.Dataset, varname: str, precomputed: xr.Dataset = None
) -> xr.Dataset:  # sourcery skip: avoid-builtin-shadow
    """Standardize the statistics of the dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to standardize the statistics of.
    precomputed : xarray.Dataset, optional
        Dataset containing precomputed statistics, by default None
    Returns
    -------
    ds : xarray.Dataset
        Dataset with standardized statistics.
    """
    logging.info("Computing mean and standard deviation...")
    if precomputed is None:
        logging.info("Calculation mean...")
        mean = ds[varname].mean().compute()
        logging.info("Calculation std...")
        std = ds[varname].std().compute()
    else:
        mean = precomputed[varname].attrs["mean"]
        std = precomputed[varname].attrs["std"]

    logging.info("Applying function...")
    ds[varname] = xr.apply_ufunc(
        standardize,
        ds[varname],
        mean,
        std,
        dask="parallelized",
    )

    ds[varname].attrs["mean"] = float(mean)
    ds[varname].attrs["std"] = float(std)

    return ds


def split_and_standardize(ds, climdata, var) -> dict:
    # Train test split
    logging.info("Splitting dataset...")
    train_test = train_test_split(ds, climdata.select.time.test_years)

    # Standardize the dataset.
    logging.info(f"Standardizing {var.name}...")
    train = compute_standardization(train_test["train"], var.name)
    test = compute_standardization(train_test["test"], var.name, train_test["train"])

    return {"train": train, "test": test}
