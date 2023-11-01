import pytest
import xarray as xr
from nc2pt.computations import (
    user_defined_transform,
    standardize,
    compute_standardization,
    split_and_standardize,
)

from omegaconf import OmegaConf

import numpy as np
from numpy.random import RandomState
from collections import namedtuple
from typing import Dict, Any
import pandas as pd

create_rand = RandomState(1234567890)


# Define a helper class to mimic the ClimateVariable class
class MockClimateVariable:
    def __init__(self, name, transform):
        self.name = name
        self.transform = transform


# Define test cases
test_cases = [
    # Happy path tests
    (
        "happy_path_1",
        xr.Dataset({"temperature": ("time", [15, 20, 25])}),
        MockClimateVariable("temperature", ["x*2"]),
        xr.Dataset({"temperature": ("time", [30, 40, 50])}),
    ),
    (
        "happy_path_2",
        xr.Dataset({"rainfall": ("time", [10, 15, 20])}),
        MockClimateVariable("rainfall", ["x+5"]),
        xr.Dataset({"rainfall": ("time", [15, 20, 25])}),
    ),
    # Edge case tests
    (
        "edge_case_empty_dataset",
        xr.Dataset(),
        MockClimateVariable("temperature", ["x*2"]),
        xr.Dataset(),
    ),
    (
        "edge_case_no_transform",
        xr.Dataset({"temperature": ("time", [15, 20, 25])}),
        MockClimateVariable("temperature", []),
        xr.Dataset({"temperature": ("time", [15, 20, 25])}),
    ),
    # Error case tests
    (
        "error_case_invalid_transform",
        xr.Dataset({"temperature": ("time", [15, 20, 25])}),
        MockClimateVariable("temperature", ["x**"]),
        pytest.raises(SyntaxError),
    ),
    (
        "error_case_nonexistent_variable",
        xr.Dataset({"temperature": ("time", [15, 20, 25])}),
        MockClimateVariable("rainfall", ["x*2"]),
        pytest.raises(KeyError),
    ),
]


@pytest.mark.parametrize(
    "test_id,input_dataset,input_variable,expected_output_or_error", test_cases
)
def test_user_defined_transform(
    test_id, input_dataset, input_variable, expected_output_or_error
):
    # Arrange
    ds = input_dataset
    var = input_variable

    # Act
    if "error" in test_id:
        with expected_output_or_error:
            user_defined_transform(ds, var)
    else:
        result = user_defined_transform(ds, var)

        # Assert
        xr.testing.assert_equal(result, expected_output_or_error)


# Test cases for the happy path, edge cases, and error cases
test_cases = [
    # Happy path tests
    {
        "id": "happy_path_1",
        "input": (xr.DataArray([1, 2, 3]), 2.0, 1.0),
        "expected": xr.DataArray([-1.0, 0.0, 1.0]),
    },
    {
        "id": "happy_path_2",
        "input": (xr.DataArray([10, 20, 30]), 20.0, 10.0),
        "expected": xr.DataArray([-1.0, 0.0, 1.0]),
    },
    # Edge cases
    {
        "id": "edge_case_1",
        "input": (xr.DataArray([0, 0, 0]), 0.0, 1.0),
        "expected": xr.DataArray([0.0, 0.0, 0.0]),
    },
    {
        "id": "edge_case_2",
        "input": (xr.DataArray([-1, 0, 1]), 0.0, 1.0),
        "expected": xr.DataArray([-1.0, 0.0, 1.0]),
    },
    # Error cases
    {
        "id": "error_case_1",
        "input": (xr.DataArray([1, 2, 3]), 0.0, 0.0),
        "expected": ZeroDivisionError,
    },
    {
        "id": "error_case_2",
        "input": (xr.DataArray([1, 2, 3]), "a", 1.0),
        "expected": TypeError,
    },
    {
        "id": "error_case_3",
        "input": (xr.DataArray([1, 2, 3]), 1.0, "a"),
        "expected": TypeError,
    },
]


@pytest.mark.parametrize("case", test_cases, ids=[case["id"] for case in test_cases])
def test_standardize(case):
    # Arrange
    x, mean, std = case["input"]
    expected = case["expected"]

    # Act
    if "error" in case["id"]:
        with pytest.raises(expected):
            standardize(x, mean, std)
    else:
        result = standardize(x, mean, std)

        # Assert
        xr.testing.assert_allclose(result, expected)


# Define a helper function to create a dataset for testing
def create_dataset(varname, data, mean=None, std=None):
    ds = xr.Dataset({varname: (["x"], data)})
    if mean is not None:
        ds[varname].attrs["mean"] = mean
    if std is not None:
        ds[varname].attrs["std"] = std
    return ds


def get_array(size, random_state=42, endpoint=True):
    rng = np.random.default_rng(random_state)
    return rng.random(size)


# Define test parameters
std_mean = namedtuple("std_mean", ["std", "mean"])
params = [
    # Happy path tests
    (
        "happy_path_1",
        create_dataset("var1", np.arange(10)),
        "var1",
        None,
        std_mean(mean=np.arange(10).mean(), std=np.arange(10).std()),
        None,
    ),
    (
        "happy_path_2",
        create_dataset("var2", get_array(100)),
        "var2",
        None,
        std_mean(mean=get_array(100).mean(), std=get_array(100).std()),
        None,
    ),
    (
        "happy_path_3",
        create_dataset("var2", np.arange(10), 0.0, 1.0),
        "var2",
        create_dataset("var2", np.arange(10), 0.0, 1.0),
        std_mean(mean=0.0, std=1.0),
        None,
    ),
    # Edge case tests
    (
        "edge_case_1",
        create_dataset("var1", np.ones(10)),
        "var1",
        None,
        std_mean(mean=1.0, std=1.0),
        pytest.raises(ZeroDivisionError),
    ),
    # Error case tests
    (
        "error_case_1",
        create_dataset("var1", np.arange(10), 0.0, 0.0),
        "var1",
        create_dataset("var1", np.arange(10), 0.0, 0.0),
        std_mean(mean=1.0, std=0.0),
        pytest.raises(ZeroDivisionError),
    ),
    (
        "error_case_2",
        create_dataset("var1", np.arange(10)),
        "var1",
        create_dataset("var1", np.arange(10), mean=5.0),
        std_mean(std=5.0, mean=np.arange(10).std()),
        pytest.raises(KeyError),
    ),
    (
        "error_case_1",
        create_dataset("var1", np.arange(10), 0.0, 0.0),
        "var2",
        create_dataset("var1", np.arange(10), 0.0, 0.0),
        std_mean(mean=1.0, std=0.0),
        pytest.raises(KeyError),
    ),
]


@pytest.mark.parametrize("test_id,ds,varname,precomputed,expected,raises", params)
def test_compute_standardization(test_id, ds, varname, precomputed, expected, raises):
    # Act

    if raises is None:
        result = compute_standardization(ds, varname, precomputed)
        # Arrange
        assert np.isclose(result[varname].attrs["mean"], expected.mean)
        assert np.isclose(result[varname].attrs["std"], expected.std)
    else:
        with raises:
            compute_standardization(ds, varname, precomputed)


test_split_and_standardize_cases = [
    # Happy path
    {
        "id": "happy_path_normal_years",
        "climatedata": OmegaConf.create({"select": {"time": {"test_years": [2000]}}}),
        "var": OmegaConf.create({"name": "var"}),
    },
]


@pytest.mark.parametrize(
    "case",
    test_split_and_standardize_cases,
    ids=[case["id"] for case in test_split_and_standardize_cases],
)
def test_split_and_standardize(case: Dict[str, Any]):
    # Arrange
    ds = xr.Dataset(
        {
            "var": (
                (
                    "lat",
                    "lon",
                    "time",
                ),
                get_array((30, 30, 3)),
            )
        },
        {"time": pd.date_range(start="2000-01-01", periods=3, freq="1AS")},
        {"lat": np.arange(30), "lon": np.arange(30)},
    )

    # Act
    result = split_and_standardize(ds, case["climatedata"], case["var"])

    # Assert
    assert np.isclose(
        result["test"]["var"].attrs["mean"], result["train"]["var"].attrs["mean"]
    ), "Mean of train and test set are not equal."
    assert np.isclose(
        result["test"]["var"].attrs["std"], result["train"]["var"].attrs["std"]
    ), "Std of train and test set are not equal."
