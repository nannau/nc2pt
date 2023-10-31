# A function that creates a xarray dataset with a time dimension from 2000 - 2005
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from hydra import compose, initialize
from hydra.utils import instantiate
from omegaconf import OmegaConf

from nc2pt.align import (
    slice_time,
    train_test_split,
    crop_field,
    interpolate,
    align_grid,
    align_with_lr,
)
from nc2pt.tests.test_data import TestData

initialize(version_base=None, config_path=".")
cfg = compose(config_name="test_config")

# Load test data from config
climate_data = instantiate(cfg)
test_data = TestData(climate_data=climate_data)
hr = test_data.hr_dataset
lr = test_data.lr_dataset
hr_ref = test_data.hr_ref_dataset
dummy_time = test_data.dummy_time_dataset


@pytest.mark.parametrize(
    "ds, expected",
    [
        (lr, xr.Dataset),
        (hr, xr.Dataset),
        (hr_ref, xr.Dataset),
        (dummy_time, xr.Dataset),
    ],
)
def test_type(ds, expected):
    assert isinstance(ds, expected)


# Test cases
test_cases = [
    # Happy path tests
    {
        "id": "happy_path_middle_range",
        "start": "2000-01-04",
        "end": "2000-01-06",
        "expected": dummy_time.sel(time=slice("2000-01-04", "2000-01-06")),
    },
    {
        "id": "happy_path_full_range",
        "start": "2000-01-01",
        "end": "2000-01-10",
        "expected": dummy_time,
    },
    # Edge cases
    {
        "id": "edge_case_single_day",
        "start": "2000-01-01",
        "end": "2000-01-01",
        "expected": dummy_time.sel(time=slice("2000-01-01", "2000-01-01")),
    },
    {
        "id": "edge_case_outside_range",
        "start": "1999-12-31",
        "end": "2000-01-11",
        "expected": dummy_time,
    },
    # Error cases
    {
        "id": "error_case_invalid_date",
        "start": "2000-01-01",
        "end": "2000-01-32",
        "expected": ValueError,
    },
    {
        "id": "error_case_end_before_start",
        "start": "2000-01-10",
        "end": "2000-01-01",
        "expected": ValueError,
    },
]


@pytest.mark.parametrize("case", test_cases, ids=[case["id"] for case in test_cases])
def test_slice_time_behaviour(case: Dict[str, Any]):
    # Arrange
    start = case["start"]
    end = case["end"]
    expected = case["expected"]

    # Act
    if isinstance(expected, xr.Dataset):
        result = slice_time(dummy_time, start, end)
        # Assert
        xr.testing.assert_equal(result, expected)
    else:
        with pytest.raises(expected):
            slice_time(dummy_time, start, end)


slice_time_count_cases = [
    # Happy path
    {
        "id": "happy_path_normal_range",
        "start": "20001001T01:00:00",
        "end": "20150929T23:00:00",
        "expected": 131_447,
    },
    {
        "id": "happy_path_single_hour",
        "start": "20001001T01:00:00",
        "end": "20001001T01:00:00",
        "expected": 1,
    },
    # Edge cases
    {
        "id": "edge_case_single_hour_dual_time",
        "start": "20001001T01:00:00",
        "end": "20001001T02:00:00",
        "expected": 2,
    },
]


@pytest.mark.parametrize("ds", [lr, hr])
@pytest.mark.parametrize(
    "case", slice_time_count_cases, ids=[case["id"] for case in slice_time_count_cases]
)
def test_slice_time_count(ds, case: Dict[str, Any]):
    start = case["start"]
    end = case["end"]
    total_hours = case["expected"]
    ds = slice_time(ds=ds, start=start, end=end)
    assert (
        ds.time.size == total_hours
    ), f"time size mismatch {ds.time.size} vs expected {total_hours}"


# Test train_test_split function
train_test_split_cases = [
    # Happy path
    {"id": "happy_path_normal_years", "years": [2000, 2001], "expected": 19_727},
    # edge case
    {"id": "edge_case_single_year", "years": [2000], "expected": 2208},
    # error case
    {"id": "error_case_empty_years", "years": [], "expected": ValueError},
    {"id": "error_case_invalid_year", "years": 200, "expected": ValueError},
]


@pytest.mark.parametrize(
    "case", train_test_split_cases, ids=[case["id"] for case in train_test_split_cases]
)
def test_train_test_split(case: Dict[str, Any]):
    # Arrange
    ds = xr.Dataset(
        {"var": (("time",), [1, 2, 3])},
        {"time": pd.date_range(start="2000-01-01", periods=3, freq="1AS")},
    )

    if "error" in case["id"]:
        # Act & Assert
        with pytest.raises(case["expected"]):
            train_test_split(ds, case["years"])
    else:
        # Act
        split_ds = train_test_split(ds, case["years"])

        # Assert
        assert "train" in split_ds
        assert "test" in split_ds


x_large = np.ones((10, 100, 100))
x_crop = x_large[:, 0:50, 0:50]
# test dataset for crop_field
crop_field_dataset = xr.Dataset(
    {
        "var": (("time", "rlat", "rlon"), x_large),
        "var2": (("time", "rlat", "rlon"), x_large),
    },
    {
        "time": pd.date_range(start="2000-01-01", periods=10, freq="1AS"),
    },
)

cropped_field_dataset = xr.Dataset(
    {
        "var": (("time", "rlat", "rlon"), x_crop),
        "var2": (("time", "rlat", "rlon"), x_crop),
    },
    {
        "time": pd.date_range(start="2000-01-01", periods=10, freq="1AS"),
    },
)

# Generate tests for crop_field
crop_field_cases = [
    # Happy path
    {
        "id": "happy_path_known_crop",
        "ds": crop_field_dataset,
        "scale_factor": 2,
        "x": OmegaConf.create({"first_index": 0, "last_index": 50}),
        "y": OmegaConf.create({"first_index": 0, "last_index": 50}),
        "expected": cropped_field_dataset,
    },
    # edge case
    {
        "id": "edge_case_unaltered",
        "ds": cropped_field_dataset,
        "scale_factor": 1,
        "x": OmegaConf.create({"first_index": 0, "last_index": 50}),
        "y": OmegaConf.create({"first_index": 0, "last_index": 50}),
        "expected": cropped_field_dataset,
    },
    # error case
    {
        "id": "error_case_not_divisible",
        "ds": lr,
        "scale_factor": 2,
        "x": OmegaConf.create({"first_index": 0, "last_index": -1}),
        "y": OmegaConf.create({"first_index": 0, "last_index": -1}),
        "expected": AssertionError,
    },
]


@pytest.mark.parametrize(
    "case", crop_field_cases, ids=[case["id"] for case in crop_field_cases]
)
def test_crop_field(case: Dict[str, Any]):
    # Arrange
    ds = case["ds"]
    scale_factor = case["scale_factor"]
    x = case["x"]
    y = case["y"]
    expected = case["expected"]

    # Act
    if isinstance(expected, xr.Dataset):
        result = crop_field(ds, scale_factor, x, y)
        # Assert
        xr.testing.assert_equal(result, expected)
    else:
        with pytest.raises(expected):
            crop_field(ds, scale_factor, x, y)


# xesmf is an external package, so we can't really test it here
test_cases_interpolate = [
    # Happy path
    {
        "id": "happy_path_normal_dataset",
        "ds": hr,
        "grid": hr,
        "expected": xr.Dataset,
    },
    # edge case
    # error case
    {
        "id": "error_case_not_dataset",
        "ds": "not a dataset",
        "grid": hr_ref,
        "expected": ValueError,
    },
    {
        "id": "error_case_not_grid",
        "ds": lr,
        "grid": "not a dataset",
        "expected": ValueError,
    },
    {"id": "error_case_no_rlon_rlat", "ds": lr, "grid": lr, "expected": ValueError},
    {
        "id": "error_case_missing_lon",
        "ds": hr.drop_vars("lon"),
        "grid": hr,
        "expected": ValueError,
    },
    {
        "id": "error_case_missing_lat",
        "ds": hr.drop_vars("lat"),
        "grid": hr,
        "expected": ValueError,
    },
    {
        "id": "error_case_missing_lon_grid",
        "ds": hr,
        "grid": hr.drop_vars("lon"),
        "expected": ValueError,
    },
    {
        "id": "error_case_missing_lat_grid",
        "ds": hr,
        "grid": hr.drop_vars("lat"),
        "expected": ValueError,
    },
]


@pytest.mark.parametrize(
    "case", test_cases_interpolate, ids=[case["id"] for case in test_cases_interpolate]
)
def test_interpolate_error_cases(case: Dict[str, Any]):
    # Act
    ds = case["ds"]
    grid = case["grid"]
    expected = case["expected"]
    if "error" in case["id"]:
        # Act & Assert
        with pytest.raises(expected):
            interpolate(ds, grid)
    else:
        # Act
        result = interpolate(ds, grid)
        # Assert
        assert isinstance(result, expected)


test_cases_align_grid = [
    # Happy path
    {
        "id": "happy_path_normal_dataset",
        "ds": hr,
        "grid": hr_ref,
        "climatedata": climate_data,
        "expected": xr.Dataset,
    },
]


@pytest.mark.parametrize(
    "case", test_cases_align_grid, ids=[case["id"] for case in test_cases_align_grid]
)
def test_align_grid(case: Dict[str, Any]):
    # Act
    ds = case["ds"]
    grid = case["grid"]
    climate_data = case["climatedata"]
    expected = case["expected"]
    assert isinstance(align_grid(ds, grid, climate_data), expected)


# Test align with lr
test_cases_align_with_lr = [
    # Happy path
    {
        "id": "happy_path_normal_dataset",
        "ds": hr,
        "grid": hr_ref,
        "climatedata": climate_data,
        "expected": xr.Dataset,
    },
]


@pytest.mark.parametrize(
    "case",
    test_cases_align_with_lr,
    ids=[case["id"] for case in test_cases_align_with_lr],
)
def test_align_with_lr(case: Dict[str, Any]):
    # Act
    ds = case["ds"]
    grid = case["grid"]
    climate_data = case["climatedata"]
    expected = case["expected"]
    assert isinstance(align_with_lr(ds, grid, climate_data), expected)
