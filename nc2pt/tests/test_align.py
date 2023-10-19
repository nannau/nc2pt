# A function that creates a xarray dataset with a time dimension from 2000 - 2005
from typing import Any, Dict

import hydra
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from hydra import compose, initialize
from hydra.utils import instantiate

from nc2pt.align import slice_time, train_test_split
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

# Simple test on the test data itself!

@pytest.mark.parametrize(
    "ds, expected",
    [
        (lr, xr.Dataset),
        (hr, xr.Dataset),
        (hr_ref, xr.Dataset),
        (dummy_time, xr.Dataset),
    ]
)
def test_type(ds, expected):
    assert isinstance(ds, expected)


# Test cases
test_cases = [
    # Happy path tests
    {'id': 'happy_path_middle_range', 'start': '2000-01-04', 'end': '2000-01-06', 'expected': dummy_time.sel(time=slice('2000-01-04', '2000-01-06'))},
    {'id': 'happy_path_full_range', 'start': '2000-01-01', 'end': '2000-01-10', 'expected': dummy_time},
    # Edge cases
    {'id': 'edge_case_single_day', 'start': '2000-01-01', 'end': '2000-01-01', 'expected': dummy_time.sel(time=slice('2000-01-01', '2000-01-01'))},
    {'id': 'edge_case_outside_range', 'start': '1999-12-31', 'end': '2000-01-11', 'expected': dummy_time},
    # Error cases
    {'id': 'error_case_invalid_date', 'start': '2000-01-01', 'end': '2000-01-32', 'expected': ValueError},
    {'id': 'error_case_end_before_start', 'start': '2000-01-10', 'end': '2000-01-01', 'expected': ValueError},
]

@pytest.mark.parametrize('case', test_cases, ids=[case['id'] for case in test_cases])
def test_slice_time_behaviour(case: dict[str, Any]):
    # Arrange
    start = case['start']
    end = case['end']
    expected = case['expected']

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
    {'id': 'happy_path_normal_range', 'start': '20001001T01:00:00', 'end': '20150929T23:00:00', 'expected': 131_447},
    {'id': 'happy_path_single_hour', 'start': '20001001T01:00:00', 'end': '20001001T01:00:00', 'expected': 1},
    # Edge cases
    {'id': 'edge_case_single_hour_dual_time', 'start': '20001001T01:00:00', 'end': '20001001T02:00:00', 'expected': 2},
]


@pytest.mark.parametrize(
    "ds", [lr, hr]
)
@pytest.mark.parametrize('case', slice_time_count_cases, ids=[case['id'] for case in slice_time_count_cases])
def test_slice_time_count(ds, case: dict[str, Any]):
    start = case['start']
    end = case['end']
    total_hours = case['expected']
    ds = slice_time(ds=ds, start=start, end=end)
    assert ds.time.size == total_hours, f"time size mismatch {ds.time.size} vs expected {total_hours}"


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

@pytest.mark.parametrize('case', train_test_split_cases, ids=[case['id'] for case in train_test_split_cases])
def test_train_test_split(case: dict[str, Any]):

    # Arrange
    ds = xr.Dataset({"var": (("time",), [1, 2, 3])}, {"time": pd.date_range(start="2000-01-01", periods=3, freq="1AS")})

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


# Generate tests for crop_field
crop_field_cases = [
    # Happy path
    {"id": "happy_path_normal_crop", "ds": lr, "scale_factor": 2, "x": 2, "y": 2, "expected": AssertionError},
    # edge case
    {"id": "edge_case_single_year", "scale_factor": 1, "x": 1, "y": 1, "expected": 1},
    # error case
    {"id": "error_case_empty_years", "scale_factor": 0, "x": 0, "y": 0, "expected": ValueError},
    {"id": "error_case_invalid_year", "scale_factor": -1, "x": -1, "y": -1, "expected": ValueError},
]
@pytest.mark.parametrize('case', crop_field_cases, ids=[case['id'] for case in crop_field_cases])
def test_crop_field(case: dict[str, Any]):


# split_test_cases = [
#     # Happy path
#     {"id": "happy_path_normal_years", "years": [2000, 2001, 2002], "expected": 19_727},
#     # edge case
#     {"id": "edge_case_single_year", "years": [2000], "expected": 2208},
#     # error case
#     {"id": "error_case_empty_years", "years": [], "expected": ValueError},
#     {"id": "error_case_invalid_year", "years": 200, "expected": ValueError},
# ]

# @pytest.mark.parametrize("ds",  [test_data.lr_dataset, test_data.hr_dataset, test_data.hr_ref_dataset])
# @pytest.mark.parametrize('case', split_test_cases, ids=[case['id'] for case in split_test_cases])
# def test_train_test_split(ds, case: dict[str, Any]):
#     ds = slice_time(ds, start="20001001T01:00:00", end="20150929T23:00:00")
#     train, test = train_test_split(ds=ds, years=years)
#     assert train.time.size == 131_447 - case["expected"], "train size mismatch"
#     assert test.time.size == case["expected"], "test size mismatch"


# def test_with_initialize() -> None:
#     with initialize(version_base=None, config_path="."):
#         # config is relative to a module
#         cfg = compose(config_name="test_config")
#         # Load test data from config
#         climate_data = instantiate(cfg)
#         hr_ref = xr.open_dataset(climate_data.climate_models[1].hr_ref.path)
#         print(hr_ref)


# if __name__ == "__main__":
#     test_with_initialize()