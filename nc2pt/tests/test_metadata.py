from nc2pt.metadata import (
    MultipleKeys,
    configure_metadata,
    match_longitudes,
)

import xarray as xr
from nc2pt.climatedata import ClimateDimension, ClimateVariable, ClimateData

from hydra import compose
from hydra.utils import instantiate
import pytest

# Initialize Hydra
cfg = compose(config_name="test_config")

# Load test data from config
climate_data = instantiate(cfg)

# Test data
dims = [
    ClimateDimension(name="time", alternative_names=["Time"]),
    ClimateDimension(name="rlat", alternative_names=["rotated_latitude"]),
    ClimateDimension(name="rlon", alternative_names=["rotated_longitude"]),
]
coords = [
    ClimateDimension(name="lat", alternative_names=["Latitude"]),
    ClimateDimension(name="lon", alternative_names=["Longitude"]),
]
var = ClimateVariable(
    name="temperature",
    is_west_negative=False,
    path="this/is/a/path",
    alternative_names=[],
)
climdata = ClimateData(
    dims=dims,
    coords=coords,
    output_path="this/is/a/path",
    climate_models=[],
    select={},
    compute={},
    loader={},
)

# Test cases
test_cases = [
    # Happy path
    {
        "id": "happy_path",
        "ds": xr.Dataset(),
        "var": var,
        "climdata": climdata,
        "expected": xr.Dataset(),
    },
    {
        "id": "happy_path_rename",
        "ds": xr.Dataset(data_vars={"temperature": (("x", "y"), [[1, 2], [3, 4]])}),
        "var": ClimateVariable(
            name="tas",
            is_west_negative=False,
            path="this/is/a/path",
            alternative_names=["temperature"],
        ),
        "climdata": climdata,
        "expected": xr.Dataset(data_vars={"tas": (("x", "y"), [[1, 2], [3, 4]])}),
    },
    # Edge case: var.name is 'hr_ref'
    {
        "id": "edge_case_var_name_hr_ref",
        "ds": xr.Dataset(),
        "var": ClimateVariable(
            name="hr_ref",
            path="this/is/a/path",
            alternative_names=[],
            is_west_negative=False,
        ),
        "climdata": climdata,
        "expected": xr.Dataset(),
    },
    # Edge case: check that variable is removed with size one
    {
        "id": "edge_case_var_size_one",
        "ds": xr.Dataset(
            data_vars={
                "temperature": (
                    ("extra_dim", "time", "rlat", "rlon"),
                    [[[[1, 2], [3, 4]], [[1, 2], [3, 4]]]],
                ),
            },
            coords={"extra_dim": [0], "time": [0, 1]},
        ),
        "var": var,
        "climdata": climdata,
        "expected": xr.Dataset(
            data_vars={
                "temperature": (
                    ("time", "rlat", "rlon"),
                    [[[1, 2], [3, 4]], [[1, 2], [3, 4]]],
                )
            },
            coords={"time": [0, 1]},
        ),
    },
    # Error case: var.name is not in ds
    {
        "id": "error_case_multiple_keys",
        "ds": xr.Dataset(
            data_vars={
                "temperature": (("x", "y"), [[1, 2], [3, 4]]),
                "wind_speed": (("x", "y"), [[1, 2], [3, 4]]),
            }
        ),
        "var": ClimateVariable(
            name="tas",
            is_west_negative=False,
            path="this/is/a/path",
            alternative_names=["temperature", "wind_speed"],
        ),
        "climdata": climdata,
        "expected": MultipleKeys,
    },
]


@pytest.mark.parametrize("case", test_cases, ids=[case["id"] for case in test_cases])
def test_configure_metadata(case):
    # Arrange
    ds = case["ds"]
    var = case["var"]
    climdata = case["climdata"]

    if "error" not in case["id"]:
        # Act
        result = configure_metadata(ds, var, climdata)

        # Assert
        assert result.equals(case["expected"])
    else:
        with pytest.raises(case["expected"]):
            configure_metadata(ds, var, climdata)


test_cases = [
    {
        "id": "happy_path_west_postitive",
        "ds": xr.Dataset(
            data_vars={"temperature": (("lat", "lon"), [[1, 2], [3, 4]])},
            coords={"lat": [0, 1], "lon": [0, 1]}
        ),
        "is_west_negative": False,
        "expected": xr.Dataset(
            data_vars={"temperature": (("lat", "lon"), [[1, 2], [3, 4]])},
            coords={"lat": [0, 1], "lon": [0, 1]}
        ),
    },
    {
        "id": "happy_path_west_negative",
        "ds": xr.Dataset(
            data_vars={"temperature": (("lat", "lon"), [[1, 2], [3, 4]])},
            coords={"lat": [0, 1], "lon": [-180, 0]}
        ),
        "is_west_negative": True,
        "expected": xr.Dataset(
            data_vars={"temperature": (("lat", "lon"), [[1, 2], [3, 4]])},
            coords={"lat": [0, 1], "lon": [180, 360]}
        ),
    },
    {
        "id": "error_case_min_max_lontitude_out_of_bounds",
        "ds": xr.Dataset(
            data_vars={"temperature": (("lat", "lon"), [[1, 2], [3, 4]])},
            coords={"lat": [0, 1], "lon": [0, 181]}
        ),
        "is_west_negative": True,
        "expected": ValueError,
    }
]


@pytest.mark.parametrize("case", test_cases, ids=[case["id"] for case in test_cases])
def test_match_longitudes(case):
    # Arrange
    ds = case["ds"]
    is_west_negative = case["is_west_negative"]
    ds = match_longitudes(ds) if is_west_negative else ds
    expected = case["expected"]

    # Act
    if "error" in case["id"]:
        with pytest.raises(expected):
            match_longitudes(ds)
    else:
        # Assert
        assert ds.equals(expected)
        assert ds.lon.min() >= 0, "min longitude is negative"
        assert ds.lon.max() <= 360, "max longitude is greater than 360"
