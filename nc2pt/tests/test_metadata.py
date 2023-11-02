from nc2pt.metadata import (
    MultipleKeys,
    MissingKey,
    configure_metadata,
    rename_keys,
    match_longitudes,
)

import xarray as xr
from nc2pt.climatedata import ClimateDimension, ClimateVariable, ClimateData


from nc2pt.tests.test_data import TestData
from hydra import compose, initialize
from hydra.utils import instantiate
import pytest

# Initialize Hydra
cfg = compose(config_name="test_config")

# Load test data from config
climate_data = instantiate(cfg)

# # Load test data from config
# test_data = TestData(climate_data=climate_data)
# hr = test_data.hr_dataset
# lr = test_data.lr_dataset

# # Create bad lr data with multiple keys
# multiple_lr_keys = lr.copy()
# multiple_lr_keys["u10"] = lr["uas"]
# # Create bad lr data with missing keys
# missing_lr_keys = lr.copy()
# missing_lr_keys = missing_lr_keys.drop_vars("uas")

# # Create bad hr data with multiple keys
# multiple_hr_keys = hr.copy()
# multiple_hr_keys = multiple_hr_keys.rename({"U10": "uas"})
# multiple_hr_keys["u10"] = multiple_hr_keys["uas"]

# # Create bad hr data with missing keys
# missing_hr_keys = hr.copy()
# missing_hr_keys = missing_hr_keys.drop_vars("U10")


# @pytest.mark.parametrize("ds", [lr, hr])
# @pytest.mark.parametrize(
#     "config, key_attr, expected",
#     [
#         (cfg.vars, "data_vars", ["U10", "V10"]),
#         (cfg.coords, "coords", ["lat", "lon"]),
#         (cfg.dims, "dims", ["time"]),
#     ],
# )
# def test_good_configure_metadata(ds, config, key_attr, expected):
#     ds = configure_metadata(ds=ds, keymaps=config, key_attr=key_attr)
#     for key in expected:
#         assert key in getattr(ds, key_attr), f"{key} not in dataset"


# bad_vars_multiple_alt = {"U10": {"alternative_names": ["u10", "uas"]}}


# @pytest.mark.parametrize(
#     "ds, config, key_attr, expected",
#     [
#         (missing_lr_keys, cfg.vars, "data_vars", MissingKeys),
#         (missing_hr_keys, cfg.vars, "data_vars", MissingKeys),
#         (multiple_lr_keys, cfg.vars, "data_vars", MultipleKeys),
#         (multiple_hr_keys, cfg.vars, "data_vars", MultipleKeys),
#     ],
# )
# def test_homogonize_missing_or_multiple_names(ds, config, key_attr, expected):
#     # Note that tas is included as an LR variable only in the config.
#     # This round of testing accounts for the lr_only: True flag in the config
#     with pytest.raises(expected):
#         ds = configure_metadata(ds=ds, keymaps=config, key_attr=key_attr)


# @pytest.mark.parametrize("ds", [hr_data])
# def test_match_longitudes(ds):
#     hr = homogenize_names(ds=ds, keymaps=cfg.coords, key_attr="coords")
#     hr = match_longitudes(hr) if cfg.hr.is_west_negative else hr
#     assert hr.lon.min() >= 0, "min longitude is negative"
#     assert hr.lon.max() <= 360, "max longitude is greater than 360"


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
                )
            }
        ),
        "var": var,
        "climdata": climdata,
        "expected": xr.Dataset(
            data_vars={
                "temperature": (
                    ("time", "rlat", "rlon"),
                    [[[1, 2], [3, 4]], [[1, 2], [3, 4]]],
                )
            }
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
