from ClimatExPrep.preprocess_helpers import (
    MissingKeys,
    MultipleKeys,
    load_grid,
    regrid_align,
    slice_time,
    train_test_split,
    homogenize_names,
    match_longitudes,
)

from hydra import compose, initialize
import pytest
import xarray as xr

# Global vars test dat

# Configuration from test_config.yaml
cfg_path = "."
cfg_name = "test_config"
initialize(config_path=cfg_path, version_base="1.1.0")
cfg = compose(config_name=cfg_name)

# Load data with regrid_align.load_grid and test in later stage
lr_data = load_grid(cfg.lr.path, engine="h5netcdf")
hr_data = load_grid(cfg.hr.path, engine="h5netcdf")

# Create bad lr data with multiple keys
multiple_lr_keys = lr_data.copy()
multiple_lr_keys["u10"] = lr_data["uas"]
# Create bad lr data with missing keys
missing_lr_keys = lr_data.copy()
missing_lr_keys = missing_lr_keys.drop_vars("uas")

# Create bad hr data with multiple keys
multiple_hr_keys = hr_data.copy()
multiple_hr_keys = multiple_hr_keys.rename({"U10": "uas"})
multiple_hr_keys["u10"] = multiple_hr_keys["uas"]

# Create bad hr data with missing keys
missing_hr_keys = hr_data.copy()
missing_hr_keys = missing_hr_keys.drop_vars("U10")


@pytest.mark.parametrize(
    "ds, expected",
    [
        (lr_data, xr.Dataset),
        (hr_data, xr.Dataset)
    ]
)
def test_type(ds, expected):
    assert isinstance(ds, expected)


@pytest.mark.parametrize(
    "ds", [lr_data, hr_data]
)
@pytest.mark.parametrize(
    "start, end, total_hours",
    [
        ("20001001T01:00:00", "20150929T23:00:00", 131_447),
        ("20001001T01:00:00", "20001001T01:00:00", 1)
    ]
)
def test_slice_time(ds, start, end, total_hours):
    ds = slice_time(ds=ds, start=start, end=end)
    assert ds.time.size == total_hours


@pytest.mark.parametrize(
    "ds, grid",
    [
        (lr_data, hr_data),
    ]
)
def test_regrid_align(ds, grid):
    ds = regrid_align(ds=ds, grid=grid)
    assert ds.rlat.size == grid.rlat.size, "rlat size mismatch"
    assert ds.rlon.size == grid.rlon.size, "rlon size mismatch"


@pytest.mark.parametrize(
    "ds", [lr_data, hr_data]
)
@pytest.mark.parametrize(
    "years, expected_size",
    [
        ([2000, 2001, 2002], 19_727),
    ]
)
def test_train_test_split(ds, years, expected_size):
    ds = slice_time(ds, start="20001001T01:00:00", end="20150929T23:00:00")
    train, test = train_test_split(ds=ds, years=years)
    assert train.time.size == 131_447 - expected_size, "train size mismatch"
    assert test.time.size == expected_size, "test size mismatch"


@pytest.mark.parametrize(
    "ds", [lr_data, hr_data]
)
@pytest.mark.parametrize(
    "config, key_attr, expected",
    [
        (cfg.vars, "data_vars", ["U10", "V10"]),
        (cfg.coords, "coords", ["lat", "lon"]),
        (cfg.dims, "dims", ["time"]),
    ]
)
def test_good_homogenize_names(ds, config, key_attr, expected):
    ds = homogenize_names(ds=ds, keymaps=config, key_attr=key_attr)
    for key in expected:
        assert key in getattr(ds, key_attr), f"{key} not in dataset"


bad_vars_multiple_alt = {"U10": {
    "alternative_names": ["u10", "uas"]
    }
}


@pytest.mark.parametrize(
    "ds, config, key_attr, expected",
    [
        (missing_lr_keys, cfg.vars, "data_vars", MissingKeys),
        (missing_hr_keys, cfg.vars, "data_vars", MissingKeys),
        (multiple_lr_keys, cfg.vars, "data_vars", MultipleKeys),
        (multiple_hr_keys, cfg.vars, "data_vars", MultipleKeys),
    ]
)
def test_homogonize_missing_or_multiple_names(ds, config, key_attr, expected):
    # Note that tas is included as an LR variable only in the config.
    # This round of testing accounts for the lr_only: True flag in the config
    with pytest.raises(expected):
        ds = homogenize_names(ds=ds, keymaps=config, key_attr=key_attr)


@pytest.mark.parametrize(
    "ds", [hr_data]
)
def test_match_longitudes(ds):
    hr = homogenize_names(ds=ds, keymaps=cfg.coords, key_attr="coords")
    hr = match_longitudes(hr) if cfg.hr.is_west_negative else hr
    assert hr.lon.min() >= 0, "min longitude is negative"
    assert hr.lon.max() <= 360, "max longitude is greater than 360"
