import os
import pytest
import xarray as xr
from nc2pt.io import write_to_zarr

# Test data
datasets = [
    xr.Dataset({"data": (("x", "y"), [[1, 2], [3, 4]])}),
    xr.Dataset({"data": (("x", "y"), [[5, 6], [7, 8]])}),
]
paths = ["nc2pt/tests/test_data/test1", "nc2pt/tests/test_data/test2"]

# Test IDs
ids = ["happy_path_1", "happy_path_2"]


@pytest.mark.parametrize("ds, path", zip(datasets, paths), ids=ids)
def test_write_to_zarr(ds, path):
    # Arrange
    expected_history = "Created by /home/nannau/nc2pt/nc2pt/io.py"

    # Act
    write_to_zarr(ds, path)

    # Assert
    result_ds = xr.open_zarr(f"{path}.zarr")
    assert "history" in result_ds.attrs
    assert result_ds.attrs["history"].startswith(expected_history)
    assert os.path.exists(f"{path}.zarr")


# Edge case: Empty dataset
@pytest.mark.parametrize(
    "ds, path",
    [(xr.Dataset(), "nc2pt/tests/test_data/test_empty")],
    ids=["edge_case_empty_dataset"],
)
def test_write_to_zarr_empty_dataset(ds, path):
    # Arrange
    expected_history = "Created by /home/nannau/nc2pt/nc2pt/io.py"

    # Act
    write_to_zarr(ds, path)

    # Assert
    result_ds = xr.open_zarr(f"{path}.zarr")
    assert "history" in result_ds.attrs
    assert result_ds.attrs["history"].startswith(expected_history)
    assert os.path.exists(f"{path}.zarr")
