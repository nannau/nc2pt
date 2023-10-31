import pytest
import xarray as xr
from nc2pt.computations import user_defined_transform


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

    if "error" in test_id:
        with expected_output_or_error:
            user_defined_transform(ds, var)

    # # Act
    # if isinstance(expected_output_or_error, type) and issubclass(
    #     expected_output_or_error, Exception
    # ):
    #     with expected_output_or_error:
    #         user_defined_transform(ds, var)
    else:
        result = user_defined_transform(ds, var)

        # Assert
        xr.testing.assert_equal(result, expected_output_or_error)
