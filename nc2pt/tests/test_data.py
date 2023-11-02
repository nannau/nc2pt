from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr
from hydra.utils import instantiate

from nc2pt.climatedata import ClimateData
from nc2pt.io import load_grid


@dataclass
class TestData:
    """Class for testing ClimateData class"""

    climate_data: ClimateData

    def __post_init__(self):
        self.climate_data = instantiate(self.climate_data)
        self.climate_models = [
            instantiate(cm) for cm in self.climate_data.climate_models
        ]

    @property
    def hr_ref_dataset(self):
        return load_grid(self.climate_models[1].hr_ref.path, engine="h5netcdf")

    @property
    def hr_dataset(self):
        climate_variables = [
            instantiate(v) for v in self.climate_models[0].climate_variables
        ]
        var_paths = [v.path for v in climate_variables]

        return load_grid(var_paths, engine="h5netcdf")

    @property
    def lr_dataset(self):
        climate_variables = [
            instantiate(v) for v in self.climate_models[1].climate_variables
        ]
        var_paths = [v.path for v in climate_variables]

        return load_grid(var_paths)

    @property
    def dummy_time_dataset(self):
        times = pd.date_range("2000-01-01", periods=10, freq="D")
        data = np.random.rand(10, 10, 10)
        return xr.Dataset(
            {"var": (("time", "x", "y"), data)},
            coords={"time": times, "x": range(10), "y": range(10)},
        )
