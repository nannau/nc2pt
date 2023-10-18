from dataclasses import dataclass, field
import logging
from typing import Any, Optional


@dataclass
class ClimateDimension:
    name: str
    alternative_name: list[str]
    chunksize: Optional[int] = None


@dataclass
class ClimateVariable:
    varname: str
    alternative_name: str
    path: str
    is_west_negative: bool
    transform: Optional[str] = field(default=None)
    invariant: Optional[bool] = field(default=False)


# Write a dataclass that loads config data from hydra-core and
# populates the class with the config data.
@dataclass
class ClimateModel:
    # These will come from instantiating the class with hydra.
    info: str
    name: str
    climate_variables: list[ClimateVariable]
    hr_ref: Optional[str] = None

    def __post_init__(self):
        logging.info(f"Instantiated Model with information: {self.info}")


@dataclass
class ClimateData:
    output_path: str
    climate_models: list[ClimateModel]
    dims: list[ClimateDimension]
    coords: list[ClimateDimension]
    select: dict
    compute: dict
    loader: dict
