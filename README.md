# ClimatExPrep
![example workflow](https://github.com/nannau/ClimatExPrep/actions/workflows/python-package-conda.yml/badge.svg?event=push)
[![codecov](https://codecov.io/gh/nannau/ClimatExPrep/branch/main/graph/badge.svg?token=XXRLD3076V)](https://codecov.io/gh/nannau/ClimatExPrep)

A repository of software for preprocessing climatex data for the deep learning pipeline.

### Requirements
* [Miniconda](https://docs.conda.io/en/latest/miniconda.html) with Python >= 3.7
* [xESMF](https://xesmf.readthedocs.io/en/latest/)


### 💽 Installation
xESMF is only available through Conda, so you will have to be able to install conda on your system. Unfortunately, this is limiting because certain HPCs don't allow conda. There may be workarounds but I have not explored them. 

Using the `envionment.yml` file in this repo, you can create a new conda environment with the requirements with
```bash
conda env create -f environment.yml
conda activate ClimatExPrep
```
### 📋 Configuration
Please see `ClimatExPrep/conf/config.yml` for an example configuration.

```yaml
hr:
  path: /home/nannau/jabbar/data/wrf/*.nc
  is_west_negative: true
  output_path: /home/nannau/gom/wrf
lr:
  path: /home/nannau/jabbar/data/era5/*.nc
  is_west_negative: false
  output_path: /home/nannau/gom/era5

# If the variables are the same, but are named differently in
# the netCDF files, use the yaml key as the "homogenizing" key, and
# list the alternative name in "alternative_names".
# This pipeline will rename the alternative name to the yaml key if found.
# E.g. if the lr and hr datasets have different
# numbers of covariates, e.g. Annau et al. 2023, then
# you can specify if a variable only occurs in the hr or lr
# dataset respectively with hr_only or lr_only: true
# The same logic above applies for the dimensions, dims, and coordinates, coords.
vars:
  U10:
    alternative_names: ["u10", "uas"]
    standardize: true
    standardize_style: "normal"
  V10:
    alternative_names: ["v10", "vas"]
    standardize: true
    standardize_style: "normal"
dims:
  time:
    alternative_names: ["Time", "Times", "times"]
  rlat:
    alternative_names: ["rotated_latitude"]
    hr_only: true
  rlon:
    alternative_names: ["rotated_longitude"]
    hr_only: true
coords:
  lat:
    alternative_names: ["latitude", "Lat", "Latitude"]
  lon:
    alternative_names: ["longitude", "Long", "Lon", "Longitude"]

# Time indexing for subsets
time:
  # Crop to the dataset with the shortest run
  # this defines the full dataset from which to subset
  full:
    start: "20001001T01:00:00"
    end: "20150929T23:00:00"
  # use this to select which years to reserve for testing
  # the remaining years in full will be used for training
  test_years: [2000, 2009, 2014]

# sets the scale factor and index slices of the rotated coordinates
spatial:
  scale_factor: 8
  x:
    first_index: 0
    last_index: 632
  y:
    first_index: 2
    last_index: 634

# xarray netcdf engine
engine: h5netcdf

# dask client parameters
compute:
  threads_per_worker: 2
  memory_limit: "20GB"
  dashboard_address: 8787

```

### 🚀 Running
You can just edit the configuration file above and run
```bash
python ClimatExPrep/preprocess.py
```

---
High-level workflow
![image](https://user-images.githubusercontent.com/10455520/218364372-ce2f6f7a-7917-4601-b06a-03f56feea423.png)
