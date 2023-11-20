<p align="center">
  <img src="https://user-images.githubusercontent.com/10455520/280422419-5f4c4a78-5811-439d-b861-9d193ffb8902.png" width="250" height="250" /> 
</p>

![example workflow](https://github.com/nannau/ClimatExPrep/actions/workflows/python-package-conda.yml/badge.svg?event=push)
[![codecov](https://codecov.io/gh/nannau/nc2pt/graph/badge.svg?token=XXRLD3076V)](https://codecov.io/gh/nannau/nc2pt)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)


![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)

## The Problem
NetCDF4 files, commonly used for storing climate and earth systems data, are not optimized for use with most machine learning applications where small amounts of data are require quickly and frequently. 

## How does nc2pt help?
It performs a preprocessing flow on climate fields and converts them from NetCDF4 (`.nc`) to an intermediate file format Zarr (`.zarr`) which allows for the parallel loading and writing to individual PyTorch Lightning files (`.pt`) that can be loaded directly onto GPUs.

## What intended use cases of nc2pt?
* standardizing and making metadata uniform between datasets

* aligns different grids perfectly by re-projecting them onto one another -- nc2pt projects the low-resolution (lr) regular grids onto the high-resolution curvilinear grids (hr). nc2pt assumes the curvilinear dimensions are like `rlat` or `rlon`. It was originally designed to support super-resolution problems.

* selects individual years as test years or training years

* organizes code into input (lr) or output (hr) fields

* meant for use with large datasets ont he order of hundreds of GB

## What preprocessing steps does nc2pt do? 🤔

High-level workflow
![image](https://github.com/nannau/nc2pt/assets/10455520/869fa159-5dc0-4e81-90b0-d7fa5d35c96a)

1. configures metadata between the datasets as defined in the config
2. slices data to a pre-determined range of dates
3. aligns the grids via interpolation, crops them to be the same size, and coarsens the low-resolution fields by the configured scale factor
4. applies user defined transforms like unit conversions or log transformations
5. splits into a train and test dataset and standardizes both datasets based on the mean and standard deviation of all grids from the training data only (also writes this information into the zarr metadata for inference)
6. writes to `.zarr`
7. `nc2pt/tools/zarr_to_torch.py` - writes to PyTorch files
8. `nc2pt/tools/single_file_to_batches.py` - batches the single PyTorch files

## What are the downsides of using PyTorch files for climate data?
The most obvious downside is that you lose the metadata associated with a netCDF dataset. The intermediate Zarr format produced by nc2pt allows for parallelized io and perserves the metadata. This is useful for inference. 


## Requirements
* [Miniconda](https://docs.conda.io/en/latest/miniconda.html) with Python >= 3.8
* [xESMF](https://xesmf.readthedocs.io/en/latest/)



### 💽 Installation
xESMF is only available through Conda, so you will have to be able to install conda on your system. Unfortunately, this is limiting because certain HPCs don't allow conda.

1. Begin by install xESMF here in a conda environment: [xESMF](https://xesmf.readthedocs.io/en/latest/)

2. Clone this repository

3. Install into your conda environment

```bash
conda install -c conda-forge pip
pip install -r requirements.txt
# editable install
pip install -e nc2pt/
```

That's it!

### 📋 Configuration
nc2pt uses [hydra](https://hydra.cc/) for configuring and by instantiating structured classes in `nc2pt/climatedata.py`. This simeultaneously defines the workflow as well as the data. Please see `nc2pt/conf/config.yml` for an example configuration, or below:

```yaml
_target_: nc2pt.climatedata.ClimateData # Iniatlizes ClimateData dataclass object
output_path: /home/nannau/data/proc/
climate_models:
  # This lists the models 
  - _target_: nc2pt.climatedata.ClimateModel
    name: hr
    info: "High Resolution USask WRF, Western Canada"
    climate_variables: # Provides a list of ClimateVariable dataclass objects to initialize
      - _target_: nc2pt.climatedata.ClimateVariable
        name: "pr"
        alternative_names: ["PREC"] # any alternative names will be renamed to name 
        path: /PREC/*.nc # wildcards are supported for xr.open_mfdataset
        is_west_negative: true # if false, add 360 deg to longitude data
        invariant: false # whether it is an invariant field or has time
        transform: null # list of custom transforms here

      - _target_: nc2pt.climatedata.ClimateVariable
        name: "uas"
        alternative_names: ["U10", "u10", "uas"]
        path: /U10/*.nc
        is_west_negative: true
        invariant: false
        transform: null

  - _target_: nc2pt.climatedata.ClimateModel
    info: "Low resolution ERA5, Western Canada"
    name: lr
    hr_ref: # Reference field to interpolate to. Will need to provide new file if not using USask WRF
      _target_: nc2pt.climatedata.ClimateVariable
      name: "hr_ref"
      alternative_names: ["T2"]
      path: nc2pt/nc2pt/data/hr_ref.nc
      is_west_negative: true

    climate_variables:
      - _target_: nc2pt.climatedata.ClimateVariable
        name: "uas"
        alternative_names: ["U10", "u10", "uas"]
        path: /proc/uas_1hr_ERA5_an_RDA-025_1979010100-2018123123_time_sliced_cropped.nc
        is_west_negative: false
        invariant: false
        transform: null

      - _target_: nc2pt.climatedata.ClimateVariable
        name: "pr"
        alternative_names: ["PREC"]
        path: /proc/pr_1hr_ERA5_fc_RDA-025_1979010106-2019010106_time_sliced_cropped_merged_forecast_time.nc
        is_west_negative: false
        invariant: false
        transform: # custom transformations! handy for unit conversions, etc.
          - "x * 3600"
          - "np.log10(x + 1)"

      - _target_: nc2pt.climatedata.ClimateVariable
        name: "Q2"
        alternative_names: ["hurs", "huss", "Q2", "q2"]
        path: /proc/huss_1hr_ERA5_an_RDA-025_1979010100-2018123123_time_sliced_cropped.nc
        is_west_negative: false
        invariant: false
        transform: null


dims: # Defines the dimensions you might find in your lr or hr dataset and lists them to be initialized as ClimateDimension objects. Typically this would match what is in your hr dataset. Intended to allow for renaming of dimensions and allows for the control of chunking
  - _target_: nc2pt.climatedata.ClimateDimension
    name: time
    alternative_names: ["forecast_initial_time", "Time", "Times", "times"]
    chunksize: 100
  - _target_: nc2pt.climatedata.ClimateDimension
    name: rlat
    alternative_names: ["rotated_latitude"]
    hr_only: true
    chunksize: -1
  - _target_: nc2pt.climatedata.ClimateDimension
    name: rlon
    alternative_names: ["rotated_longitude"]
    hr_only: true
    chunksize: -1

# similar to dims, just as coodinates instead. coordinates might not match dims on curvilinear grids
coords:
  - _target_: nc2pt.climatedata.ClimateDimension
    name: lat
    alternative_names: ["latitude", "Lat", "Latitude"]
    chunksize: -1
  - _target_: nc2pt.climatedata.ClimateDimension
    name: lon
    alternative_names: ["longitude", "Long", "Lon", "Longitude"]
    chunksize: -1

# subsample data temporally or spatially!
select:
  # Time indexing for subsets
  time:
    # Crop to the dataset with the shortest run
    # this defines the full dataset from which to subset
    range:
      start: "20001001T01:00:00"
      end: "20150928T12:00:00"

    # use this to select which years to reserve for testing
    # the remaining years in full will be used for training
    test_years: [2000, 2009, 2014]

  # IMPORTANT!
  # sets the scale factor and index slices of the hr dimensions. you will usually have to determine this ahead of time and check that (first_index - last_index)/8 == your lr dataset
  spatial:
    scale_factor: 8
    x:
      first_index: 110
      last_index: 622
    y:
      first_index: 20
      last_index: 532



# dask client parameters
compute:
  # xarray netcdf engine
  engine: h5netcdf
  dask_dashboard_address: 8787

# for tools scripts
# you can randomize or also combine pytorch files into pytorch batches with tools/single_files_to_batches.py 
loader:
  batch_size: 2
  randomize: true
  seed: 0


```

### 🚀 Running
1. Explore data and ensure compatibility
2. Configure `nc2pt/conf/config.yaml`
3. Run the `nc2pt/preprocess.py` script which will run through your preprocessing steps. This creates the zarr files
4. Run the `nc2pt/tools/zarr_to_torch.py` script which serializes each time step in the `.zarr` file to an individual PyTorch `.pt` file.
5. Optional: run the `nc2pt/tools/single_files_to_batches.py` which combines individual files from the previous step into random batches. This setup allows for less io in your machine learning pipeline.



---
