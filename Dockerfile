FROM continuumio/miniconda3

USER root

# Create a working directory
RUN conda create -n xesmf_env && conda activate xesmf_env
RUN conda install -c conda-forge xesmf
RUN conda install -c conda-forge dask distributed netCDF4 
RUN conda install -c conda-forge matplotlib cartopy jupyterlab
RUN conda install pip

COPY ./requirements.txt /workspace/requirements.txt
RUN pip install -r /workspace/requirements.txt
