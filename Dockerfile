FROM python-3.10-slim-buster

USER root

# Create a working directory
WORKDIR /workspace/

RUN apt-get update && apt-get install -y git 

RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install cdo
RUN apt-get -y install libgeos-dev
RUN pip install jupyterlab
RUN apt-get install -y gdal-bin
RUN apt-get install -y libgdal-dev


COPY ./requirements.txt /workspace/requirements.txt
RUN pip install -r /workspace/requirements.txt
