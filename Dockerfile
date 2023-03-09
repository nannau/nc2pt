FROM continuumio/miniconda3

USER root

COPY . ./

RUN conda env update --file environment.yml --name base
RUN conda install pip
RUN pip install -e .
