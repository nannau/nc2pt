FROM jupyter/base-notebook:python-3.10.4

USER root

# Create a working directory
WORKDIR /workspace/
# VOLUME [/usr/src/app/

# WORKDIR /usr/src/app

# Build with some basic utilities
RUN apt-get update && apt-get install -y \
    python3-pip \
    git 

# alias python='python3'
# RUN ln -s /usr/bin/python3 /usr/bin/python
# RUN sudo apt-get -y install cdo
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install cdo
RUN sudo apt-get -y install libgeos-dev

RUN pip install cartopy

COPY ./requirements.txt /workspace/requirements.txt
RUN pip install -r /workspace/requirements.txt

# # Required for password management and customization
# RUN jupyter notebook --generate-config
# RUN echo "LOOK HERE ${JUPYTER_CONFIG_DIR}"
# COPY ./jupyter_notebook_config.py ${JUPYTER_CONFIG_DIR}jupyter_notebook_config.py

# Add Tini. Tini operates as a process subreaper for jupyter. This prevents
# kernel crashes.
# ENV TINI_VERSION v0.6.0
# ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
# RUN chmod +x /usr/bin/tini
# ENTRYPOINT ["/usr/bin/tini", "--"]

# EXPOSE 8888
# CMD ["/bin/bash"]
# CMD ["jupyter", "lab", "--port=8888", "--no-browser", "--ip=0.0.0.0"]