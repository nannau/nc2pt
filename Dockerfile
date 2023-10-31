FROM continuumio/miniconda3

WORKDIR /app

# Create the environment:
COPY . .
RUN conda install -n base conda-libmamba-solver && conda config --set solver libmamba

RUN conda env create -f environment.yml
RUN pip install -e .

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "nc2pt_env", "/bin/bash", "-c", "activate", "nc2pt_env"]

# The code to run when container is started:
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "nc2pt_env", "python"]