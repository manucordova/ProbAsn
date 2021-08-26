#!/bin/bash

# Install the python environment using Conda

conda create -n ProbAsn python=3.8

conda install -n ProbAsn -c conda-forge numpy ase openbabel networkx ipykernel

conda activate ProbAsn

python -m ipykernel install --user --name ProbAsn --display-name "ProbAsn"
