#!/bin/bash

# Install the python environment using Conda

conda create -n ProbAsn python=3.10 -y

source activate ProbAsn

pip install .

conda install -y ipykernel

python -m ipykernel install --user --name ProbAsn --display-name "ProbAsn"
