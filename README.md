# ProbAsn
Probabilistic assignment of organic crystals

This is the code for the paper "Bayesian Probabilistic Assignment of Chemical Shifts in Organic Solids", doi: [...].

# Installation

## Python

You first need to install Python and the libraries required to run the software:

- Python 3 (3.8)
- numpy (version 1.20.3)
- ase (version 3.19.0)
- openbabel (version 3.1.0)
- networkx (version 2.5)

Alternatively, a Python environment containing all the required libraries can be installed by running the "install_env.sh" script.

## Database

The database is available from: https://drive.google.com/drive/folders/1vqzamV5kXTM8ggIeD3HSXQ7efdqJjY0a?usp=sharing

Place the downloaded directories in the "db" directory.

Then, you should be good to go!

# Running the software

To perform the assignment of a molecule, prepare an input file as described in "input_file_description.md" or modify an existing input file that you can find in the "example" directory. Then, run the assignment by running:

> python run.py input_file.in

Alternatively, you can run the "run.ipynb" notebook and modify the input file in cell 2.

