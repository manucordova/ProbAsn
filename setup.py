import os
import setuptools

VERSION = "1.0.0"
DESCRIPTION = "ProbAsn"

LONG_DESCRIPTION = "Bayesian probabilistic assignment of chemical shifts in organic solids"

os.system("conda install openbabel -c conda-forge -y")

setuptools.setup(
    name = "ProbAsn",
    version = VERSION,
    author = "Manuel Cordova",
    author_email = "manuel.cordova@epfl.ch",
    description = DESCRIPTION,
    long_description = LONG_DESCRIPTION,
    url = "TBA",
    include_package_data = True,
    package_dir = {"": "src"},
)

