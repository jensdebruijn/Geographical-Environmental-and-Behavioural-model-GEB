Installation
#############

GEB runs on Python 3.11+ and can be installed using pip. In the future, GEB will be available on PyPI, but for now, you can install it from GitHub:

We recommend to install GEB in a conda environment. If you don't have conda installed, you can install it from https://docs.conda.io/en/latest/miniconda.html. Then create a conda environment with Python 3.11 and activate it:

.. code-block:: bash

    conda config --add channels conda-forge
    conda create -n geb python=3.11
    conda activate geb

Then install the required packages:

.. code-block:: bash

    conda config --set solver libmamba
    conda install rasterio numba tbb pandas geopandas numpy deap  pyyaml xarray dask rioxarray pybind11 scipy netCDF4 flopy bmipy xmipy xlrd pyflow s3fs xesmf hydromt tqdm openpyxl xclim xesmf

Then install GEB from GitHub. You need to install the full version, which includes all dependencies:

.. code-block:: bash

    pip install geb[full]@git+https://github.com/GEB-model/GEB

Installation in development mode
--------------------------------

If you want to contribute to GEB, you can install it in development mode. This will install the package in editable mode, so that changes to the source code are immediately available. In that case, you need to clone the following repositories, and install in editable mode.

.. code-block:: bash

    git clone https://github.com/GEB-model/ABCWatM
    pip install -e ABCWatM

    git clone https://github.com/GEB-model/hydromt_geb
    pip install -e hydromt_geb

Then install GEB in editable mode:

.. code-block:: bash

    git clone https://github.com/GEB-model/GEB
    pip install -e GEB
