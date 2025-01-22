##############
Preprocessing
##############

GEB uses HydroMT to preprocess all data required for the model. HydroMT (Hydro Model Tools) is an open-source Python package that facilitates the process of building and analyzing spatial geoscientific models with a focus on water system models. It does so by automating the workflow to go from raw data to a complete model instance which is ready to run. This plugin provides an implementation for the GEB model, which can be found `here <https://github.com/GEB-model/hydromt_geb>`_.

Obtaining the data
------------------
Most of the data that the HydroMT plugin uses to create the input data for GEB is downloaded by the tool itself when it is run. However, some data needs to be aquired seperately. To obtain this data, please send an email to Jens de Bruijn (jens.de.bruijn@vu.nl).

.. A basin id from the MERIT Hydro IHU dataset (https://zenodo.org/records/7936280), please refer to the file 30sec_basids.tif or 30sec_basins.gpkg.
.. An outflow point (longitude, latitude) of the river network from which the upstream subbasin can be derived automatically. Here, it is important that this point is located in the main river, which can be checked on the same MERIT Hydro IHU page, in the 30sec_uparea.tif file.

Building to model
-------------------
To set up the model you need two files, a `model.yml`-file and a `build.yml`-file. The `model.yml`-file specifies the configuration of the model, including start and end time of model runs, agent-paramters etc. The `build.yml`-file specifies the configuration of the preprocessing. An example of the `model.yml`-file and `build.yml`-file are provided in the repository.

The `build.yml`-file contains the name of functions that should be run to preprocess the data. The functions are defined in the HydroMT-geb plugin of HydroMT. You can build the model using the following command, assuming you are in the working directory of the model which contains the `model.yml`-file and `build.yml`-file:

.. code-block:: python

    geb build

This will preprocess all the data required for the model. The data will be stored in the "input" folder in the working directory. The data is stored in a format that is compatible with the GEB model. Optionally, you can specify the path to the `build.yml`-file using the `-b/--build-config` flag, and the path to the `model.yml`-file using the `-c/--config` flag. You can find more information about the flags by running:

.. code-block:: python

    geb build --help

Updating the model
-------------------

It is also possible to update an already existing model by running the following command.

.. code-block:: python

    geb update

This assumes you have a "update.yml"-file in the working directory. The `update.yml`-file contains the name of functions that should be run to update the data. The functions are defined in the "geb" plugin of HydroMT. The data will be updated in the "input" folder in the working directory. The data is stored in a format that is compatible with the GEB model.

For example to update the forcing data of the model, your "update.yml"-file could look like this:

.. code-block:: yaml

    setup_forcing:
        data_source: isimip
        starttime: 1979-01-01
        endtime: 2080-12-31
        resolution_arcsec: 1800
        forcing: gfdl-esm4
        ssp: ssp370

Optionally, you can specify the path to the "update.yml"-file using the `-b/--build-config` flag, and the path to the `model.yml`-file using the `-c/--config` flag. You can find more information about the flags by running:

.. code-block:: python

    geb update --help