##############
Preprocessing
##############

GEB uses HydroMT to preprocess all data required for the model. HydroMT (Hydro Model Tools) is an open-source Python package that facilitates the process of building and analyzing spatial geoscientific models with a focus on water system models. It does so by automating the workflow to go from raw data to a complete model instance which is ready to run. This plugin provides an implementation for the GEB model, which can be found `here <https://github.com/jensdebruijn/hydromt_geb>`_.

Installing hydromt-geb
----------------------
Hydromt-geb should be installed with the GEB model, but if not it can be installed using pip:

.. code-block:: bash

    pip install git+https://github.com/jensdebruijn/hydromt_geb.git

Obtaining the data
------------------
Most of the data that the HydroMT plugin uses to create the input data for GEB is downloaded by the tool itself when it is run. However, some data needs to be aquired seperately. To obtain this data, please send an email to Jens de Bruijn (jens.de.bruijn@vu.nl).

Preprocessing
-------------
The preprocessing contains two steps:
1. Creating all non-agent data
2. Creating all data for agent attributes

The file build_model.py contains the code to build the model, including step 1 and 2. In this script the agent attributes are generated randomly. This allows you to quickly try to model. As input, build_model.py requires a configuration file, which specifies the pour point (pour_point) of the (sub-)basin in the "general" section of the yml-file. This file also contains all other parameters that are required to run the model. The configuration file is a YAML file. An example configuration file is provided in the repository (sandbox.yml). Here, you need to both specify the config (containing paths for the input data) and build configuration file. Then run the following command:

.. code-block:: python

    geb build --config model.yml --build build.yml

For more advanced uses, in particular when the distribution of farmers depend on the preprocessing of land use data for the model, you can also build the model, and later update the farmers. To do so you can use `geb update`. First create a csv-file with farmer characteristics contain the following columns: season_#1_crop, season_#2_crop, season_#3_crop, irrigation_source, household_size, daily_non_farm_income_family and daily_consumption_per_capita. The csv file "agents/farmers/farmers.csv" is best located in the preprocessing folder, which was created with "geb build". An example for creating such a csv file based on the land use and census data, is provided in "examples/sandbox/create_farmers.py". Now you need to provide another yml file which specifies the update data for hydromMT. For the farmers, an example is provided in "examples/sandbox/update_farmers.yml". Then run the following command:

.. code-block:: python

    geb update --config model.yml --build-update build_farmers.yml

`geb update` can also be used to update other parts of (already) preprocessed data by making another yml file with only the functions that should be updated and running `geb update` with that yml-file.