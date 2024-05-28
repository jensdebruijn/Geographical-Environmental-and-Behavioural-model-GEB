Configuration
#####################

The GEB model uses 2 configuration files in YAML-format:

* The first one is build the model (see example in "examples/sandbox/build.yml"). These are that hydroMT uses.
* The configuration file configures the model (see example in "examples/sandbox/model.yml"), which options are discussed below. Not that this yml-file also refers to the CWatM configuration file, which can be seperately configured if needed. To configure this file we refer to the `CWatM documentation <https://cwatm.iiasa.ac.at/>`_.

.. autoyaml:: examples/sandbox/model.yml