1) DOWNLOAD PACKAGES

So PlantFATE runs with the use of 2 other packages and with the coupling package you need to checkout 4 different github repositories. 

The packages are:

1) Coupling package: https://github.com/jensdebruijn/couple-plantFATE-CWatM (branch: develop)
2) PlantFATE package: https://github.com/jaideep777/Plant-FATE (branch: feature_stepByStepSim)
3) libpspm package: https://github.com/jaideep777/libpspm (branch: develop)
4) Phydro package: https://github.com/jaideep777/phydro (branch: master)

For ease of running and set up put these in one big project root folder (not necessary for the coupling package):

  root
  |---- phydro
  |     |--- inst/include
  |
  |---- libpspm
  |     |--- include
  |     |--- lib
  |
  |---- Plant-FATE
  |     |--- inst/include

2) INSTALL PLANTFATE and PYBIND

- module load cm-eigen3/3.3.7
- module load shared
- module load 2022
- module load GSL/2.7-GCC-11.3.0

As per instructions in PlantFATE readme, install natively in C++.

You may need to install additional libraries, specifically pybind: 

* pip

  * pip install pybind11

  * pip install cppimport
* libpspm

  * cd libpspm

  * make clean testclean

  * make

  * make check

* Phydro

  * cd phydro

  * make clean testclean

  * make check

* PlantFATE

  * cd Plant-FATE

  * make clean testclean
  
  * make python