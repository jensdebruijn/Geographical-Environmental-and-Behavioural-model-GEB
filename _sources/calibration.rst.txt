Calibration
#####################

GEB has a prebuilt calibration function. It can be used for both one- and multi-objective optimization. To run the calibration follow the steps below. Use the command below, setting the `-wd` command to the folder that includes your `model.yml` file. See also the `.sh` script for Linux cluster runs below.

1. Determine which objectives/factors you want to calibrate. Add a function which calculates the score to `calibrate.py` and conditionally add the function to the code in `calibrate.py` below. Lastly, activate the function under `calibration_targets` in the `model.yml` file. For the rest of the steps, we will assume calibration on discharge data.

   ::
   
       for score in config['calibration']['calibration_targets']:
           if score == 'KGE_discharge':
               scores.append(get_KGE_discharge(run_directory, individual, config, gauges, observed_streamflow))

2. Determine the location of the gauge(s) for which you have observed discharge data. Make sure the coordinates match with the location of the river in the model. Do this by checking the upstream_area file of your model (input/routing/kinematic/upstream_area) and checking the location of the gauge(s). The location of the gauge should be in the river. The observed discharge data should be stored as a `.csv` file where the file name is the coordinates without a comma separation (in this example: "75.8477777 17.4130555.csv"). The first column should be "date" and the second "flow", i.e.:

   ::
   
       date, flow
       2001-07-21,184.8000031

   Make sure to check which symbol is used as the seperator in the csv. file. In the calibration script, the comma (,) is assumed to be the seperator in the .csv file. 

3. Store this file in the location below:

   ::
   
       original_data/calibration/streamflow/75.8477777 17.4130555.csv
   
   If this folder does not yet exist, create the folder manually and (if needed) change the path in the model.yml file.

4. Next, add this location under `gauges` in the `model.yml` file and under `target_variables` and `report_hydrology`. See the example of the `model.yml` file below.

5. For multiple observed locations of observed data the process is similar, just repeat step 2 to 4.

6. Add the parameters you want calibrated under `parameters` in the `model.yml` file. First set a min and max. After, set the variable name, which refers to the location in the overall `model.yml` file, e.g.:

   ::
   
       expenditure_cap:
           variable: agent_settings.expected_utility.decisions.expenditure_cap
           min: 0.05
           max: 0.4

   would setup calibration for x between 0.05 and 0.4 for the following variable:

   ::
   
       agent_settings:
           expected_utility:
               decisions:
                   expenditure cap: x

7. Next, set the DEAP parameters. `ngen` specifies the number of total generations, `mu` the initial population size, `lambda` the number of children to produce at each generation, and `select_best` controls the number of best individuals selected from the population for the next generation.

8. Lastly, run the model with the command below, or use a bash script to run it on the cluster.

   .. code-block:: bash

       geb calibrate -wd location/of/your_model.yml/folder

   # An example of a Linux bash script calling the calibration. Note: the cpus-per-task command sets the number of parallel runs if multiprocessing is enabled.

   ::
   
       #!/bin/bash
       #SBATCH --job-name=
       #SBATCH --output=logs/calibrate-%j.out
       #SBATCH --ntasks=1
       #SBATCH --nodes=1
       #SBATCH --ntasks-per-node=1
       #SBATCH --cpus-per-task=30
       #SBATCH --mem=120G
       #SBATCH --time=600:00:00
       #SBATCH --mail-type=END,FAIL
       #SBATCH --mail-user=
       
       echo $1
       
       source ~/.bashrc
       
       SCRIPT_DIR="$HOME/GEB/GEB_models/"
       cd $SCRIPT_DIR
       
       conda activate geeb  # activate conda environment
       
       cd models/
       geb calibrate -wd $1/base
       
       echo "done"

   # Running the bash script with your study area of choice (here the Bhima Basin): 

   ::
   
       sbatch "path_to/script/bash.sh" bhima

   # Example `model.yml` with settings for the calibration 

   ::
   
       gauges:
           - [75.8477777, 17.41305556]
       calibration:
         pre_spinup_time: 1980-01-01
         spinup_time: 1980-01-01
         start_time: 2001-01-01
         end_time: 2011-12-31
         path: calibration_multi_5
         gpus: 0
         scenario: adaptation
         monthly: false
         calibration_targets:
           KGE_discharge: 1
         DEAP:
           use_multiprocessing: true
           ngen: 10
           mu: 60
           lambda_: 25
           select_best: 10
         target_variables:
           # Variables required to calculate calibration score from cwatm, e.g. discharge at a certain gauge 
           report_hydrology:
               75.8477777 17.41305556:
                   varname: data.grid.discharge
                   function: sample_coord,75.8477777,17.41305556
                   format: csv
                   save: save
           # Variables required to calculate calibration from GEB, e.g. yield ratio 
           report:
               yield_ratio:
                   type: farmers
                   function: mean
                   varname: yearly_yield_ratio[:,1]
                   save: save
                   format: csv 
                   frequency:
                     every: month
                     day: 1
           # The to be calibrated parameters 
           parameters:
               soildepth_factor:
                   variable: parameters.soildepth_factor
                   min: 0.8
                   max: 1.8
               preferentialFlowConstant:
                   variable: parameters.preferentialFlowConstant
                   min: 0.5
                   max: 8
               arnoBeta_add:
                   variable: parameters.arnoBeta_add
                   min: 0.01
                   max: 1.0
               factor_interflow:
                   variable: parameters.factor_interflow
                   min: 0.33
                   max: 3.0
               recessionCoeff_factor:
                   variable: parameters.recessionCoeff_factor
                   min: 0.05
                   max: 10
               manningsN:
                   variable: parameters.manningsN
                   min: 0.1
                   max: 10.0
               lakeAFactor:
                   variable: parameters.lakeAFactor
                   min: 0.333
                   max: 5.0
               lakeEvaFactor:
                   variable: parameters.lakeEvaFactor
                   min: 0.8
                   max: 3.0
               max_reservoir_release_factor:
                   variable: agent_settings.reservoir_operators.max_reservoir_release_factor
                   min: 0.01
                   max: 0.05
