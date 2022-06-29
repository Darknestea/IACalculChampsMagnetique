# ShadowTEM

This project tries to implement methods to partly automatize the configuration
of a transmission electron microscope and mimic its behavior using machine learning.


## Installation
Download this root folder, set up a python environment using `*TODO add poetry*`
then follow the **Usage** section.

## Usage
Before anything create a _my_id.py_ file containing `MY_ID = "your_id"`, this will serve to avoid any conflict between different runs

### Creating a new dataset from a configuration (generated or directly form the microscope)
- Choose an experiment name like `EXPERIMENT_TEST="test"` and place it in constants.py then set it as the active experiment _`ACTIVE_EXPERIMENT=EXPERIMENT_TEST`_
- Create a raw folder _Exp_test_ using `create_raw_experiment_folders` main
- Generate using _generate_configurations.py_ a _configurations.csv_ (or import from a _configuration.yaml_ file from a microscope run) and place it in _Raw/Exp_your_experiment_name_
- If you generated the file, perform all the configurations of the generated csv file on the microscope 
- In _constants.py_ set `SPECIFIC_REAL_BEAM_SLICES_FOLDER=None` and place all the images from the microscope as _n.png_, where _n_ stands for the id, in _Raw/Exp_your_experiment_name/RealBeamSlices_ folder **OR** set `SPECIFIC_REAL_BEAM_SLICES_FOLDER=your_folder_name` using the folder containing those images 
- Set the modifiable parameters in _Utils/constants.py_
- Run _main.py_ program to generate the cleaned and processed data in the _Cleaned/Exp_your_experiment_name_ folder

### Creating a dataset of perfect ellipses to compare ellipse parameter extraction methods
- Choose an experiment name like `EXPERIMENT_TEST="test"` and place it in constants.py then set it as the active experiment _`ACTIVE_EXPERIMENT=EXPERIMENT_TEST`_
- Create a raw folder _Exp_test_ using `create_raw_experiment_folders` main and set the type to `EXPERIMENT_FROM_GENERATED_ELLIPSES`
- In _constants.py_ set the `ELLIPSE_PARAMETER_EXTRACTION_METHODS` so that it contains the methods you want to compare, to train an ellipse detection model you can set this to `None`
- Use _generate_ellipses.py_ main function to generate the ellipses and perform the comparison (results will be in _Results/_)

### Training a model
**TODO**

### Notes
Everything done before 27/06/2022 for simulation is invalid because the acceleration current was wrong