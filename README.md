# flytrackerJAABA_analysis
Scripts for working with Flytracker, JAABA, and their output data

_____________________________

fly2py.py 

The main script in this repo. This is a python module that is able to extract and parse data from the Flytracker JAABA output files and the JAABA score output files.
The module can also be used to plot tracks, plot behavior scores, plot ethodgrams, and plot interaction networks.
Documentation will arrive soon.

To set up the conda environment and install all necessary dependancies, use the FLY2PY.yml file in a terminal as follows:

```
conda env create -f FLY2PY.yml
```

Then, activate the environment in your terminal or your IDE:

```
conda activate FLY2PY
```

Now you can use fly2py!

The fly2py_demo directory contains an example script, ftjp_demo.py, that demonstrates the use of many of functions of fly2py

_____________________________

flytracker_manual_run.m 

Script to run flytracker in parallel through the command line. The videos must be calibrated first, and a calibration file must be in the video directory.


_____________________________

flytracker_job.sh 

Job script to run flytracker_manual_run.m on Temple University Compute Server using `qsub`.

_____________________________