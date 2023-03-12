# flytrackerJAABA_analysis
Scripts for working with Flytracker, JAABA, and their output data

_____________________________

flytrackerJAABA_parse_plot.py 
_____________________________
The main script in this repo. This is a python module that is able to extract and parse data from the Flytracker JAABA output files and the JAABA score output files.
The module can also be used to plot tracks, plot behavior scores, plot ethodgrams, and plot interaction networks.
Documentation will arrive soon.


_____________________________

flytracker_manual_run.m 
_____________________________
Script to run flytracker in parallel through the command line. The videos must be calibrated first, and a calibration file must be in the video directory.


_____________________________

flytracker_job.sh 
_____________________________
Job script to run flytracker_manual_run.m on Temple University Compute Server using `qsub`.

