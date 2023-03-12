#!/bin/sh
#PBS -l walltime=36:00:00
#PBS -N job_name
#PBS -l nodes=1:ppn=10
#PBS -m bae
#PBS -M email@email.com



# change to directory where 'qsub' was called
cd $PBS_O_WORKDIR


module load matlab


# run my program
matlab -r flytracker_manual_run, exit 
