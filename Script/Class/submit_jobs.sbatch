#!/bin/bash
#SBATCH --job-name=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem-per-cpu=2000
#SBATCH --mail-type=ALL
#SBATCH --time=8:00:00
#SBATCH --partition=dali
#SBATCH --account=pi-lgrandi
#SBATCH --qos=dali
#SBATCH --output=/home/gvolta/XENONnT/LedAnalysis/Script/Class/strax_log/makerun.log
#SBATCH --error=/home/gvolta/XENONnT/LedAnalysis/Script/Class/strax_log/makerun_error.log
#SBATCH --mail-user=gvolta@physik.uzh.ch

xnt_env

python make_runs.py


