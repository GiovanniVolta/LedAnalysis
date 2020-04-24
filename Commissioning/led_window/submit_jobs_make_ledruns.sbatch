#!/bin/bash
#SBATCH --job-name=led_runs
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem-per-cpu=12000
#SBATCH --mail-type=ALL
#SBATCH --time=8:00:00
#SBATCH --partition=dali
#SBATCH --account=pi-lgrandi
#SBATCH --qos=dali
#SBATCH --output=/home/gvolta/XENONnT/LedAnalysis/Commissioning/led_window/strax_log/makerun_ledruns.log
#SBATCH --error=/home/gvolta/XENONnT/LedAnalysis/Commissioning/led_window/strax_log/makerun_ledruns_error.log
#SBATCH --mail-user=gvolta@physik.uzh.ch

echo "Activating conda environment"
source /dali/lgrandi/strax/miniconda3/bin/activate strax

echo SPE and Gain analysis
commissioning=/dali/lgrandi/giovo/XENONnT/commissioning
date=$(date +%Y%m%d)

mkdir -p $commissioning/Plot/$date/SPEfit
mkdir -p $commissioning/Plot/$date/AmpSpec
mkdir -p $commissioning/Data/$date

python make_ledruns.py


