#!/dali/lgrandi/strax/miniconda3/envs/strax/bin/python
echo "$(tput setaf 1)#### #### #### ####$(tput sgr 0)"
echo "$(tput setaf 1)$(tput setab 7)SPE acceptance$(tput sgr 0)"
echo "$(tput setaf 1)#### #### #### ####$(tput sgr 0)"

pwd=/dali/lgrandi/giovo/XENONnT/spe_acceptance
date=$(date +%Y%m%d)

echo "$(tput setaf 1)$(tput setab 7)Strax environment$(tput sgr 0)"
eval "$(/dali/lgrandi/strax/miniconda3/bin/conda shell.bash hook)"
conda activate strax

echo "$(tput setab 7)Make forlder to save data (yymmdd)$(tput sgr 0)"
mkdir -p $pwd/Plot/$date
mkdir -p $pwd/Data/$date


echo "$(tput setaf 1)$(tput setab 7)1. Windows Identification: $(tput sgr 0)"
python LEDwindows.py
echo "$(tput setaf 1)$(tput setab 7)2. SPE acceptance: $(tput sgr 0)"
python SPE.py

echo "$(tput setaf 1)$(tput setab 7)Removing data$(tput sgr 0)"
mkdir -p $pwd/tmp
rm -r $pwd/tmp

echo "$(tput setaf 1)$(tput setab 7)The output are in: /dali/lgrandi/giovo/XENONnT/spe_acceptance $(tput sgr 0)"
echo "$(tput setaf 1)#### #### #### ####$(tput sgr 0)"