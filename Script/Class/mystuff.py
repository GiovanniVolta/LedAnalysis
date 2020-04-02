import numpy as np
import configparser as cp
import scanner

### EDIT BELOW TO CHANGE CONFIG SETTINGS ###
Config = cp.ConfigParser()
Config.read('/home/gvolta/XENONnT/LedAnalysis/Script/Class/configuration.ini')

straxdata_rawrecords = Config.get('runs_pars','straxdata_rawrecords')

straxdata_spe   = Config.get('runs_pars','straxdata_spe')
run_spe_topbulk = Config.get('runs_pars','run_spe_topbulk')
run_spe_topring = Config.get('runs_pars','run_spe_topring')

straxdata_gain  = Config.get('runs_pars','straxdata_gain')
run_gain_step0  = Config.get('runs_pars', 'run_gain_step0')
run_gain_step1  = Config.get('runs_pars', 'run_gain_step1')
run_gain_step2  = Config.get('runs_pars', 'run_gain_step2')
run_gain_step3  = Config.get('runs_pars', 'run_gain_step3')
run_gain_step4  = Config.get('runs_pars', 'run_gain_step4')

light_level     = Config.get('runs_pars','light_level')

plugin = 'raw_records'
config = { }
       
run_ids = [run_gain_step3] # a 1-min background run 

strax_options = []

#Enumerate over all possible options to create a strax_options list for scanning later.
for run in run_ids:
    print('Setting %s:' % run)
    strax_options.append({'run_id' : run, 'plugin' : plugin, 'config' : config})

print(strax_options)

#scan over everything in strax_options
scanner.scan_parameters(strax_options, straxdata_rawrecords)