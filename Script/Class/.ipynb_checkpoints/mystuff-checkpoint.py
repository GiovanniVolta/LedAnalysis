import numpy as np
import configparser as cp
import scanner

### EDIT BELOW TO CHANGE CONFIG SETTINGS ###
Config = cp.ConfigParser()
Config.read('/home/gvolta/XENONnT/LedAnalysis/Script/Class/configuration.ini')
        
run_led = Config.get('runs_pars','run_led')
run_noise = Config.get('runs_pars','run_noise')
run_ids = [run_led, run_noise] # a 1-min background run 

strax_options = []

#Enumerate over all possible options to create a strax_options list for scanning later.
for run in run_ids:
    print('Setting %s:' % run)
    config = {}
    strax_options.append({'run_id' : run, 'config' : config})

print(strax_options)

#scan over everything in strax_options
scanner.scan_parameters(strax_options, '/dali/lgrandi/giovo/XENONnT/strax_data')