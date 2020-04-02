import json
import logging
import os
import random 
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
import pandas as pd

import strax
import straxen

#SBATCH --job-name=scanner_{name}_{config}                                                             
# Previous issue was that the job-name was too long. 
# There's some quota on how many characters or something a job-name can be.
# mem-per-cpu argument doesn't work on dali not sure why. Too many computers requested I believe.
JOB_HEADER = """#!/bin/bash
#SBATCH --job-name=scan_{name}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={n_cpu}
#SBATCH --time={max_hours}:00:00
#SBATCH --partition={partition}
#SBATCH --account=pi-lgrandi
#SBATCH --qos={partition}
#SBATCH --output={log_fn}
#SBATCH --error={log_fn}
{extra_header}
# Conda
. "{conda_dir}/etc/profile.d/conda.sh"
{conda_dir}/bin/conda activate {env_name}
echo Starting scanner
python {python_file} {run_id} {data_path} {config_file} 
"""


def scan_parameters(strax_options, directory,
                    **kwargs):
    """Called in mystuff.py to run specified jobs. 
    
    This main function goes through and runs jobs based on the options given in strax.
    Currently this is constructed to call submit_setting, which then eventually starts a job and proceeds to get a dataframe of event_info, thereby processing all data for specified runs.
    Params: 
    strax_options: a dictionary which must include:
        - 'run_id' (default is '180215_1029')
        - 'config' (strax configuration options)
        - A directory to save the job logs in. Default is current directory in a new file.
        - kwargs: max_hours, extra_header, n_cpu, ram (typical dali job sumission arguments apply.)
    """
    
    # Check that directory exists, else make it
    #if not os.path.exists(directory):
    #    os.makedirs(directory)

    # Submit each of these, such that it calls me (parameter_scan.py) with submit_setting option
    for i, strax_option in enumerate(strax_options):
        print('Submitting %d with %s' % (i, strax_option))
        submit_setting(**strax_option, directory=directory)

    pass

def submit_setting(run_id, config, plugin, directory, **kwargs):
    """
    Submits a job with a certain setting.
    
    Given input setting arguments, submit_setting will take a current run and submit a job to dali that calles this code itself (so not runs scanner.py directly.) First, files are created temporarily in directory (arg) that document what you have submitted. These are then used to submit a typical job to dali. This then bumps us down to if __name__ == '__main__' since we call it directly. You could alternatively switch this so that if we submit_setting we run a different file. 
    
    Arguments:
    run_id (str): run id of the run to submit the job for. Currently not tested if multiple run id's work but they should
    config (dict): a dictionary of configuration settings to try (see strax_options in mystuff.py for construction of these config settings)
    directory (str): Place where the job logs will get saved. These are short files just with the parameters saved and log of what happened, but can be useful in debugging.
    **kwargs (optional): can include the following
        - n_cpu : numbers of cpu for this job. Default 40. Decreasing this to minimum needed will get your job to submit to dali faster!
        - max_hours: max hours to run job. Default 8
        - name: the appendix you want to scan_{name} which shows up in dali. Defaults "magician" if not specified so change if you're not a wizard.
        - mem_per_cpu: (MB) default is 4480 MB for RAM per CPU. May need more to run big processing.
        - partition: Default to dali
        - conda_dir: Where is your environment stored? Default is /dali/lgrandi/strax/miniconda3 (for backup strax env). Change?
        - env_name: Which conda env do you want, defaults to strax inside conda_dir
    """
    
    job_fn    = tempfile.NamedTemporaryFile(delete=False, dir=directory).name
    log_fn    = tempfile.NamedTemporaryFile(delete=False, dir=directory).name
    config_fn = tempfile.NamedTemporaryFile(delete=False, dir=directory).name
    
    #Takes configuration parameters and dumps the stringed version into a file called config_fn
    with open(config_fn, mode='w') as f:
        json.dump(config, f)

    # TODO: move these default settings out to above somehow.  Maybe a default dictionary
    # that gets overloaded?
    with open(job_fn, mode='w') as f:
        # Rename such that not just calling header, I think this is done now, no?
        # TODO PEP8
        f.write(JOB_HEADER.format(  
            log_fn=log_fn,
            config=str(config),
            python_file=os.path.abspath(__file__),
            config_file = config_fn,
            name = kwargs.get('job_name',
                              'gvolta'),
            n_cpu=kwargs.get('n_cpu',
                             40),            
            max_hours=kwargs.get('max_hours',
                                 8),
            mem_per_cpu=kwargs.get('mem-per-cpu',
                                   4480),
            partition=kwargs.get('partition',
                                    'dali'),
            conda_dir=kwargs.get('conda_dir',
                                 '/dali/lgrandi/strax/miniconda3'),
            env_name=kwargs.get('env_name', 'strax'),
            run_id=kwargs.get('run_id',
                               run_id),
            data_path=kwargs.get('data_path', 
                                 directory), 
            extra_header=kwargs.get('extra_header',
                                    ''),
        ))
    print(sys.argv)

    print("\tSubmitting sbatch %s" % job_fn)
    result = subprocess.check_output(['sbatch', job_fn])

    print("\tsbatch returned: %s" % result)
    job_id = int(result.decode().split()[-1])

    print("\tYou have job id %d" % job_id)

def work(run_id, plugin, data_path, config, **kwargs):
    st = straxen.contexts.strax_SPE()
    print(run_id, plugin, data_path, config)
    #st.storage[-1] = strax.DataDirectory(data_path, provide_run_metadata=False)
    
    path_to_df = 'scanner_dfs/'
    df = st.get_array(run_id, plugin, config=config, max_workers = kwargs.get('n_cpu', 40))

if __name__ == "__main__": #happens if submit_setting() is called
    if len(sys.argv) == 1: # argv[0] is the filename
        print('hi I am ', __file__)
        scan_parameters()
    elif len(sys.argv) == 4:
        run_id = sys.argv[1]
        data_path = sys.argv[2]
        config_fn = sys.argv[3]
        print(run_id, data_path, config_fn)
        print("Things are changing")
        # Reread the config file to grab the config parameters
        with open(config_fn, mode='r') as f:
            config = json.load(f)
        
        work(run_id=run_id, data_path=data_path, config=config, max_workers=40)
    else:
        raise ValueError("Bad command line arguments")