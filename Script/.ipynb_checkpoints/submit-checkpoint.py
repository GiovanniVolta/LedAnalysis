#!/usr/bin/env python
from __future__ import print_function    # (at top of module)

import argparse
import tempfile
import time
import os.path as osp
import shutil
import csv
from datetime import datetime

import random
import string
import os
import subprocess

class LacyDB():
    def __init__(self, p = None):
        self.fieldnames = ['job_start', 'job_id', 'job_name', 'job_batch', 'job_log']
        self.select = []
        if p != None and os.path.isfile(p):
            self.filepath = p
            self.path = os.path.split(self.filepath)[0]
        else:
            self.filepath = os.path.join(os.getenv("HOME"), ".tmp_job_submission", "jobDB.csv")
            self.path = os.path.split(self.filepath)[0]
        
        #Just in case the path does not exists yet:
        if not os.path.exists(self.path):
            os.makedirs(self.path)
            
        #If the path/file does not exists:
        if not os.path.isfile(self.filepath):
            self.CreateEmpty()
            
    def CreateEmpty(self):
        ###Create an emtpy file for storage"""
        with open(self.filepath, mode='w') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=self.fieldnames)
            writer.writeheader()
        
        
    def WriteEntry(self, entry):
        #get line inputs right:
        line_obj = []
        for key in self.fieldnames:
            
            val = entry.get(key)
            if isinstance(val, datetime):
                val = val.strftime("%m/%d/%Y-%H:%M:%S")
            if isinstance(val, int):
                val = str(val)
            line_obj.append(val)
            
        with open(self.filepath, mode='a') as csv_file:
            writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(line_obj)
        
    def ReadEntry(self, job_name=None, job_id=None):
        
        if isinstance(job_name, str):
            job_name = [job_name]
        if isinstance(job_id, str) or isinstance(job_id, int):
            job_id = [int(job_id)]
        
        with open(self.filepath, mode='r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            self.select = []
            for i_row in csv_reader:
                if job_name != None and i_row.get('job_name') in job_name:
                    self.select.append(i_row)
                if job_id != None and int(i_row.get('job_id')) in job_id:
                    self.select.append(i_row)
    
    def ReadAll(self):
        
        with open(self.filepath, mode='r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            self.select = []
            for i_row in csv_reader:
                self.select.append(i_row)

    def Get(self):
        return self.select
    
    def Show(self):
        k_sorted = sorted(self.select, key=lambda item:(datetime.strptime(item['job_start'], '%m/%d/%Y-%H:%M:%S')), reverse=True)
        
        print("Your batch submission history:")
        print("------------------------------")
        for ielement in k_sorted:
            job_batch_size =  os.stat(ielement['job_batch'])
            job_log_size =  os.stat(ielement['job_log'])
            
            print("  <> Name        : {0}".format(ielement['job_name']))
            print("  <> ID          : {0}".format(ielement['job_id']))
            print("  <> Start       : {0}".format(ielement['job_start']))
            print("  <> Batch script: {0} ({1} bytes)".format(ielement['job_batch'], job_batch_size.st_size))
            print("  <> Batch output: {0} ({1} bytes)".format(ielement['job_log'], job_log_size.st_size))
            print("     ------------")
        print("--Hint:")
        print(" - Size in bytes for batch output indicates if job has started already")
        

def submission_script(gpu=False):
    basic_header = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={log_fn}
#SBATCH --error={log_fn}
#SBATCH --account=pi-lgrandi
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={n_cpu}
#SBATCH --mem-per-cpu={mem_per_cpu}
#SBATCH --time={max_hours}
{extra_header}

echo Starting batch job

"""

    gpu_header = """\
#SBATCH --partition=gpu2
#SBATCH --gres=gpu:1

module load cuda/9.1
"""

    cpu_header = """\
#SBATCH --qos dali
#SBATCH --partition dali
"""
    if gpu==False:
        b = basic_header.replace("{extra_header}", cpu_header)
    else:
        b = basic_header.replace("{extra_header}", gpu_header)
    
    return b
    
def make_executable(path):
    """Make the file at path executable, see """
    mode = os.stat(path).st_mode
    mode |= (mode & 0o444) >> 2    # copy R bits to X
    os.chmod(path, mode)
    

def get_ongoing_jobs(username):
    q = subprocess.check_output(['squeue', '-u', username])
    job_id = []
    for line in q.decode().splitlines():
        if line.split()[0] != 'JOBID':
            job_id.append(int(line.split()[0]))
    return job_id

def check_job_exists(username, jobname):
    q = subprocess.check_output(['squeue', '-u', username])
    job_id = -1
    for line in q.decode().splitlines():
        if jobname in line:
            job_id = int(line.split()[0])
            print("You still have a running job: ID {0}".format(job_id))
            break
    return job_id
    
def submit_job(job_fn_name):
    print("job_fn_name:", job_fn_name)
    result = subprocess.check_output(['sbatch', job_fn_name])
    job_id = int(result.decode().split()[-1])
    
    return job_id
    
def main():
    parser = argparse.ArgumentParser(
        description='Submit a xenon containerd job to Midway/Dali queue')
    
    parser.add_argument('--timeout', 
        default=120, type=int,
        help='Seconds to wait to start')
    
    parser.add_argument('--cpu', 
        default=4, type=int, 
        help='Number of CPUs to request.')
    
    parser.add_argument('--cpu-memmory', dest='cpu_memmory', 
        default=4480, type=int, 
        help='Memory request job')
    
    parser.add_argument('--cpu-time', dest='cpu_time', 
        default="02:00:00", type=str, 
        help='Computing time request. Default 2 hours. State with: HH:MM:SS. GPU always 2 hours')
    
    parser.add_argument('--gpu', 
        action='store_true', default=False,
        help='Request to run on a GPU partition. Limits runtime to 2 hours.')
    
    parser.add_argument('--job',
        default="BtJ-0", type=str,
        help="Specify your batch job name")
    
    parser.add_argument('--command', dest='command',
        default="", type=str,
        help="Your executable script")
    
    parser.add_argument('--clean', 
        action='store_true', default=False,
        help='Clean your submission folder: batch and log files only.')
    
    parser.add_argument('--find-id', dest='find_id', 
        default=None, type=str,
        help='Find jobs by job id.')
    
    parser.add_argument('--find-job', dest='find_job', 
        default=None, type=str,
        help='Find jobs by job name (can result multiple jobs)')
    
    parser.add_argument('--list', dest='list', 
        default=None, type=str,
        help='List all jobs: "all" or "ongoing"')
    
    parser.add_argument('--container', 
        default='osgvo-xenon:latest',
        help='Singularity container to load'
            'See wiki page https://xe1t-wiki.lngs.infn.it/doku.php?id=xenon:xenonnt:dsg:computing:environment_tracking'
            'Default container: "latest"')

    args = parser.parse_args()
    n_cpu = args.cpu
    n_mem = args.cpu_memmory
    n_tim = args.cpu_time
    n_job = args.job
    s_container = args.container
    pcommand = args.command



    #find where you are:
    homepath = os.getenv("HOME")
    username = os.getenv("USER")
    dirpath = os.getcwd()
    
    #Create a tmp_job_submission directory if not existing:
    tmp_dir = os.path.join(homepath, '.tmp_job_submission')
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    
    #Clean up your log and batch script files::
    if args.clean == True:
        db = LacyDB()
        
        print("Clean up: ", tmp_dir)    
        print("----------------------")
        files = os.listdir(tmp_dir)
        all_ids = get_ongoing_jobs(username)
        
        db.ReadEntry(job_id = all_ids)
        all_ongoing = db.Get()
        
        select = []
        for running in all_ongoing:
            select.append(running.get('job_batch'))
            select.append(running.get('job_log'))
        
        for f in files:
            if f.find("tmp") >= 0 and os.path.join(tmp_dir,f) not in select:
                os.remove(os.path.join(tmp_dir,f))
        exit()

    if (args.find_id != None or args.find_job != None) and args.list == None:
        if args.find_id != None:
            print("Look up jobs by ID {0}".format(args.find_id))
        if args.find_job != None:
            print("Look up jobs by name {0}".format(args.find_job)) 
        db = LacyDB()
        db.ReadEntry(args.find_job, args.find_id)
        db.Show()
        exit()

    if (args.find_id == None or args.find_job == None) and args.list != None:
        db = LacyDB()
        if args.list == 'all':
            db.ReadAll()
            db.Show()
        elif args.list == 'ongoing':
            all_ids = get_ongoing_jobs(username)
            db.ReadEntry(job_id = all_ids)
            db.Show()
        else:
            pass
        exit()
    
    

    #Specify:
    job_fn = tempfile.NamedTemporaryFile(delete=False, dir=tmp_dir)
    log_fn = tempfile.NamedTemporaryFile(delete=False, dir=tmp_dir)
    
    
    batch_job = submission_script(args.gpu).format(
                                            job_name=args.job,
                                            log_fn=log_fn.name,
                                            max_hours=2 if args.gpu else n_tim,
                                            n_cpu=n_cpu,
                                            mem_per_cpu=n_mem)
    
    batch_job += '/dali/lgrandi/xenonnt/submission/xnt_environment {s_container} {command}'.format(s_container=s_container,
                                                                                                             command=pcommand)    
    
    print('Your batch submission script:')
    print('-----------------------------')
    print(batch_job)
    print('-----------------------------')
    print()
    print()
    
    with open(job_fn.name, 'w') as f:
        f.write(batch_job)
    
    time.sleep(1)
    
    
    print("Start job...")
    print("------------")
    make_executable(job_fn.name)
    
    recent_job_id = submit_job(job_fn.name)
    
    print("Job name: ", args.job)
    print("Job id: ", recent_job_id)
    print("Log: ", log_fn.name)
    
    data = {
        'job_start': datetime.now(),
        'job_name': args.job,
        'job_id': recent_job_id,
        'job_batch': job_fn.name,
        'job_log': log_fn.name
        }
    #store stuff:
    db = LacyDB()
    db.WriteEntry(data)
    
    
    
if __name__ == "__main__":
    main()