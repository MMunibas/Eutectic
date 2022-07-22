#!/usr/bin/python

# Basics
import os
import subprocess
import numpy as np

# Miscellaneous
import time
from glob import glob
import getpass

runfile = "%RFILE%"
inpfile = "%IFILE%"
outfile = "%OFILE%"


def check_status():
    
    with open(inpfile, 'r') as f:
        
        # Read lines
        inplines = f.read()
        
    # Check heat output
    if os.path.exists("heat.dcd"):
        inplines = inplines.replace("set mini 0", "set mini 1")
    
    # Check equi output
    if os.path.exists("equi.dcd"):
        inplines = inplines.replace("set heat 0", "set heat 1")
    
    # Check dcd output
    if os.path.exists("dyna.0.dcd"):
        inplines = inplines.replace("set equi 0", "set equi 1")
            
    # Write modified input
    with open(inpfile, 'w') as f:
        f.write(inplines)
        

# Start/Continue simulation
check_status()
task = subprocess.run(['sbatch', runfile], capture_output=True)
tskid = int(task.stdout.split()[-1])

finished = False
error = False
latest_step = 0
while not finished:
    
    # Wait a minute
    time.sleep(60)
    
    # Check slurm file for error message
    slurmfile = 'slurm-{:d}.out'.format(tskid)
    if os.path.exists(slurmfile):
        
        with open(slurmfile, 'r') as f:
            
            # Read lines
            lines = f.readlines()
            
            # Look for "core dumped" error
            for line in lines:
                if ("forrtl: severe (174): SIGSEGV, segmentation fault occurred" 
                    in line):
                    error = True
    else:
        
        # Job has not started yet
        continue
        
    # If eror occurs, adopt current step and rerun simulation.
    # Else, check for completeness
    if error:
        
        # Cancel job
        subprocess.run(['scancel', '{:d}'.format(tskid)])
        
        # Check current step
        dcdfiles = glob('dyna.*.dcd')
        idcd = np.array(
            [dcdfile.split('.')[1] for dcdfile in dcdfiles], dtype=int)
        if len(idcd):
            current_step = np.max(idcd)
        else:
            current_step = 0
        
        # Update current step in input file
        with open(inpfile, 'r') as f:
            
            # Read lines
            inplines = f.read()
        
        for line in inplines.split('\n'):
            if 'set n ' in line:
                defline = line
                break
        
        inplines = inplines.replace(defline, 'set n {:d}'.format(
            current_step))
        
        # Also check if progress is made, if not change frequency for updating
        # neighbor list
        if current_step == latest_step:
            
            for line in inplines.split('\n'):
                if 'ntrfrq' in line:
                    for it, tag in enumerate(line.split()):
                        if tag=='ntrfrq':
                            nfrq = int(line.split()[it + 1])
                    nfrqtag = 'ntrfrq {:d}'.format(nfrq)
                    if nfrq - 1 <= 100:
                        nfrq = 201
                    new_nfrqtag = 'ntrfrq {:d}'.format(nfrq - 1)
                    inplines = inplines.replace(nfrqtag, new_nfrqtag)
        
        # Update latest step
        latest_step = current_step
        
        # Write modified input
        with open(inpfile, 'w') as f:
            f.write(inplines)
        
        # Restart simulation
        check_status()
        task = subprocess.run(['sbatch', runfile], capture_output=True)
        tskid = int(task.stdout.split()[-1])
        
        # Reset error flag
        error = False
            
    else:
        
        # Get task id list
        user = getpass.getuser()
        tsklist = subprocess.run(
            ['squeue', '-u', user], capture_output=True)
        idslist = [
            int(ids.split()[0])
            for ids in tsklist.stdout.decode().split('\n')[1:-1]]
        
        # If task is still running (and no error in slurm) ...
        if tskid in idslist:

            continue
        
        # Else, check if job terminated succesfully 
        # (I expect a large output file and dont want to always check it)
        with open(outfile, 'r') as f:
            
            # Read lines
            outlines = f.readlines()
            
        for line in outlines:
            if 'NORMAL TERMINATION BY NORMAL STOP' in line:
                finished = True
                
        
                    
        
        
