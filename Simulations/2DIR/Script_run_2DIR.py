#!/usr/bin/python

# Basics
import os
import time
import shutil
import pickle
import subprocess
import numpy as np

# Miscellaneous
import time
from glob import glob

#---------------------------------
# Input Section
#---------------------------------

# Main directory
data_maindr = os.getcwd()

# Sample data directory
data_smpdir = '../results_samples'

# Sample data frequencies
data_frqtag = 'freqs_{:s}_{:s}_{:s}_{:s}_SCN.dat'

# Sample data positions
data_postag = 'snaps_{:s}_{:s}_{:s}_{:s}_SCN.xyz'

# Sample data unit cell
data_ucltag = 'ucell_{:s}_{:s}_{:s}_{:s}_SCN.dat'

# Data parameter order [separator, [temp, mix, sample number]]
data_sepinf = ['_', [1, 2, 3, 4]]

# Output file
data_outfle = 'out_{:s}_{:s}_{:s}_{:s}_SCN.txt'

# Template script directory
temp_scrdir = 'templates'

# 2DIR simulation script template
temp_script = 'navgCoupleSCN.c'

# 2DIR simulation script template parameter flags
temp_nprocs = '%NPRC%'
temp_ntstep = '%NTSP%'
temp_ntmdst = '%NTMD%'
temp_w0freq = '%WAVG%'

# C-Compiler program
comp_cmpprg = 'gcc'

# Executable name
comp_exctbl = 'CoupleSCN'

# Compiler flags
comp_cflags = ['-lm', '-fopenmp', '-O3']

# Script parameter
comp_nprocs = 16
comp_ntstep = 15
comp_ntmdst = 1250

# Endless loop
unfinished_buisness = True
data_params = {}
if os.path.exists('w0_constant.pkl'):
    with open('w0_constant.pkl', 'rb') as f:
        w0_constant = pickle.load(f)
else:
    w0_constant = {}
while unfinished_buisness:
    
    #---------------------------------
    # Preparation Section
    #---------------------------------
    
    # Get sample list
    data_frqlst = glob(
        os.path.join(data_smpdir, data_frqtag.format('*', '*', '*', '*')))
    
    # Get system parameter
    for frqfle in data_frqlst:
        
        # Extract information
        temp, mix, irun, isample = [
            frqfle.split('/')[-1].split(data_sepinf[0])[ii] 
            for ii in data_sepinf[1]]
        
        # Add to parameter list
        if not temp in data_params.keys():
            
            data_params[temp] = {}
            
        if not mix in data_params[temp].keys():
            
            data_params[temp][mix] = {}
            
        if not irun in data_params[temp][mix].keys():
            
            data_params[temp][mix][irun] = []

        if not isample in data_params[temp][mix][irun]:
            
            data_params[temp][mix][irun].append(isample)
    
    #---------------------------------
    # Average Section
    #---------------------------------
    
    # Get average frequency
    if os.path.exists('w0_constant.pkl'):
        with open('w0_constant.pkl', 'rb') as f:
            w0_constant = pickle.load(f)
    
    for frqfle in data_frqlst:
        
        # Extract information
        temp, mix, irun, isample = [
            frqfle.split('/')[-1].split(data_sepinf[0])[ii] 
            for ii in data_sepinf[1]]
        
        # Add to parameter list
        if not temp in w0_constant.keys():
            
            w0_constant[temp] = {}
        
        if not mix in w0_constant[temp].keys():
            
            w0_constant[temp][mix] = None
        
        # Skip if average frequency already given
        if w0_constant[temp][mix] is None:
           
            # Read reference frequencies
            with open(frqfle, 'r') as f:
                frqlns = f.readlines()
            refstm = []
            refsfr = []
            for lne in frqlns:
                refstm.append(float(lne.split()[0]))
                refsfr.append(np.array(lne.split()[1:], dtype=float))
            refstm = np.array(refstm, dtype=float)
            refsfr = np.array(refsfr, dtype=float).T
            
            # Compute average frequency
            w0_constant[temp][mix] = np.mean(refsfr)
       
    with open('w0_constant.pkl', 'wb') as f:
        pickle.dump(w0_constant, f)
    
    #---------------------------------
    # Iteration Section
    #---------------------------------
    
    # Iterate over system parameter
    for temp in data_params.keys():
        for mix in list(data_params[temp].keys())[::-1]:
            for irun in data_params[temp][mix].keys():
                for isample in data_params[temp][mix][irun]:
                    
                    # Working directory
                    wrkdir = os.path.join(temp, mix, "{:s}_{:s}".format(
                        irun, isample))
                    if not os.path.exists(wrkdir):
                        os.makedirs(wrkdir)
                    
                    # Check for completeness
                    outfle = data_outfle.format(temp, mix, irun, isample)
                    if os.path.exists(os.path.join(wrkdir, outfle)):
                        """
                        finished = False
                        with open(os.path.join(wrkdir, outfle), 'r') as f:
                            outlns = f.readlines()
                            for lne in outlns:
                                if 'Total time:' in lne:
                                    finished = True
                        if finished:
                            print(
                                'Simualation already done' 
                                + '(Temp: {:s}, Mix: {:s}, '.format(
                                    temp, mix)
                                + 'Run: {:s} Sample: {:s})'.format(
                                    irun, isample))
                            continue
                        """
                        finished = True
                        print(
                            'Skip Simulation '
                            + '(Temp: {:s}, Mix: {:s}, '.format(
                                temp, mix)
                            + 'Run: {:s} Sample: {:s})'.format(
                                irun, isample))
                        continue

                    # Copy data files to working directory
                    for fle in [
                        data_frqtag.format(temp, mix, irun, isample),
                        data_postag.format(temp, mix, irun, isample),
                        data_ucltag.format(temp, mix, irun, isample)]:
                        
                        # Source and target file
                        src = os.path.join(data_smpdir, fle)
                        trg = os.path.join(wrkdir, fle)
                        
                        # Copy files if necessary
                        if not os.path.exists(trg):
                            shutil.copyfile(src, trg)
                    
                    # Prepare script file
                    src = os.path.join(temp_scrdir, temp_script)
                    trg = os.path.join(wrkdir, temp_script)
                    with open(src, 'r') as f:
                        scrlns = f.read()
                    scrlns = scrlns.replace(temp_nprocs, '{:d}'.format(
                        comp_nprocs))
                    scrlns = scrlns.replace(temp_ntstep, '{:d}'.format(
                        comp_ntstep))
                    scrlns = scrlns.replace(temp_ntmdst, '{:d}'.format(
                        comp_ntmdst))
                    scrlns = scrlns.replace(temp_w0freq, '{:8.2f}'.format(
                        w0_constant[temp][mix]))
                    with open(trg, 'w') as f:
                        f.write(scrlns)
                    
                    # Change to work directory
                    os.chdir(wrkdir)
                    
                    # Print simulation start flag
                    print(
                        'Simualation starts ' 
                        + '(Temp: {:s}, Mix: {:s}, '.format(
                            temp, mix)
                        + 'Run: {:s} Sample: {:s})'.format(
                            irun, isample))
                    
                    # Compile script
                    subprocess.run(
                        ['gcc', temp_script, '-o', comp_exctbl, *comp_cflags])
                    
                    # Run script
                    with open(outfle, 'w') as f:
                        subprocess.run(
                            ['./{:s}'.format(comp_exctbl), mix, irun, isample], 
                            stdout=f)
                    
                    # Print finished flag
                    print(
                        'Simualation finished ' 
                        + '(Temp: {:s}, Mix: {:s}, '.format(
                            temp, mix)
                        + 'Run: {:s} Sample: {:s})'.format(
                            irun, isample))
                    
                    # Change to main directory
                    os.chdir(data_maindr)
    
    time.sleep(300)
