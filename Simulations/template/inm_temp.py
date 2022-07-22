import os
import sys
import time 
import getpass
import subprocess
import numpy as np

import MDAnalysis

from itertools import product
from glob import glob

#------------
# Parameters
#------------

# Eutectic Mixture 
mixture = %MIX%

# Trajectory data
seldcd = [%TDCD%, %DDCD%, %SDCD%]
dcdname = 'dyna.{:d}.dcd'
dcdselection = (
    np.arange(seldcd[0]*seldcd[1])
    .reshape(seldcd[0], -1)[:,:seldcd[2]].reshape(-1))

# Residue label
residues = ['SCN']

# Number of atoms per residue
Natms = [3]

# Maximum number of tasks running
Nmaxt = %NMAXT%

# Number of tasks per dcd file
Ntpdf = %NTPDF%

# Skip frames
Nskpf = %NSKPF%

# Optimization steps
Nstps = %NSTPS%

# Template file
tinpfile = 'step3_temp.inp'
trunfile = 'step3_temp.sh'
tevafile = 'step3_temp.py'

# Results directory in workdir
rsltdir = 'results'

# Flag file directory
flgsdir = 'flags'

#--------------
# Preparations
#--------------

# Get dcd files
dcdfiles = [dcdname.format(ii) for ii in dcdselection]

# Sort dcd files
index = []
for dcdfile in dcdfiles:
    ii = int(dcdfile.split('/')[-1].split('.')[1])
    index.append(ii)
isort = np.argsort(index)
ifiles = np.array(index)[isort] 
dcdfiles = np.array(dcdfiles)[isort]

# Get number of images per dcd file
psffile = 'step1_pdbreader.psf'
while not os.path.exists(dcdfiles[1]):
    time.sleep(300)
dcd = MDAnalysis.Universe(psffile, dcdfiles[0])
Nimages = len(dcd._trajectory)

# Scan for residues
resatoms = {}
resnum = {}
for ires, res in enumerate(residues):

    # Read system information
    listres = []
    resnumres = 0
    with open('step1_pdbreader.crd', 'r') as f:
        for line in f:
            if res in line:
                listres.append(line)
                resnumres += 1
    resnumres = int(resnumres/Natms[ires])
    
    # Get atom numbers for each residue
    atoms = np.zeros([resnumres, Natms[ires]], dtype=int)
    for ir in range(resnumres):
        ri = Natms[ires]*ir
        for ia in range(Natms[ires]):
            atoms[ir, ia] = int(listres[ri + ia].split()[0])
    
    resatoms[res] = atoms
    resnum[res] = resnumres
    
if not os.path.exists(rsltdir):
    os.mkdir(rsltdir)

if not os.path.exists(flgsdir):
    os.mkdir(flgsdir)
    
#--------------
# Calculations
#--------------

# Get tasks
systems = []
for ires, res in enumerate(residues):
    systems += list(product(residues, range(resnum[res])))

# Task ids and script file list
tskids = []
tsksrc = []

# Start jobs for each dcd file
for istep, file_i in enumerate(dcdfiles):
    
    ifile = ifiles[istep]
    if not ifile in dcdselection:
        continue
    
    # Iterate over residues
    for sys_i in systems:
        
        # Parameters
        res = sys_i[0]
        ires = sys_i[1]
        
        # Iterate over tasks per residue
        for itask in range(Ntpdf):
            
            # Script file
            scrfile = 'step3_{:d}_{:s}_{:d}_{:d}.sh'.format(
                ifile, res, ires, itask)
            # Input file
            inpfile = 'step3_{:d}_{:s}_{:d}_{:d}.inp'.format(
                ifile, res, ires, itask)
            # Evaluation file
            evafile = 'step3_{:d}_{:s}_{:d}_{:d}.py'.format(
                ifile, res, ires, itask)
            # Output file
            outfile = 'step3_{:d}_{:s}_{:d}_{:d}.out'.format(
                ifile, res, ires, itask)
            # Result file
            resfile = os.path.join(
                rsltdir, 'freq_{:d}_{:d}_{:s}_{:d}_{:d}.dat'.format(
                    mixture, ifile, res, ires, itask))
            # Finished flag file
            flgfile = os.path.join(
                flgsdir, 'flag_{:d}_{:d}_{:s}_{:d}_{:d}.txt'.format(
                    mixture, ifile, res, ires, itask))
            
            # Job tag
            jobtag = '{:d}_{:d}_{:s}_{:d}_{:d}'.format(
                mixture, ifile, res, ires, itask)
            
            # Check flag file
            if os.path.exists(resfile):
                filesize = os.path.getsize(resfile)
            else:
                filesize = 0
            if os.path.exists(flgfile) and filesize > 100:
                continue
            
            # Prepare input file
            ftemp = open(tinpfile, 'r')
            inplines = ftemp.read()
            ftemp.close()
            
            # Prepare parameters
            # DCD file
            inplines = inplines.replace('DDD', file_i)
            # First SCN index
            inplines = inplines.replace('IND', '{:d}'.format(1))
            # Number of SCN residues
            inplines = inplines.replace('NMX', '{:d}'.format(
                resnum[res]))
            # Constraint parameter
            const = ''
            atoms = resatoms[res][ires]
            stror = ' .or. '
            for atom in atoms:
                const += 'bynum {:d}'.format(atom)
                const += stror
            const = const[:-len(stror)]
            inplines = inplines.replace('BYS', const)
            # Image parameter
            Nstart = int(itask*Nimages/Ntpdf)
            Nend = int((itask + 1)*Nimages/Ntpdf)
            inplines = inplines.replace('SKP', str(Nskpf))
            inplines = inplines.replace('STR', str(Nstart))
            inplines = inplines.replace('LST', str(Nend))
            # Optimization
            inplines = inplines.replace('ONS', str(Nstps))
            inplines = inplines.replace('ONP', str(Nstps))
            # Vibration vector
            vibvec = 3*resatoms[res].shape[1]
            inplines = inplines.replace('VNM', str(vibvec))
            
            # Write input file
            finp = open(inpfile, 'w')
            finp.write(inplines)
            finp.close()
            
            # Prepare script file
            ftemp = open(trunfile, 'r')
            inplines = ftemp.read()
            ftemp.close()
            
            # Prepare parameters
            inplines = inplines.replace('JBNM', jobtag)
            inplines = inplines.replace('INPF', inpfile)
            inplines = inplines.replace('OUTF', outfile)
            inplines = inplines.replace('EVAF', evafile)
            inplines = inplines.replace('RESF', resfile)
            inplines = inplines.replace('FLGF', flgfile)
            
            # Write script file
            finp = open(scrfile, 'w')
            finp.write(inplines)
            finp.close()
            
            # Prepare evaluation file
            ftemp = open(tevafile, 'r')
            inplines = ftemp.read()
            ftemp.close()
            
            # Write evaluation file
            finp = open(evafile, 'w')
            finp.write(inplines)
            finp.close()
            
            # Check if next dcd file already exists
            if not os.path.exists(dcdname.format(ifile + 1)):
                ready = False
                while not ready:
                    time.sleep(300)
                    ready = os.path.exists(dcdname.format(ifile + 1))
            
            # Execute CHARMM
            task = subprocess.run(["sbatch", scrfile], capture_output=True)
            tskids.append(int(task.stdout.decode().split()[-1]))
            tsksrc.append(scrfile)
            print(task.stdout.decode())
            
            # Check if maximum task number is reached 
            if len(tskids) < Nmaxt:
                continue
            
            max_cap = True
            while max_cap:
            
                # Get current tasks ids
                user = getpass.getuser()
                tsklist = subprocess.run(
                    ['squeue', '-u', user], capture_output=True)
                
                idslist = [
                    int(job.split()[0])
                    for job in tsklist.stdout.decode().split('\n')[1:-1]]
                
                # Check if task is still running
                for it, tid in enumerate(tskids):
                    if not tid in idslist:
                        del tskids[it]
                        if os.path.exists(tsksrc[it]):
                            os.remove(tsksrc[it])
                        del tsksrc[it]
                        if len(tskids) <= Nmaxt:
                            max_cap = False
                        break
                
                # Wait 30 seconds
                if max_cap:
                    time.sleep(30)

