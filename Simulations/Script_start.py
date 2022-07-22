import os
import sys
import numpy as np

import subprocess

from shutil import copyfile, copytree
from itertools import product

#------------
# Parameters
#------------

# Number of cpus per task
Ncpu = 8

# Temperatures
temperatures = [300]

# Mixtures
mixtures = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95]

# System composition (KSCN, ACEM, TIP3)
composition = {
    0:   [75, 225,   0],
    10:  [75, 217,  23],
    20:  [75, 209,  50],
    30:  [75, 198,  81],
    40:  [75, 186, 119],
    50:  [75, 171, 164],
    60:  [75, 153, 219],
    70:  [75, 130, 290],
    80:  [75, 100, 381],
    90:  [75,  59, 506],
    95:  [75,  30, 570],
    100: [75,   0, 685]}

# Time step [ps]
dt = 0.001

# Propagation steps
# Number of simulation samples
Nsmpl = 10
# Heating steps
Nheat = 40000
# Equilibration steps
Nequi = 140000
# Production runs and dynamic steps
Nprod = 50
Ndyna = 100000

# Step size for storing coordinates
Nwrte = 10  # each 10 fs

# Instantaneous Normal Mode analysis
Nvtms = 3
Nvdly = 20
Nvstp = 3

# Maximum number of tasks
Nmaxt = 20

# Number of tasks per dcd file
Ntpdf = 1

# Skip frames
Nskpf = 10  # each 100 fs

# Optimization steps
Nstps = 100

# Main directory
maindir = os.getcwd()

# Workdir label
worktag = ''

# Template directory
tempdir = 'template'

# Source directory
sourdir = 'source'

#-----------------------------
# Preparations - General
#-----------------------------

# List all systems of different conditions
systems = np.array(list(product(temperatures, mixtures, np.arange(Nsmpl))))

# Iterate over systems
for sys in systems:
    
    # System data
    temp = sys[0]
    mix = sys[1]
    ismpl = sys[2]
    
    # Generate working directories
    workdir = os.path.join(maindir, worktag + str(temp))
    if not os.path.exists(workdir):
        os.mkdir(workdir)
    workdir = os.path.join(maindir, worktag + str(temp), "{:d}_{:d}".format(
        mix, ismpl))
    if not os.path.exists(workdir):
        os.mkdir(workdir)
        
    

#-----------------------------
# Preparations - Packmol
#-----------------------------

# Iterate over systems
for sys in systems:
    
    # System data
    temp = sys[0]
    mix = sys[1]
    ismpl = sys[2]
    workdir = os.path.join(maindir, worktag + str(temp), "{:d}_{:d}".format(
        mix, ismpl))
    
    # Prepare input file - packmol.inp
    ftemp = open(os.path.join(
        maindir, tempdir, 'packmol_temp.inp'), 'r')
    inplines = ftemp.read()
    ftemp.close()
    
    # Random seed
    inplines = inplines.replace('RRR', '{:d}'.format(np.random.randint(1e6)))
    # Source directory
    sdir = os.path.join(maindir, sourdir)
    inplines = inplines.replace('SSS', sdir)
    # Molecule numbers
    inplines = inplines.replace('AAA', '{:d}'.format(composition[mix][0]))
    inplines = inplines.replace('BBB', '{:d}'.format(composition[mix][1]))
    inplines = inplines.replace('CCC', '{:d}'.format(composition[mix][2]))
    # Delete zero molecule inputs
    newlines = ''
    writetemp = False
    templines = ''
    for line in inplines.split('\n'):
        
        if 'end structure' in line:
            templines += line + '\n'
            if number!=0:
                newlines += templines
            templines = ''
        elif 'structure' in line:
            writetemp = True
            templines += line + '\n'
        elif writetemp:
            if 'number' in line:
                number = int(line.split()[-1])
            templines += line + '\n'
        else:
            newlines += line + '\n'
    
    # Write input file
    finp = open(os.path.join(workdir, 'packmol.inp'), 'w')
    finp.write(newlines)
    finp.close()
    
    # Execute packmol
    subprocess.run(
        'cd {:s} ; packmol < packmol.inp'.format(workdir), shell=True)
    
#-----------------------------
# Preparations - Generation
#-----------------------------

# Iterate over systems
for sys in systems:
    
    # System data
    temp = sys[0]
    mix = sys[1]
    ismpl = sys[2]
    workdir = os.path.join(maindir, worktag + str(temp), "{:d}_{:d}".format(
        mix, ismpl))
    
    for f in [
        'crystal_image.str', 'ion_scn.lpun', 'toppar.str', 'toppar']:
        src = os.path.join(maindir, sourdir, f)
        trg = os.path.join(maindir, workdir, f)
        
        if os.path.isfile(src) and not os.path.exists(trg):
            copyfile(src, trg)
        elif os.path.isdir(src) and not os.path.exists(trg):
            copytree(src, trg)
        elif os.path.exists(trg):
            print("File/Directory '{:s}' already exist".format(src))
        else:
            print("Copying source '{:s}' omitted".format(src))
    
    # Get single pdb files
    fsys = open(os.path.join(workdir, 'init.pdb'), 'r')
    syslines = fsys.readlines()
    fsys.close()
    
    # SCN
    pdb_scn = ''
    for line in syslines:
        if 'ATOM' in line:
            if 'SCN' in line:
                pdb_scn += line
        else:
            pdb_scn += line  
    fscn = open(os.path.join(workdir, 'init.scn.pdb'), 'w')
    fscn.write(pdb_scn)
    fscn.close()
    
    # Pot
    pdb_pot = ''
    for line in syslines:
        if 'ATOM' in line:
            if 'POT' in line:
                pdb_pot += line
        else:
            pdb_pot += line  
    fpot = open(os.path.join(workdir, 'init.pot.pdb'), 'w')
    fpot.write(pdb_pot)
    fpot.close()
    
    # ACEM
    pdb_acem = ''
    Nacem = 0
    for line in syslines:
        if 'ATOM' in line:
            if 'ACEM' in line:
                Nacem += 1
                pdb_acem += line
        else:
            pdb_acem += line  
    if Nacem:
        facem = open(os.path.join(workdir, 'init.acem.pdb'), 'w')
        facem.write(pdb_acem)
        facem.close()
        
    # TIP3
    pdb_tip3 = ''
    Ntip3 = 0
    for line in syslines:
        if 'ATOM' in line:
            if 'TIP3' in line:
                Ntip3 += 1
                pdb_tip3 += line
        else:
            pdb_tip3 += line  
    if Ntip3:
        ftip3 = open(os.path.join(workdir, 'init.tip3.pdb'), 'w')
        ftip3.write(pdb_tip3)
        ftip3.close()
    
    # Prepare input file - step1.inp
    ftemp = open(os.path.join(
        maindir, tempdir, 'step1_temp.inp'), 'r')
    inplines = ftemp.readlines()
    ftemp.close()
    
    # Prepare parameters
    newlines = ''
    writeline = True
    for line in inplines:
        
        if writeline:
            newlines += line
            
        if '! Read ACEM' in line:
            if Nacem==0:
                writeline = False
        if '! END ACEM' in line:
            writeline = True
        if '! Read TIP3' in line:
            if Ntip3==0:
                writeline = False
        if '! END TIP3' in line:
            writeline = True
            
    # Write input file
    finp = open(os.path.join(workdir, 'step1.inp'), 'w')
    finp.write(newlines)
    finp.close()
    
#-----------------------------
# Preparations - Production
#-----------------------------

# Iterate over systems
for sys in systems:
    
    # System data
    temp = sys[0]
    mix = sys[1]
    ismpl = sys[2]
    workdir = os.path.join(maindir, worktag + str(temp), "{:d}_{:d}".format(
        mix, ismpl))
    
    # Prepare input file - step2.inp
    ftemp = open(os.path.join(maindir, tempdir, 'step2_temp.inp'), 'r')
    inplines = ftemp.read()
    ftemp.close()
    
    # Prepare parameters
    # Number of SCN residues
    inplines = inplines.replace('FFF', '{:d}'.format(composition[mix][0]))
    # Temperature
    inplines = inplines.replace('XXX', '{:d}'.format(temp))
    # First SCN index
    inplines = inplines.replace('IND', '{:d}'.format(1))
    # Number of SCN residues
    inplines = inplines.replace('NMX', '{:d}'.format(composition[mix][0]))
    # Random seed generator
    inplines = inplines.replace('RRRHH1', str(np.random.randint(1000000)))
    inplines = inplines.replace('RRRHH2', str(np.random.randint(1000000)))
    inplines = inplines.replace('RRRHH3', str(np.random.randint(1000000)))
    inplines = inplines.replace('RRRHH4', str(np.random.randint(1000000)))
    # Step size - Heating
    inplines = inplines.replace('TTT1', '{:d}'.format(Nheat))
    # Step size - Equilibration
    inplines = inplines.replace('TTT2', '{:d}'.format(Nequi))
    # Step size - Production
    inplines = inplines.replace('SSS', '{:.4f}'.format(dt))
    # Step size - Production
    inplines = inplines.replace('TTT3', '{:d}'.format(Ndyna))
    # Step size written to dcd file
    inplines = inplines.replace('NSV', '{:d}'.format(Nwrte))
    # Production runs
    inplines = inplines.replace('NNN', '{:d}'.format(Nprod))
    
    # Write input file
    finp = open(os.path.join(workdir, 'step2.inp'), 'w')
    finp.write(inplines)
    finp.close()

#----------------------------
# Preparations - INM script
#----------------------------

# Iterate over systems
for sys in systems:
    
    # System data
    temp = sys[0]
    mix = sys[1]
    ismpl = sys[2]
    workdir = os.path.join(maindir, worktag + str(temp), "{:d}_{:d}".format(
        mix, ismpl))
    
    # Prepare input file - inm_temp.py
    ftemp = open(os.path.join(maindir, tempdir, 'inm_temp.py'), 'r')
    inplines = ftemp.read()
    ftemp.close()
    
    # Prepare parameters
    # Mixture
    inplines = inplines.replace('%MIX%', '{:d}'.format(mix))
    # DCD selection
    inplines = inplines.replace('%TDCD%', '{:d}'.format(Nvtms))
    inplines = inplines.replace('%DDCD%', '{:d}'.format(Nvdly))
    inplines = inplines.replace('%SDCD%', '{:d}'.format(Nvstp))
    # Production runs
    inplines = inplines.replace('%NPROD%', '{:d}'.format(Nprod))
    # Tasks management
    inplines = inplines.replace('%NMAXT%', '{:d}'.format(Nmaxt))
    inplines = inplines.replace('%NTPDF%', '{:d}'.format(Ntpdf))
    inplines = inplines.replace('%NSKPF%', '{:d}'.format(Nskpf))
    inplines = inplines.replace('%NSTPS%', '{:d}'.format(Nstps))
    
    # Write input file
    finp = open(os.path.join(workdir, 'inm.py'), 'w')
    finp.write(inplines)
    finp.close()
    
    # Copy template files in working directory
    for f in ['step3_temp.inp', 'step3_temp.sh', 'step3_temp.py']:
        src = os.path.join(maindir, tempdir, f)
        trg = os.path.join(workdir, f)
        
        #if os.path.isfile(src) and not os.path.exists(trg):
        copyfile(src, trg)
        #else:
            #print("Copying template '{:s}' omitted".format(src))
            
#----------------------------
# Preparations - run script
#----------------------------


# Iterate over systems
scrlines = ""
for sys in systems:
    
    # System data
    temp = sys[0]
    mix = sys[1]
    ismpl = sys[2]
    workdir = os.path.join(maindir, worktag + str(temp), "{:d}_{:d}".format(
        mix, ismpl))
    
    # Prepare script file
    ftemp = open(os.path.join(tempdir, 'run_temp.sh'), 'r')
    inplines = ftemp.read()
    ftemp.close()
    
    # Prepare parameters
    # Temperature
    inplines = inplines.replace('TPAR', '{:d}'.format(temp))
    # Mixture
    inplines = inplines.replace('MPAR', '{:d}'.format(mix))
    # Sample
    inplines = inplines.replace('SPAR', '{:d}'.format(ismpl))
    # Tasks
    inplines = inplines.replace('CPAR', '{:d}'.format(Ncpu))
    
    # Write script file
    finp = open(os.path.join(workdir, 'run.sh'), 'w')
    finp.write(inplines)
    finp.close()
    
    # Prepare observation file
    ftemp = open(os.path.join(tempdir, 'observe_temp.py'), 'r')
    inplines = ftemp.read()
    ftemp.close()
    
    # Prepare parameters
    # Script file
    inplines = inplines.replace('%RFILE%', 'run.sh')
    # Input file
    inplines = inplines.replace('%IFILE%', 'step2.inp')
    # output file
    inplines = inplines.replace('%OFILE%', 'step2.out')
    
    # Write input file
    finp = open(os.path.join(workdir, 'observe.py'), 'w')
    finp.write(inplines)
    finp.close()
    
    # Start lines
    scrlines += "cd {:s}\n".format(workdir)
    scrlines += "python observe.py &\n"
    scrlines += "python inm.py &\n"
    scrlines += "cd {:s}\n".format(maindir)
    
# Write start file
finp = open("start.sh", 'w')
finp.write(scrlines)
finp.close()


    
