# Basics
import os
import sys
import time
import numpy as np
from itertools import product
from glob import glob

# Trajectory reader
import MDAnalysis

# Matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# ASE
from ase import Atoms
from ase.io import read, write
from ase.visualize import view

# Multiprocessing
from multiprocessing import Pool

#------------
# Parameters
#------------

# Source directory
source = '.'

# Temperatures
temp_tag = ''
temperatures = [300]

# Mixtures
mixtures = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95]

# Number of simulation samples
Nsmpl = 10

# Trajectory source
traj_dcdtag = 'dyna.{:d}.dcd'
Nvtms = 3
Nvdly = 20
Nvstp = 3
traj_dcdsel = (
    np.arange(Nvtms*Nvdly)
    .reshape(Nvtms, -1)[:,:Nvstp])
traj_dcdtme = 0.25
# first: split condition, second: index for irun
traj_dcdinfo = ['.', 1]
traj_crdfile = 'step1_pdbreader.crd'
traj_psffile = 'step1_pdbreader.psf'

# Frequency source
freq_resdir = 'results'
freq_resfiletag = 'freq_{:d}_{:d}_{:s}_{:d}_*.dat'
freq_ntasks = 1
freq_vibnum = -1
freq_Nskip = 10
# Indices: mix, irun, residue, ires, ifile
freq_resfileinfo = ['_', [1, 2, 3, 4, 5]]

# Frequency range 
freq_warning = [1850, 2250]

# Residue of interest
eval_residue = 'SCN'
eval_resids = 'all'

# Regarding residues
eval_resreg = ['SCN', 'ACEM', 'TIP3', 'K']
eval_resnum = [3, 9, 3, 1]
eval_ressym = {
    'SCN': ['N', 'C', 'S'],
    'ACEM': ['C', 'C', 'N', 'H', 'H', 'O', 'H', 'H', 'H'],
    'TIP3': ['O', 'H', 'H'],
    'K': ['K']}

# Writing step size in ps
eval_stepsize = 0.2
eval_maxtme = 250.0

# Workdir label
worktag = ''

# Result directory
res_maindir = 'results_samples'

#--------------
# Preparations
#--------------

# Iterate over systems
info_systems = np.array(list(product(temperatures, mixtures)))

# Get atoms and pair information
numres = {}
atomsint = {}

# Iterate over systems
for sys in info_systems:
    
    # Data directory
    temp = sys[0]
    mix = sys[1]
    for irun in range(Nsmpl):
        
        datadir = os.path.join(source, worktag + str(temp), "{:d}_{:d}".format(
            mix, irun))
        
        if os.path.exists(os.path.join(datadir, traj_crdfile)):
            break
    
    # System tag
    tag = '{:d}_{:d}'.format(temp, mix)
    
    # Check if all residues are considered
    all_residues = eval_resreg
    if eval_residue not in eval_resreg:
        all_residues.append(eval_residue)
    
    # Read residue information
    listres = {}
    numres[tag] = {}
    for ires, res in enumerate(all_residues):
        info_res = []
        with open(os.path.join(datadir, traj_crdfile), 'r') as f:
            for line in f:
                if res in line:
                    info_res.append(line)
        listres[res] = info_res
        numres[tag][res] = int(len(info_res)/eval_resnum[ires])
        
    # Get residue atom numbers
    atomsint[tag] = {}
    for ires, res in enumerate(all_residues):
        atomsinfo = np.zeros(
            [numres[tag][res], eval_resnum[ires]], dtype=int)
        for ir in range(numres[tag][res]):
            ri = eval_resnum[ires]*ir
            for ia in range(eval_resnum[ires]):
                info = listres[res][ri + ia].split()
                atomsinfo[ir, ia] = int(info[0]) - 1
        atomsint[tag][res] = atomsinfo
        
# Make result directory
if not os.path.exists(res_maindir):
    os.mkdir(res_maindir)
    
#---------------------
# Collect system data
#---------------------

info_systems = np.array(
    list(
        product(
            temperatures, mixtures, 
            list(range(Nsmpl)), 
            list(range(traj_dcdsel.shape[0]))
            )
        )
    )

unfinished_buisness = True
while unfinished_buisness:
    
    n = 0
    
    # Iterate over systems and resids
    for sys in info_systems:
        
        # Data directory
        temp = sys[0]
        mix = sys[1]
        irun = sys[2]
        isample = sys[3]
        sample = traj_dcdsel[isample]
        datadir = os.path.join(source, worktag + str(temp), "{:d}_{:d}".format(
            mix, irun))
        
        # System tag
        tag = '{:d}_{:d}'.format(temp, mix)
        
        # Check eval_resids
        if isinstance(eval_resids, str):
            resids = []
            if eval_resids=='all':
                resids = list(range((numres[tag][eval_residue])))
        
        # Check for frequency files
        all_is_there = True
        for ir, resid in enumerate(resids):
            for idcd in sample:
                eval_freqfiles = np.array(glob(os.path.join(
                        datadir, freq_resdir, freq_resfiletag.format(
                            mix, idcd, eval_residue, resid))))
                
                if not len(eval_freqfiles)//freq_ntasks:
                    all_is_there = False
                
        if not all_is_there:
            continue
        
        # Read dcd files 
        #----------------
        
        # Output file
        snapsfile = os.path.join(
            res_maindir, 'snaps_{:d}_{:d}_{:d}_{:d}_{:s}.xyz'.format(
                temp, mix, irun, isample, eval_residue))
        freqsfile = os.path.join(
            res_maindir, 'freqs_{:d}_{:d}_{:d}_{:d}_{:s}.dat'.format(
                temp, mix, irun, isample, eval_residue))
        ucellfile = os.path.join(
            res_maindir, 'ucell_{:d}_{:d}_{:d}_{:d}_{:s}.dat'.format(
                temp, mix, irun, isample, eval_residue))
        
        if os.path.exists(freqsfile):
            continue
        
        # Get dcd files
        #dcdfiles = np.array(glob(os.path.join(datadir, traj_dcdtag)))
        dcdfiles = np.array([
            os.path.join(datadir, traj_dcdtag.format(ii)) for ii in sample])
        iruns = np.array([
            int(dcdfile.split('/')[-1].split(traj_dcdinfo[0])[traj_dcdinfo[1]])
            for dcdfile in dcdfiles])
        psffile = os.path.join(datadir, traj_psffile)
        
        # Sort dcd files
        dcdsort = np.argsort(iruns)
        dcdfiles = dcdfiles[dcdsort]
        iruns = iruns[dcdsort]
        
        # Open snapsfile
        fsnaps = open(snapsfile, 'w')
        
        # Open freqssfile
        ffreqs = open(freqsfile, 'w')
        
        # Open bsizesfile
        fucell = open(ucellfile, 'w')
        
        # Snapshot counter
        isnaps = 0
        
        # Frequency counter
        ifreqs = 0
        
        # Total time counter
        tottime = 0.0
        
        # Iterate over dcd files
        for dcdfile in dcdfiles:
            
            # Check time
            if tottime >= eval_maxtme:
                break
            
            idcd = int(
                dcdfile.split('/')[-1].split(traj_dcdinfo[0])[traj_dcdinfo[1]])
            print('Read dcd file: Temp {:d} K, Mix {:d}%, dcd {:d}'.format(
                temp, mix, idcd))
            
            # Open dcd file
            dcd = MDAnalysis.Universe(psffile, dcdfile)
            
            # Get trajectory parameter
            Nframes = len(dcd.trajectory)
            Nskip = int(dcd.trajectory.skip_timestep)
            dt = np.round(
                float(dcd.trajectory._ts_kwargs['dt']), decimals=8)/Nskip
            
            # Get atom names
            dcdatoms = dcd.atoms.names
            dcdnatoms = len(dcdatoms)
            
            # Initialize frequency list
            eval_freqs = [[] for _ in resids]
            
            # Iterate over frames and write xyz information
            dcdxyz = ''
            dcdidx = []
            ucellline = '{:<5d}' + ' {:10.6f}'*3 + '\n'
            for i, frame in enumerate(dcd.trajectory):
                
                # Current time step
                timestep = tottime + i*dt*Nskip
                
                # Check time
                if timestep >= eval_maxtme:
                    break
                
                # Current cell size
                ucell = frame._unitcell
                
                # Check timestep
                rest = timestep%eval_stepsize
                maxrest = dt*Nskip/2.
                if (rest < maxrest) or (eval_stepsize - rest < maxrest):
                    
                    # Add atom number
                    dcdxyz += '{:d}\n'.format(dcdnatoms)
                    
                    # Add comment (extended xyz format)
                    extcomment = (
                        'Lattice=' 
                        + '"{:.6f} 0.0 0.0 0.0 {:.6f} 0.0 0.0 0.0 {:.6f}"'.
                        format(*ucell) + ' Properties=species:S:1:pos:R:3 '
                        + 'Time={:.3f}\n'.format(timestep))
                            
                    dcdxyz += extcomment
                    
                    # Add coordinates
                    for j, pos in enumerate(frame._pos):
                        
                        dcdxyz += '  {:<9s}'.format(dcdatoms[j])
                        dcdxyz += (
                            '{: 12.6f}    {: 12.6f}    {: 12.6f}\n'.format(
                                *pos))
                        
                    # Write unit cell line
                    ufline = ucellline.format(isnaps, *ucell)
                    
                    # Write line to file
                    fucell.write(ufline)
                
                    # Save frame index
                    dcdidx.append(i)
                    
                    # Increment snapshot counter
                    isnaps += 1
            
            # Update total time counter
            tottime = tottime + (i + 1)*dt*Nskip
            
            # Write structure to file
            fsnaps.write(dcdxyz)
            
            print(
                'Read frequency files: Temp {:d} K, Mix {:d}%, dcd {:d}'.format(
                    temp, mix, idcd))
            
            # Iterate over resids
            for ir, resid in enumerate(resids):
                
                # Get frequency files
                eval_freqfiles = np.array(glob(os.path.join(
                    datadir, freq_resdir, freq_resfiletag.format(
                        mix, idcd, eval_residue, resid))))
    
                # Sort frequency files
                sortfile = []
                for ffile in eval_freqfiles:
                    sortfile.append(int(
                        ffile.split('/')[-1].split('.')[-2].split(
                        freq_resfileinfo[0])[freq_resfileinfo[1][4]]))
                sortfile = np.array(sortfile)
                
                eval_freqfiles = eval_freqfiles[sortfile]
                
                # Read frequency file
                isnap = 0
                for ffile in eval_freqfiles:
                    with open(ffile, 'r') as f:
                        for il, line in enumerate(f):
                            if isnap in dcdidx:
                                eval_freqs[resid].append(
                                    float(line.split()[freq_vibnum - 1]))
                                if (eval_freqs[resid][-1] < freq_warning[0] or
                                    eval_freqs[resid][-1] > freq_warning[1]):
                                    print(
                                        'WARNING: {:s}, line {:d}, Freq {:.4f}'.
                                        format(
                                            ffile, il, eval_freqs[resid][-1]))
                            isnap += freq_Nskip
            
            # Write frequencies to file
            print(temp, mix, irun, isample)
            eval_freqs = np.array(eval_freqs).T
            linetemp = '{:<5d}' + ' {:10.6f}'*eval_freqs.shape[1] + '\n'
            for i, idx in enumerate(dcdidx):
                
                # Create line
                fline = linetemp.format(ifreqs, *eval_freqs[i])
                
                # Write line to file
                ffreqs.write(fline)
                
                # Increment snapshot counter
                ifreqs += 1
            
        # Close snapsfile
        fsnaps.close()
        
        # Close freqsfile
        ffreqs.close()
        
        # Close ucellfile
        fucell.close()
        
        n += 1
    
    if n==len(info_systems):
        unfinished_buisness = False
        
    time.sleep(300)


#-----------------------
# Plot INM each samples
#-----------------------


# Binning paramater
dbin_freqs = 5.0
bins_freqs = np.arange(
    freq_warning[0], freq_warning[1] + dbin_freqs, dbin_freqs)
cntr_freqs = bins_freqs[:-1] + dbin_freqs/2.

# Plot properties
SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 18

pltfont = 'normal'
plt.rc('font', size=SMALL_SIZE, weight=pltfont)
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# Figure parameters
figsize = (10, 6)
left = 0.15
bottom = 0.15
column = [0.70, 0.00]
row = [0.70, 0.00]

skip_sample = 1

# Iterate over systems and resids
for temp in temperatures:
    
    # Create figure and axis system
    fig = plt.figure(num=0, figsize=figsize)
    axs = fig.add_axes([left, bottom, column[0], row[0]])
    
    for mix in mixtures:
        
        # INM histogram
        hist_freqs = np.zeros(cntr_freqs.shape[0], dtype=int)
        
        for irun in range(Nsmpl):
            
            # Data directory
            datadir = os.path.join(source, worktag + str(temp), "{:d}_{:d}".format(
                mix, irun))
            
            # System tag
            tag = '{:d}_{:d}'.format(temp, mix)
            
            # Check eval_resids
            if isinstance(eval_resids, str):
                resids = []
                if eval_resids=='all':
                    resids = list(range((numres[tag][eval_residue])))
            
            for isample, sample in enumerate(traj_dcdsel):
                
                if isample%skip_sample:
                    continue
                
                # Get dcd files
                dcdfiles = np.array([
                    os.path.join(datadir, traj_dcdtag.format(ii)) for ii in sample])
                iruns = np.array([
                    int(dcdfile.split('/')[-1].split(traj_dcdinfo[0])[traj_dcdinfo[1]])
                    for dcdfile in dcdfiles])
                psffile = os.path.join(datadir, traj_psffile)
                
                # Frequency counter
                ifreqs = 0
                
                # Total time counter
                tottime = 0.0
                
                # Iterate over dcd files
                for dcdfile in dcdfiles:
                    
                    # Prepare INM frequency list
                    eval_freqs = [[] for _ in resids]
                    
                    idcd = int(
                        dcdfile.split('/')[-1].split(traj_dcdinfo[0])[traj_dcdinfo[1]])
                    print('Read dcd file: Temp {:d} K, Mix {:d}%, Run {:d}, Sample {:d}, dcd {:d}'.format(
                        temp, mix, irun, isample, idcd))
                    
                    # Iterate over resids
                    for ir, resid in enumerate(resids):
                        
                        # Get frequency files
                        eval_freqfiles = np.array(glob(os.path.join(
                            source, temp_tag + str(temp), str(mix) + '_' + str(irun), 
                            freq_resdir, freq_resfiletag.format(
                                mix, idcd, eval_residue, resid))))
                        
                        # Sort frequency files
                        sortfile = []
                        for ffile in eval_freqfiles:
                            sortfile.append(int(
                                ffile.split('/')[-1].split('.')[-2].split(
                                freq_resfileinfo[0])[freq_resfileinfo[1][4]]))
                        sortfile = np.array(sortfile)
                        
                        if not len(eval_freqfiles):
                            continue
                        
                        # Read frequency file
                        for ffile in eval_freqfiles:
                            with open(ffile, 'r') as f:
                                for il, line in enumerate(f):
                                    eval_freqs[resid].append(
                                        float(line.split()[freq_vibnum - 1]))
                                    if (eval_freqs[resid][-1] < freq_warning[0] or
                                        eval_freqs[resid][-1] > freq_warning[1]):
                                        print(
                                            'WARNING: {:s}, line {:d}, Freq {:.4f}'.
                                            format(
                                                ffile, il, eval_freqs[resid][-1]))
                    
                    # INM frequencies
                    eval_freqs = np.array([frqs for frqs in eval_freqs if len(frqs)]).reshape(-1)
                    
                    # Binning
                    hist_freqs[:] += np.histogram(
                        eval_freqs, bins=bins_freqs)[0]
                    
        # Plot INMs
        hist = hist_freqs
        hist = np.array(hist, dtype=float)/(np.sum(hist)*dbin_freqs)
        
        axs.plot(
            cntr_freqs, hist, 
            label="{:d}%".format(mix))
            
    # Axis label
    axs.set_title("INM spectra {:d} K ".format(temp) + r"($[0, 5]$\,ns)")
    axs.set_xlabel(r"Frequency (cm$^{-1}$)")
    axs.set_ylabel(r"Intensity")
    
    # Axis limits
    axs.set_xlim(bins_freqs[0], 2150)
    
    # Add legend
    axs.legend(loc='upper left', title="Mixture")
    
    # Save figure
    fig.savefig(
        os.path.join(res_maindir, "INM_{:d}.png".format(
            temp)),
        format='png', dpi=100)
    plt.close()
    
            
     
