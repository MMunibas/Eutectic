# Basics
import os
import sys
import time
import numpy as np
import pandas as pd
from itertools import product, combinations_with_replacement
from glob import glob
import pickle as pkl

# Clustering algorithm and auxiliaries 
from scipy.spatial import distance_matrix
from scipy.spatial.distance import squareform, pdist

# Trajectory reader
import MDAnalysis
from MDAnalysis.analysis.distances import distance_array, self_distance_array
from MDAnalysis.coordinates.memory import MemoryReader
from MDAnalysis import transformations

# Matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, HPacker, VPacker

# Viewer
from ase.visualize import view

# Multiprocessing
from multiprocessing import Pool

#------------
# Parameters
#------------

# Source directory
source = '.'

# Temperatures
temperatures = [300]

# Mixtures
mixtures = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95]

# Parallel evaluation runs
tasks = 1

# Runs
Nrun = 10

# Trajectory source
traj_dcdtag = 'dyna.*.dcd'
# first: split condition, second: index for irun
traj_dcdinfo = ['.', 1]
traj_crdfile = 'step1_pdbreader.crd'
traj_psffile = 'step1_pdbreader.psf'

# Frequency source
freq_dirtag = 'Vib'
freq_resdir = 'results'
freq_resfiletag = 'freq_{:s}_{:d}_{:s}_{:d}_*.dat'
freq_vibnum = -1
# Indices: mix, irun, residue, ires, ifile
freq_resfileinfo = ['_', [1, 2, 3, 4, 5]]

# Result directory
dirs_maindir = 'results_coordination'
dirs_evaldir = 'evalfiles'

# Residue of interest: label, residue center
eval_reslab = 'TIP3'
eval_resctr = 0
eval_resdst = [3.0, 3.5, 4.0, 4.5, 5.0, 5.5]
evel_nummax = 48
eval_pltlab = r'H_2O'

# Time step for averaging in ps
eval_timestep = 1.000
#eval_timestep = 1.000

# Maximum time to evaluate in ps
eval_maxtime = 5000.0
#eval_maxtime = 100.0

# System information
syst_reslab = ['SCN', 'ACEM', 'TIP3', 'K']
syst_resnum = {
    'SCN': 3,
    'ACEM': 9,
    'TIP3': 3,
    'K': 1}
syst_ressym = {
    'SCN': ['N', 'C', 'S'],
    'ACEM': ['C', 'C', 'N', 'H', 'H', 'O', 'H', 'H', 'H'],
    'TIP3': ['O', 'H', 'H'],
    'K': ['K']}
syst_reschr = {
    'SCN': [-0.46, -0.36, -0.18],
    'ACEM': [-0.27, 0.55, -0.62, 0.32, 0.30, -0.55, 0.09, 0.09, 0.09],
    'TIP3': [-0.84, 0.42, 0.42],
    'K': [1.00]}
syst_resmss = {
    'SCN': [14.01, 12.01, 32.06],
    'ACEM': [12.01, 12.01, 14.01, 1.01, 1.01, 12.01, 1.01, 1.01, 1.01],
    'TIP3': [14.01, 1.01, 1.01],
    'K': [39.10]}
syst_seglab = {
    'SCN': 'SCN',
    'ACEM': 'ACEM',
    'TIP3': 'TIP3',
    'K': 'POT'}

# Plot options

# Fontsize
SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 18

#plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
font = 'normal'
plt.rc('font', size=SMALL_SIZE, weight=font)  # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

dpi = 300

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
    temp = str(sys[0])
    mix = str(sys[1])
    datadir = os.path.join(source, temp, mix + '_0')
    
    # System tag
    tag = temp + '_' + mix
    
    # Check if all residues are considered
    all_residues = syst_reslab
    if eval_reslab not in syst_reslab:
        all_residues.append(eval_reslab)
    
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
        numres[tag][res] = int(len(info_res)/syst_resnum[res])
        
    # Get residue atom numbers
    atomsint[tag] = {}
    for ires, res in enumerate(all_residues):
        atomsinfo = np.zeros(
            [numres[tag][res], syst_resnum[res]], dtype=int)
        for ir in range(numres[tag][res]):
            ri = syst_resnum[res]*ir
            for ia in range(syst_resnum[res]):
                info = listres[res][ri + ia].split()
                atomsinfo[ir, ia] = int(info[0]) - 1
        atomsint[tag][res] = atomsinfo
        
# Make result directory
if not os.path.exists(dirs_maindir):
    os.mkdir(dirs_maindir)
    
if not os.path.exists(os.path.join(dirs_maindir, dirs_evaldir)):
    os.mkdir(os.path.join(dirs_maindir, dirs_evaldir))


#---------------------
# Collect system data
#---------------------

# Iterate over systems and resids
info_systems = np.array(list(product(temperatures, mixtures, range(Nrun))))

def read_sys(i):
    
    # Begin timer
    start = time.time()
    
    # Data directory
    temp = str(info_systems[i][0])
    mix = str(info_systems[i][1])
    run = str(info_systems[i][2])
    datadir = os.path.join(source, temp, "{:s}_{:s}".format(mix, run))
    
    # System tag
    tag = temp + '_' + mix
    
    # Read dcd files and get atom distances
    #---------------------------------------
    
    # Get dcd files
    dcdfiles = np.array(glob(os.path.join(datadir, traj_dcdtag)))
    iruns = np.array([
        int(dcdfile.split('/')[-1].split(traj_dcdinfo[0])[traj_dcdinfo[1]])
        for dcdfile in dcdfiles])
    psffile = os.path.join(datadir, traj_psffile)
    crdfile = os.path.join(datadir, traj_crdfile)
    
    # Sort dcd files
    dcdsort = np.argsort(iruns)
    dcdfiles = dcdfiles[dcdsort]
    iruns = iruns[dcdsort]
    
    # Initialize cluster dictionary
    coordination = {
        'residue' : eval_reslab,
        'cutoff': np.array(eval_resdst),
        'N': np.zeros([len(eval_resdst), evel_nummax + 1])}
    
    crdnfile = os.path.join(
        dirs_maindir, dirs_evaldir, 
        'crdn_{:s}_{:s}_{:s}_{:s}.npy'.format(temp, mix, run, eval_reslab))
    
    if not os.path.exists(crdnfile):
        
        # Initialize distance matrix and time list
        dmtrx_list = []
        time_list = []
        
        # Initialize trajectory time counter in ps
        traj_time_dcd = 0.0
        
        # Iterate over dcd files
        for idcd, dcdfile in enumerate(dcdfiles):
            
            # Open dcd file
            dcd = MDAnalysis.Universe(psffile, dcdfile)
            
            # Get atom types
            atoms = [ai[:1] for ai in dcd._topology.names.values]
            
            # Get trajectory parameter
            Nframes = len(dcd.trajectory)
            Natoms = len(atoms)
            Nskip = int(dcd.trajectory.skip_timestep)
            dt = np.round(
                float(dcd.trajectory._ts_kwargs['dt']), decimals=8)/Nskip
            
            # Evaluate structure at centre of eval_timestep
            timesteps = np.arange(Nframes)*dt*Nskip
            timewindows = []
            for istep, timestep in enumerate(timesteps):
                
                aux = timestep%eval_timestep
                if (aux < dt*Nskip/2.) or (eval_timestep - aux < dt*Nskip/2.):
                    
                    timewindows.append(istep)
                    
            if timesteps[-1] not in timewindows:
                
                timewindows.append(istep + 1)
                    
            timewindows = np.array(timewindows)
            timecenters = np.array([
                (timewindows[it] + timewindows[it - 1])//2 
                for it in range(1, len(timewindows))])
            
            # Iterate over centers frames
            for ic, tc in enumerate(timecenters):
                
                # Get positions
                positions = dcd.trajectory[tc]._pos
                
                # Get cell information
                cell = dcd.trajectory[tc]._unitcell
                maxc = np.max(cell[:3])
                
                # Residue positions
                eval_pos = positions[atomsint[tag][eval_reslab]]
                
                # Residue center
                if isinstance(eval_resctr, int):
                    res_cnt = eval_pos[:, eval_resctr]
                elif isinstance(eval_resctr, (tuple, list, np.ndarray)):
                    totmass = np.sum(
                        [syst_resmss[eval_reslab][ai] for ai in eval_resctr])
                    res_cnt = np.sum([
                        eval_pos[:, ai]*syst_resmss[eval_reslab][ai] 
                        for ai in eval_resctr], axis=0)/totmass
                else:
                    raise IOError("eval_resctr has to be a list or an integer")
                
                # Calculate distance matrix
                dtria = self_distance_array(res_cnt, box=cell)
                dmtrx = squareform(dtria)
                
                for ir, rcut in enumerate(eval_resdst):
                    
                    # Apply conditions
                    coordinated = np.logical_and(dmtrx > 0.1, dmtrx < rcut)
                    
                    # Sum up coordination numbers
                    ncoords = np.sum(coordinated, axis=1)
                    
                    # Do binning
                    bcoords = np.bincount(ncoords)
                    
                    # Add to results
                    coordination['N'][ir,:len(bcoords)] += bcoords
                    
                # Set time
                traj_time = traj_time_dcd + Nskip*dt*timewindows[ic + 1]
                
                # Check time
                if traj_time >= eval_maxtime:
                    
                    # Save results
                    pkl.dump(coordination, open(crdnfile, "wb"))
                    
                    # End timer
                    end = time.time()
                    print('System {:s}, {:s} done in {:4.1f} s'.format(
                        temp, mix, end - start))
                    
                    return
                
            # Update total trajectory time
            traj_time_dcd = traj_time
            
        # Save results anyways if time limit is not reached
        pkl.dump(coordination, open(crdnfile, "wb"))
        
        # End timer
        end = time.time()
        print(
            'System {:s}, {:s} does not reach eval_maxtime! - {:.1f}'.format(
                traj_time),
            '\n  Done in {:4.1f} s'.format(
                temp, mix, end - start))
        
        return
        

if tasks==1:
    for i in range(0, len(info_systems)):
        read_sys(i)
else:    
    if __name__ == '__main__':
        pool = Pool(tasks)
        pool.imap(read_sys, range(0, len(info_systems)))
        pool.close()
        pool.join()

#---------------------
# Combine results
#---------------------

for temp in temperatures:
    for mix in mixtures:
        
        comb_crdnfile = os.path.join(
            dirs_maindir, dirs_evaldir, 
            'crdn_{:s}_{:s}_{:s}.npy'.format(str(temp), str(mix), eval_reslab))
        
        comb_coordination = {
            'residue' : eval_reslab,
            'cutoff': np.array(eval_resdst),
            'N': np.zeros([len(eval_resdst), evel_nummax + 1])}
    
        
        if not os.path.exists(comb_crdnfile):
            
            for run in range(Nrun):
                
                crdnfile = os.path.join(
                    dirs_maindir, dirs_evaldir, 
                    'crdn_{:s}_{:s}_{:s}_{:s}.npy'.format(
                        str(temp), str(mix), str(run), eval_reslab))
                    
                coordination = pkl.load(open(crdnfile, "rb"))
                
                for ir, rcut in enumerate(eval_resdst):
                    
                    comb_coordination['N'][ir, :] += coordination['N'][ir, :]
                    
            # Save results
            pkl.dump(comb_coordination, open(comb_crdnfile, "wb"))
    
"""
#---------------------
# Plot comparison
#---------------------

color_scheme = ['b', 'r', 'g', 'purple', 'orange', 'magenta']
label_plt = [
    ['A', 'B', 'C', 'D', 'E', 'F'], 
    ["A'", "B'", "C'", "D'", "E'", "F'"]]

plot_idxdst = [0, 3]

# Get maximum
cmax = 0
crdnN = np.zeros([len(plot_idxdst), len(mixtures), evel_nummax + 1])
for sys in info_systems:
    
    # Data directory
    temp = sys[0]
    mix = sys[1]
    datadir = os.path.join(source, str(temp), str(mix))
    
    # Load cluster file
    crdnfile = os.path.join(
        dirs_maindir, dirs_evaldir, 
        'crdn_{:d}_{:d}_{:s}.npy'.format(temp, mix, eval_reslab))
    
    coordination = pkl.load(open(crdnfile, "rb"))
    
    resdst = coordination['cutoff']
    
    for ir, idxr in enumerate(plot_idxdst):
        
        crdnN[ir, np.where(mixtures==mix)[0][0], :] = (
            coordination['N'][idxr]/np.sum(coordination['N'][idxr]))
        
        cimax = np.max(crdnN[ir, np.where(mixtures==mix)[0][0], :])
        if cimax > cmax:
            cmax = cimax
            
        

# Plot cluster size distribution
figsize = (6, 8)
sfig = float(figsize[0])/float(figsize[1])
fig = plt.figure(figsize=figsize)

# Alignment
left = 0.15
bottom = 0.10
column = np.array([0.65, 0.05])/len(plot_idxdst)
row = np.array([0.75, 0.10])/len(mixtures)

for isys, sys in enumerate(info_systems):
    
    # Data directory
    temp = sys[0]
    mix = sys[1]
    tag = str(temp) + '_' + str(mix)
    Nres = numres[tag][eval_reslab]
    
    for ir, idxr in enumerate(plot_idxdst):
        
        rcut = eval_resdst[idxr]
        
        # Add axis
        axs = fig.add_axes([
            left + ir*np.sum(column), 
            bottom + (len(info_systems) - 1 - isys)*np.sum(row), 
            column[0], row[0]])
        
        # Plot
        ncrdntn = crdnN[ir, np.where(mixtures==mix)[0][0], :21]
        Nmax = len(ncrdntn)
        axs.bar(range(Nmax), ncrdntn, color=color_scheme[isys])
        
        # Plot average
        avg = np.sum(ncrdntn*np.arange(Nmax))
        axs.plot([avg, avg], [0, cmax*1.1], '--k')
        axs.text(
            avg + 0.1*Nmax, 0.85*cmax, r'$\bar{n} = $' + '{:.1f}'.format(avg),
            fontsize=SMALL_SIZE - 2)
        
        axs.set_xlim(-1, Nmax)
        axs.set_ylim(0, cmax*1.1)
        
        if len(plot_idxdst)%2:
            if ir==len(plot_idxdst)//2 and (len(info_systems) - 1 - isys)==0:
                axs.set_xlabel(
                    r'Coordination number $n_\mathrm{{{:s}}}$'.format(
                        eval_pltlab),
                    fontweight=font)
                axs.get_xaxis().set_label_coords(0.5, -0.05*len(info_systems))
        else:
            if ir==len(plot_idxdst)//2 and (len(info_systems) - 1 - isys)==0:
                axs.set_xlabel(
                    r'Coordination number $n_\mathrm{{{:s}}}$'.format(
                        eval_pltlab),
                    fontweight=font)
                axs.get_xaxis().set_label_coords(0.0, -0.07*len(info_systems))
        
        if len(info_systems)%2:
            if isys==len(info_systems)//2 and ir==0:
                axs.set_ylabel(r'$P(n_\mathrm{{{:s}}})$'.format(eval_pltlab))
                axs.get_yaxis().set_label_coords(-0.05*len(resdst), 0.5)
        else:
            if isys==len(info_systems)//2 and ir==0:
                axs.set_ylabel(r'$P(n_\mathrm{{{:s}}})$'.format(eval_pltlab))
                axs.get_yaxis().set_label_coords(-0.05*len(resdst), 1.0)
        
        axs.set_xticks(list(range(Nmax)))
        if (len(info_systems) - 1 - isys)==0:
            xlabel = ['']*Nmax
            for ii in range(0, Nmax + 1, 5):
                xlabel[ii] = '{:d}'.format(ii)
            axs.set_xticklabels(xlabel)
        else:
            axs.set_xticklabels([])
        
        axs.set_yticks([0.0, 0.3, 0.6, 0.9])
        if ir!=0:
            axs.set_yticklabels([])
            
        if isys==0:
            title = (
                r'$r_\mathrm{cut} = $' + '{:.1f}'.format(rcut) 
                + r' $\mathrm{\AA}$')
            axs.set_title(
                title, fontweight=font)
            
            if idxr==plot_idxdst[-1]:
                
                tbox = TextArea('Water\nratio', textprops=dict(
                    color='k', fontsize=MEDIUM_SIZE, ha='center'))

                anchored_tbox = AnchoredOffsetbox(
                    loc='center', child=tbox, pad=0., frameon=False,
                    bbox_to_anchor=(1.45, 1.26),
                    bbox_transform=axs.transAxes, borderpad=0.)

                axs.add_artist(anchored_tbox)
        
        if idxr==plot_idxdst[-1]:
                
            tbox = TextArea('{:d}%\n({:d})'.format(mix, Nres), textprops=dict(
                color='k', fontsize=MEDIUM_SIZE, ha='center'))

            anchored_tbox = AnchoredOffsetbox(
                loc='center', child=tbox, pad=0., frameon=False,
                bbox_to_anchor=(1.45, 0.5),
                bbox_transform=axs.transAxes, borderpad=0.)

            axs.add_artist(anchored_tbox)
        
        tbox = TextArea(label_plt[ir][isys], textprops=dict(
            color='k', fontsize=MEDIUM_SIZE))

        anchored_tbox = AnchoredOffsetbox(
            loc='upper right', child=tbox, pad=0., frameon=False,
            bbox_to_anchor=(0.95, 0.95),
            bbox_transform=axs.transAxes, borderpad=0.)

        axs.add_artist(anchored_tbox)
        
# Save figure
plt.savefig(
    os.path.join(
        dirs_maindir, 'paper_ncoord_{:s}_{:d}.png'.format(eval_reslab, temp)),
    format='png', dpi=dpi)
plt.close()
"""
