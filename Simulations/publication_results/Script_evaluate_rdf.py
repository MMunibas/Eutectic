# Basics
import os
import sys
import time
import numpy as np
from itertools import product, groupby
from glob import glob

# Trajectory reader
import MDAnalysis

# Matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, HPacker, VPacker

# ASE
from ase import Atoms
from ase.io import read, write
from ase.visualize import view

# Multiprocessing
from multiprocessing import Pool

#------------
# Parameters
#------------

# Parallel evaluation runs
tasks = 20

# Source directory
source = '.'

# Temperatures
temperatures = [300]

# Mixtures
mixtures = [0, 10, 20, 30, 40, 50, 60, 80, 70, 90, 95]

# Runs
Nrun = 10

# Trajectory source
traj_dcdtag = 'dyna.*.dcd'
# first: split condition, second: index for irun
traj_dcdinfo = ['.', 1]
traj_crdfile = 'step1_pdbreader.crd'
traj_psffile = 'step1_pdbreader.psf'

# Result directory
#res_maindir = 'results_analysis'
res_maindir = 'results_rdf'
res_evaldir = 'evalfiles'

# Residue of interest
eval_residue = 'SCN'
eval_resids = range(75)

# Regarding residues
eval_resreg = ['SCN', 'ACEM', 'TIP3', 'K']
eval_resnum = {
    'SCN': 3,
    'ACEM': 9,
    'TIP3': 3,
    'K': 1}
eval_ressym = {
    'SCN': ['N', 'C', 'S'],
    'ACEM': ['C', 'C', 'N', 'H', 'H', 'O', 'H', 'H', 'H'],
    'TIP3': ['O', 'H', 'H'],
    'K': ['K']}
eval_reschr = {
    'SCN': [-0.46, -0.36, -0.18],
    'ACEM': [-0.27, 0.55, -0.62, 0.32, 0.30, -0.55, 0.09, 0.09, 0.09],
    'TIP3': [-0.84, 0.42, 0.42],
    'K': [1.00]}
eval_resmss = {
    'SCN': [14.01, 12.01, 32.06],
    'ACEM': [12.01, 12.01, 14.01, 1.01, 1.01, 12.01, 1.01, 1.01, 1.01],
    'TIP3': [14.01, 1.01, 1.01],
    'K': [39.10]}

# Distances from eval_residue atoms to eval_resreg atoms
evaluation = {
    1: {
        'SCN': [0, 1, 2],
        'ACEM': [0, 1, 2, 3, 4, 5, 6, 7, 8],
        'TIP3': [0, 1, 2],
        'K': [0]},
    0: {
        'SCN': [0, 1, 2],
        'ACEM': [0, 1, 2, 3, 4, 5, 6, 7, 8],
        'TIP3': [0, 1, 2],
        'K': [0]},}

#evaluation = {
    #(0,1): {
        #'SCN': [[0,1], 2],
        #'ACEM': [1, 2, 3, 4, 5],
        #'TIP3': [0, 1, 2],
        #'K': [0]}}


# Van der Waals radii to evaluate close distances
eval_vdWradii = {'H': 1.1, 'C': 1.7, 'N': 1.55, 'O': 1.52, 'S': 1.8, 'K': 2.75}

# Time step for averaging in ps
eval_timestep = 1.00

# Maximum time to evaluate in ps
eval_maxtime = 5000.0

# Time step in plots for averaging in ps
plot_timestep = 1.0

# Distance of interest to plot
plot_dint = 5.0


# Plot pattern
plot_pattern = [
    [['K', 0],   ['TIP3', 0], ['ACEM', 0]],
    [['SCN', 0],  ['TIP3', 1], ['ACEM', 1]],
    [['SCN', 1],  ['TIP3', 2], ['ACEM', 2]]]
plot_pattern = np.array(plot_pattern, dtype=object)
plot_residue = r'SCN$^{-}$'
plot_ressym = {
    'SCN': ['N', 'C', 'S', r'SCN$^{-}$'],
    'ACEM': [
        'C', 'C', 'N', r'H$_\mathrm{N}$', r'H$_\mathrm{N}$', 'O', 'H', 'H', 'H',
        'acetamide'],
    'TIP3': ['O', 'H', 'H', r'H$_2$O'],
    'K': [r'K$^{+}$', r'K$^{+}$']}


# Fontsize
SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 18

#plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
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
"""
# Get atoms and pair information
numres = {}
atomsint = {}

# Iterate over systems
for sys in info_systems:
    
    # Data directory
    temp = str(sys[0])
    mix = str(sys[1])
    datadir = os.path.join(source, temp, mix + "_0")
    
    # System tag
    tag = temp + '_' + mix
    
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
        numres[tag][res] = int(len(info_res)/eval_resnum[res])
        
    # Get residue atom numbers
    atomsint[tag] = {}
    for ires, res in enumerate(all_residues):
        atomsinfo = np.zeros(
            [numres[tag][res], eval_resnum[res]], dtype=int)
        for ir in range(numres[tag][res]):
            ri = eval_resnum[res]*ir
            for ia in range(eval_resnum[res]):
                info = listres[res][ri + ia].split()
                atomsinfo[ir, ia] = int(info[0]) - 1
        atomsint[tag][res] = atomsinfo
        
# Make result directory
if not os.path.exists(res_maindir):
    os.mkdir(res_maindir)
    
if not os.path.exists(os.path.join(res_maindir, res_evaldir)):
    os.mkdir(os.path.join(res_maindir, res_evaldir))
    


#---------------------
# Collect system data
#---------------------

# Iterate over systems and resids
info_systems_resids = np.array(
    list(product(temperatures, mixtures, range(Nrun), eval_resids)))

def read_sys(i):
    
    # Begin timer
    start = time.time()
    
    # Data directory
    temp = str(info_systems_resids[i][0])
    mix = str(info_systems_resids[i][1])
    run = str(info_systems_resids[i][2])
    resid = info_systems_resids[i][3]
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
    
    # Sort dcd files
    dcdsort = np.argsort(iruns)
    dcdfiles = dcdfiles[dcdsort]
    iruns = iruns[dcdsort]
    
    # Distance file
    distfile = os.path.join(
        res_maindir, res_evaldir, 
        'dists_{:s}_{:s}_{:s}_{:s}_{:d}.npy'.format(
            temp, mix, run, eval_residue, resid))
    
    if not os.path.exists(distfile):
        
        # Initialize distance dictionary
        # Final build: 
        # Residue atom, Target residue type, ...
        # ... target residue (all), target residue atom, time
        eval_dists = {}
        for (resatomi, item) in evaluation.items():
            if isinstance(resatomi, tuple):
                key_i = 'COM' + (len(resatomi)*'_{:d}').format(*resatomi)
            else:
                key_i = resatomi
            eval_dists[key_i] = {}
            for (resj, resatomsj) in item.items():
                eval_dists[key_i][resj] = []
        
        # Initialize trajectory time counter in ps
        traj_time_dcd = 0.0
        
        # Iterate over dcd files
        for idcd, dcdfile in enumerate(dcdfiles):
            print(temp, mix, idcd)
            # Open dcd file
            dcd = MDAnalysis.Universe(psffile, dcdfile)
            
            # Get trajectory parameter
            Nframes = len(dcd.trajectory)
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
            
            # Iterate over centre frames
            for ic, tc in enumerate(timecenters):
                
                # Get positions
                positions = dcd.trajectory[tc]._pos
                
                # Get cell information
                cell = dcd.trajectory[tc]._unitcell
                
                # Residue positions
                eval_posres = positions[atomsint[tag][eval_residue][resid]]
                
                # Iterate over defined eval_residue atoms
                for ires, (resatomi, item) in enumerate(evaluation.items()):
                    
                    if isinstance(resatomi, tuple):
                        key_i = 'COM' + (len(resatomi)*'_{:d}').format(
                            *resatomi)
                        totmass = sum(
                            [eval_resmss[eval_residue][ai] for ai in resatomi])
                        eval_posres_i = sum([
                            eval_posres[ai]*eval_resmss[eval_residue][ai] 
                            for ai in resatomi])/totmass
                    else:
                        key_i = resatomi
                        eval_posres_i = eval_posres[key_i]
                        
                    # Iterate over defined target residue atoms
                    for jres, (resj, resatomsj) in enumerate(item.items()):
                        
                        # Initiate distance arrays
                        distances = np.zeros(
                            [numres[tag][resj], len(resatomsj)], 
                            dtype=np.float32)
                        
                        # Iterate over number of target residues
                        for nresj in range(numres[tag][resj]):
                            
                            # Get residue position
                            posres_j = positions[atomsint[tag][resj][nresj]]
                            
                            # Get atom distances to evaluation atoms
                            for ja, aj in enumerate(resatomsj):
                                
                                if isinstance(aj, list):
                                    totmass = sum([
                                        eval_resmss[resj][ai]
                                        for ai in aj])
                                    pos_j = sum([
                                        posres_j[ai]*eval_resmss[resj][ai] 
                                        for ai in aj])/totmass
                                else:
                                    pos_j = posres_j[aj]
                                    
                                # Check periodic boundary conditions
                                d = pos_j - eval_posres_i
                                shift = np.zeros_like(pos_j)
                                for i in range(3):
                                    L = cell[i]
                                    shift[i] = (
                                        (d[i] + L/2.) % L - L/2. - d[i])
                                
                                # Apply periodic boundary shift
                                pos_j += shift
                                
                                # Get atom distances
                                r = np.linalg.norm(
                                    eval_posres_i - pos_j)
                                distances[nresj, ja] = r
                        
                        eval_dists[key_i][resj].append(distances)
                        
                # Set time
                traj_time = traj_time_dcd + Nskip*dt*timewindows[ic + 1]
                
                # Check time
                if traj_time >= eval_maxtime:
                    
                    # List to array
                    for (resatomi, item) in evaluation.items():
                        if isinstance(resatomi, tuple):
                            key_i = 'COM' + (len(resatomi)*'_{:d}').format(
                                *resatomi)
                        else:
                            key_i = resatomi
                        for (resj, resatomsj) in item.items():
                            eval_dists[key_i][resj] = np.array(
                                eval_dists[key_i][resj])
                    
                    # Save result file of frames
                    np.save(distfile, eval_dists, allow_pickle=True)
                    
                    # End timer
                    end = time.time()
                    print('System {:s}, {:s}, {:d} done in {:4.1f} s'.format(
                        temp, mix, resid, end - start))
                    
                    return
                
            traj_time_dcd = traj_time
                
    else:
        
        print('System {:s}, {:s}, {:d} already done'.format(
            temp, mix, resid))
        return
    
#for isys, sysi in enumerate(info_systems_resids):
    #read_sys(0)


if tasks==1:
    for i in range(0, len(info_systems_resids)):
        read_sys(i)
else:    
    if __name__ == '__main__':
        pool = Pool(tasks)
        pool.imap(read_sys, range(0, len(info_systems_resids)))
        pool.close()
        pool.join()
"""
#----------------------------
# Paper Plot g(r) vs. mix V
#----------------------------

# Fontsize
SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 18

#plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

dpi = 300

info_systems = np.array(
    list(product(temperatures, mixtures)))

rad_lim = (1.00, 7.00)
rad_bins = np.linspace(rad_lim[0], rad_lim[1], num=121)
rad_dist = rad_bins[1] - rad_bins[0]
rad_cent = rad_bins[:-1] + rad_dist/2.

for isys, sysi in enumerate(info_systems):
    
    # Data directory
    temp = str(sysi[0])
    mix = str(sysi[1])
    datadir = os.path.join(source, temp, mix)
    
    gfile = os.path.join(
        res_maindir, 'g_histogramm_data_{:s}_{:s}.npy'.format(temp, mix))
    
    if not os.path.exists(gfile):
        
        # Coordination number histogram
        # Residue, Residue atom, radial distances
        n_hist = {}
        g_hist = {}
        #for resj, resatomsj in evaluation[(0,1)].items():
        for resj, resatomsj in evaluation[1].items():
            
            n_hist[resj] = {}
            g_hist[resj] = {}
            for atomj in resatomsj:
                if isinstance(atomj, list):
                    keyj = '{:d}_{:d}'.format(*atomj)
                else:
                    keyj = atomj
                n_hist[resj][keyj] = np.zeros(
                    [rad_cent.shape[0]],
                    dtype=np.float)
                g_hist[resj][keyj] = np.zeros(
                    [rad_cent.shape[0]],
                    dtype=np.float)
                
        for irun in range(Nrun):
        
            for resid in eval_resids:
                print(mix, resid)
                # Distance file
                distfile = os.path.join(
                    res_maindir, res_evaldir, 
                    'dists_{:s}_{:s}_{:s}_{:s}_{:d}.npy'.format(
                        temp, mix, str(irun), eval_residue, resid))
                
                # Load data
                # Residue atom, Target residue type, ...
                # ... time, target residue (all),  target residue atom
                distsdata = np.load(distfile, allow_pickle=True).item(0)
                
                # Iterate over residue types
                #for resj, resatomsj in evaluation[(0,1)].items():
                for resj, resatomsj in evaluation[1].items():
                    
                    # Get distances
                    #dists = distsdata['COM_0_1'][resj]
                    dists = distsdata[1][resj]
                    
                    for ja, atomj in enumerate(resatomsj):
                        
                        if isinstance(atomj, list):
                            keyj = '{:d}_{:d}'.format(*atomj)
                        else:
                            keyj = atomj
                        
                        distslist = dists[:,:,ja].reshape(-1)
                        
                        Ntime = float(dists.shape[0])
                        Nresi = float(len(eval_resids))
                        nr_hist = np.histogram(
                            distslist[distslist > 0.0], bins=rad_bins)[0]/Ntime
                        
                        for ir, rc in enumerate(rad_cent):
                            
                            n_hist[resj][keyj][ir] += np.sum(nr_hist[:ir])/Nresi
                        
                        g_hist[resj][keyj] += nr_hist
                        
            np.save(gfile, g_hist, allow_pickle=True)
        
    else:
        
        g_hist = np.load(gfile, allow_pickle=True).item(0)

# Plot options

# Figure
figsize = (12, 6)
sfig = float(figsize[0])/float(figsize[1])
fig = plt.figure(figsize=figsize)

# Alignment
left = 0.10
bottom = 0.15
column = [0.38, 0.10]
row = [0.35, 0.03]

line_scheme = [
    'solid', 'dotted', 'dashed', 'dashdot', (0, (3, 1, 3, 1, 1, 1)), 
    (0, (3, 1, 1, 1, 1, 1))]

color_scheme = ['b', 'r', 'g', 'purple', 'orange', 'magenta']

legl = [
    '0%',
    '20%',
    '50%',
    '80%',
    '90%',
    '95%']

# Add axis
axs1 = fig.add_axes([
    left + 0*sum(column), bottom + 1*sum(row), column[0], row[0]])
axs2 = fig.add_axes([
    left + 0*sum(column), bottom + 0*sum(row), column[0], row[0]])
axs3 = fig.add_axes([
    left + 1*sum(column), bottom + 1*sum(row), column[0], row[0]])
axs4 = fig.add_axes([
    left + 1*sum(column), bottom + 0*sum(row), column[0], row[0]])

info_systems = np.array(
    list(product(temperatures, [0, 20, 50, 80, 90, 95])))

for isys, sysi in enumerate(info_systems):
        
    # Data directory
    temp = str(sysi[0])
    mix = str(sysi[1])
    datadir = os.path.join(source, temp, mix)
    
    gfile = os.path.join(
        res_maindir, 'g_histogramm_data_{:s}_{:s}.npy'.format(temp, mix))
    
    g_hist = np.load(gfile, allow_pickle=True).item(0)
    
    # Normalize radial distribution
    g_hist1 = g_hist['SCN'][1]
    g_hist2 = g_hist['K'][0]
    g_hist3 = g_hist['TIP3'][0]
    g_hist4 = g_hist['ACEM'][2]
    
    V = 4./3.*np.pi*rad_lim[1]**3
    
    N = np.sum(g_hist1)
    print(temp, mix, N)
    if N!=0:
        g_hist1 = V*g_hist1/rad_dist/N
        g_hist1 = g_hist1/(4.0*np.pi*rad_cent**2)
    else:
        g_hist1 = np.zeros_like(g_hist1)
        
    N = np.sum(g_hist2)
    if N!=0:
        g_hist2 = V*g_hist2/rad_dist/N
        g_hist2 = g_hist2/(4.0*np.pi*rad_cent**2)
    else:
        np.zeros_like(g_hist2)
    
    N = np.sum(g_hist3)
    if N!=0:
        g_hist3 = g_hist3/rad_dist/Nrun#*V/N
        g_hist3 = g_hist3/(4.0*np.pi*rad_cent**2)
    else:
        np.zeros_like(g_hist3)
    
    N = np.sum(g_hist4)
    if N!=0:
        g_hist4 = g_hist4/rad_dist/Nrun#*V/N
        g_hist4 = g_hist4/(4.0*np.pi*rad_cent**2)
    else:
        np.zeros_like(g_hist4)
    
    axs1.plot(
        rad_cent, g_hist1,
        color=color_scheme[isys],
        linestyle=line_scheme[isys],
        label=legl[isys],
        lw=2)
    
    axs2.plot(
        rad_cent, g_hist2,
        color=color_scheme[isys],
        linestyle=line_scheme[isys],
        label=legl[isys],
        lw=2)
    
    axs3.plot(
        rad_cent, g_hist3,
        color=color_scheme[isys],
        linestyle=line_scheme[isys],
        label=legl[isys],
        lw=2)
    
    axs4.plot(
        rad_cent, g_hist4,
        color=color_scheme[isys],
        linestyle=line_scheme[isys],
        label=legl[isys],
        lw=2)
    
    if isys==0:
            
        axs2.set_xlabel(r'Radius $r$ ($\mathrm{\AA}$)')
        axs2.get_xaxis().set_label_coords(0.50, -0.20)
        axs4.set_xlabel(r'Radius $r$ ($\mathrm{\AA}$)')
        axs4.get_xaxis().set_label_coords(0.50, -0.20)
        
        axs2.set_ylabel(r'$g(r)$')
        axs2.get_yaxis().set_label_coords(-0.12, 1.00)
        axs3.set_ylabel(r'$g(r) \cdot \rho(\mathrm{H_2O})$')
        axs3.get_yaxis().set_label_coords(-0.12, 0.50)
        axs4.set_ylabel(r'$g(r) \cdot \rho(\mathrm{acetamide})$')
        axs4.get_yaxis().set_label_coords(-0.12, 0.50)
        
        axs1.set_xlim(rad_lim)
        axs2.set_xlim(rad_lim)
        axs3.set_xlim(rad_lim)
        axs4.set_xlim(rad_lim)
        
        axs1.set_ylim([-4/20., 4 + 4/20])
        axs2.set_ylim([-11/20., 11 + 11/20])
        axs3.set_ylim([-7/20., 7 + 7/20])
        axs4.set_ylim([-2/20., 2 + 2/20])
        
        axs1.set_xticklabels([])
        axs3.set_xticklabels([])
        
        axs1.set_yticks([0, 1, 2, 3, 4])
        axs2.set_yticks([0, 2, 4, 6, 8, 10])
        axs3.set_yticks([0, 2, 4, 6])
        axs4.set_yticks([0, 1, 2])
        
        tbox = TextArea(
            'A', 
            textprops=dict(color='k', fontsize=18))
    
        anchored_tbox = AnchoredOffsetbox(
            loc='lower left', child=tbox, pad=0., frameon=False,
            bbox_to_anchor=(0.02, 0.84),
            bbox_transform=axs1.transAxes, borderpad=0.)
        
        axs1.add_artist(anchored_tbox)
        
        tbox = TextArea(
            r'C$_\mathrm{SCN^-}-$C$_\mathrm{SCN^-}$', 
            textprops=dict(color='k', fontsize=MEDIUM_SIZE))
    
        anchored_tbox = AnchoredOffsetbox(
            loc='lower left', child=tbox, pad=0., frameon=False,
            bbox_to_anchor=(0.02, 0.64),
            bbox_transform=axs1.transAxes, borderpad=0.)
        
        axs1.add_artist(anchored_tbox)
        
        tbox = TextArea(
            'B', 
            textprops=dict(color='k', fontsize=18))
    
        anchored_tbox = AnchoredOffsetbox(
            loc='lower left', child=tbox, pad=0., frameon=False,
            bbox_to_anchor=(0.02, 0.84),
            bbox_transform=axs2.transAxes, borderpad=0.)
        
        axs2.add_artist(anchored_tbox)
        
        tbox = TextArea(
            r'C$_\mathrm{SCN^-}-$K$^\mathrm{+}$', 
            textprops=dict(color='k', fontsize=MEDIUM_SIZE))
    
        anchored_tbox = AnchoredOffsetbox(
            loc='lower left', child=tbox, pad=0., frameon=False,
            bbox_to_anchor=(0.02, 0.64),
            bbox_transform=axs2.transAxes, borderpad=0.)
        
        axs2.add_artist(anchored_tbox)
        
        tbox = TextArea(
            'C', 
            textprops=dict(color='k', fontsize=18, ha='left'))
    
        anchored_tbox = AnchoredOffsetbox(
            loc='lower left', child=tbox, pad=0., frameon=False,
            bbox_to_anchor=(0.02, 0.84),
            bbox_transform=axs3.transAxes, borderpad=0.)
        
        axs3.add_artist(anchored_tbox)
        
        tbox = TextArea(
            '\n' + r'C$_\mathrm{SCN^-}-$O$_\mathrm{H_2O}$', 
            textprops=dict(color='k', fontsize=MEDIUM_SIZE, ha='right'))
    
        anchored_tbox = AnchoredOffsetbox(
            loc='lower left', child=tbox, pad=0., frameon=False,
            bbox_to_anchor=(0.98, 0.84),
            bbox_transform=axs3.transAxes, borderpad=0.)
        
        axs3.add_artist(anchored_tbox)
        
        tbox = TextArea(
            'D', 
            textprops=dict(color='k', fontsize=18, ha='left'))
    
        anchored_tbox = AnchoredOffsetbox(
            loc='lower left', child=tbox, pad=0., frameon=False,
            bbox_to_anchor=(0.02, 0.84),
            bbox_transform=axs4.transAxes, borderpad=0.)
        
        axs4.add_artist(anchored_tbox)
        
        tbox = TextArea(
            '\n' + r'C$_\mathrm{SCN^-}-$N$_\mathrm{acetamide}$', 
            textprops=dict(color='k', fontsize=MEDIUM_SIZE, ha='right'))
    
        anchored_tbox = AnchoredOffsetbox(
            loc='lower left', child=tbox, pad=0., frameon=False,
            bbox_to_anchor=(0.98, 0.84),
            bbox_transform=axs4.transAxes, borderpad=0.)
        
        axs4.add_artist(anchored_tbox)
    
    
    if isys==(len(info_systems)-1):
        axs2.legend(
            loc=[0.72, 0.22], ncol=1, framealpha=1.0, fontsize=SMALL_SIZE-2)
    
#plt.show()

plt.savefig(
    os.path.join(
        res_maindir, 'paper_gdist_300_V.png'),
    format='png', dpi=dpi)

            
            
            
            
            
            
            
    











#----------------------------
# Paper Plot g(r) vs. mix VI
#----------------------------

# Fontsize
SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 18

#plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

dpi = 300

info_systems = np.array(
    list(product(temperatures, mixtures)))

rad_lim = (1.00, 7.00)
rad_bins = np.linspace(rad_lim[0], rad_lim[1], num=121)
rad_dist = rad_bins[1] - rad_bins[0]
rad_cent = rad_bins[:-1] + rad_dist/2.

for isys, sysi in enumerate(info_systems):
    
    # Data directory
    temp = str(sysi[0])
    mix = str(sysi[1])
    datadir = os.path.join(source, temp, mix)
    
    gfile = os.path.join(
        res_maindir, 'g_histogramm_data_{:s}_{:s}.npy'.format(temp, mix))
    
    if not os.path.exists(gfile):
        
        # Coordination number histogram
        # Residue, Residue atom, radial distances
        n_hist = {}
        g_hist = {}
        #for resj, resatomsj in evaluation[(0,1)].items():
        for resj, resatomsj in evaluation[1].items():
            
            n_hist[resj] = {}
            g_hist[resj] = {}
            for atomj in resatomsj:
                if isinstance(atomj, list):
                    keyj = '{:d}_{:d}'.format(*atomj)
                else:
                    keyj = atomj
                n_hist[resj][keyj] = np.zeros(
                    [rad_cent.shape[0]],
                    dtype=np.float)
                g_hist[resj][keyj] = np.zeros(
                    [rad_cent.shape[0]],
                    dtype=np.float)
                
        for irun in range(Nrun):
        
            for resid in eval_resids:
                print(mix, resid)
                # Distance file
                distfile = os.path.join(
                    res_maindir, res_evaldir, 
                    'dists_{:s}_{:s}_{:s}_{:s}_{:d}.npy'.format(
                        temp, mix, str(irun), eval_residue, resid))
                
                # Load data
                # Residue atom, Target residue type, ...
                # ... time, target residue (all),  target residue atom
                distsdata = np.load(distfile, allow_pickle=True).item(0)
                
                # Iterate over residue types
                #for resj, resatomsj in evaluation[(0,1)].items():
                for resj, resatomsj in evaluation[1].items():
                    
                    # Get distances
                    #dists = distsdata['COM_0_1'][resj]
                    dists = distsdata[1][resj]
                    
                    for ja, atomj in enumerate(resatomsj):
                        
                        if isinstance(atomj, list):
                            keyj = '{:d}_{:d}'.format(*atomj)
                        else:
                            keyj = atomj
                        
                        distslist = dists[:,:,ja].reshape(-1)
                        
                        Ntime = float(dists.shape[0])
                        Nresi = float(len(eval_resids))
                        nr_hist = np.histogram(
                            distslist[distslist > 0.0], bins=rad_bins)[0]/Ntime
                        
                        for ir, rc in enumerate(rad_cent):
                            
                            n_hist[resj][keyj][ir] += np.sum(nr_hist[:ir])/Nresi
                        
                        g_hist[resj][keyj] += nr_hist
                        
            np.save(gfile, g_hist, allow_pickle=True)
        
    else:
        
        g_hist = np.load(gfile, allow_pickle=True).item(0)

# Plot options

# Figure
figsize = (12, 6)
sfig = float(figsize[0])/float(figsize[1])
fig = plt.figure(figsize=figsize)

# Alignment
left = 0.10
bottom = 0.15
column = [0.38, 0.10]
row = [0.35, 0.03]

line_scheme = [
    'solid', 'dotted', 'dashed', 'dashdot', (0, (3, 1, 3, 1, 1, 1)), 
    (0, (3, 1, 1, 1, 1, 1))]

color_scheme = ['b', 'r', 'g', 'purple', 'orange', 'magenta']

# Add axis
axs1 = fig.add_axes([
    left + 0*sum(column), bottom + 1*sum(row), column[0], row[0]])
axs2 = fig.add_axes([
    left + 0*sum(column), bottom + 0*sum(row), column[0], row[0]])
axs3 = fig.add_axes([
    left + 1*sum(column), bottom + 1*sum(row), column[0], row[0]])
axs4 = fig.add_axes([
    left + 1*sum(column), bottom + 0*sum(row), column[0], row[0]])

for isys, sysi in enumerate(info_systems):
        
    # Data directory
    temp = str(sysi[0])
    mix = str(sysi[1])
    datadir = os.path.join(source, temp, mix)
    
    gfile = os.path.join(
        res_maindir, 'g_histogramm_data_{:s}_{:s}.npy'.format(temp, mix))
    
    g_hist = np.load(gfile, allow_pickle=True).item(0)
    
    # Normalize radial distribution
    g_hist1 = g_hist['SCN'][1]
    g_hist2 = g_hist['K'][0]
    g_hist3 = g_hist['TIP3'][0]
    g_hist4 = g_hist['ACEM'][2]
    
    V = 4./3.*np.pi*rad_lim[1]**3
    
    N = np.sum(g_hist1)
    print(temp, mix, N)
    if N!=0:
        g_hist1 = V*g_hist1/rad_dist/N
        g_hist1 = g_hist1/(4.0*np.pi*rad_cent**2)
    else:
        g_hist1 = np.zeros_like(g_hist1)
        
    N = np.sum(g_hist2)
    if N!=0:
        g_hist2 = V*g_hist2/rad_dist/N
        g_hist2 = g_hist2/(4.0*np.pi*rad_cent**2)
    else:
        np.zeros_like(g_hist2)
    
    N = np.sum(g_hist3)
    if N!=0:
        g_hist3 = g_hist3/rad_dist/Nrun#*V/N
        g_hist3 = g_hist3/(4.0*np.pi*rad_cent**2)
    else:
        np.zeros_like(g_hist3)
    
    N = np.sum(g_hist4)
    if N!=0:
        g_hist4 = g_hist4/rad_dist/Nrun#*V/N
        g_hist4 = g_hist4/(4.0*np.pi*rad_cent**2)
    else:
        np.zeros_like(g_hist4)
    
    legl = '{:s}%'.format(mix)
    axs1.plot(
        rad_cent, g_hist1,
        label=legl,
        lw=2)
    
    axs2.plot(
        rad_cent, g_hist2,
        label=legl,
        lw=2)
    
    axs3.plot(
        rad_cent, g_hist3,
        label=legl,
        lw=2)
    
    axs4.plot(
        rad_cent, g_hist4,
        label=legl,
        lw=2)
    
    if isys==0:
            
        axs2.set_xlabel(r'Radius $r$ ($\mathrm{\AA}$)')
        axs2.get_xaxis().set_label_coords(0.50, -0.20)
        axs4.set_xlabel(r'Radius $r$ ($\mathrm{\AA}$)')
        axs4.get_xaxis().set_label_coords(0.50, -0.20)
        
        axs2.set_ylabel(r'$g(r)$')
        axs2.get_yaxis().set_label_coords(-0.12, 1.00)
        axs3.set_ylabel(r'$g(r) \cdot \rho(\mathrm{H_2O})$')
        axs3.get_yaxis().set_label_coords(-0.12, 0.50)
        axs4.set_ylabel(r'$g(r) \cdot \rho(\mathrm{acetamide})$')
        axs4.get_yaxis().set_label_coords(-0.12, 0.50)
        
        axs1.set_xlim(rad_lim)
        axs2.set_xlim(rad_lim)
        axs3.set_xlim(rad_lim)
        axs4.set_xlim(rad_lim)
        
        axs1.set_ylim([-4/20., 4 + 4/20])
        axs2.set_ylim([-11/20., 11 + 11/20])
        axs3.set_ylim([-7/20., 7 + 7/20])
        axs4.set_ylim([-2/20., 2 + 2/20])
        
        axs1.set_xticklabels([])
        axs3.set_xticklabels([])
        
        axs1.set_yticks([0, 1, 2, 3, 4])
        axs2.set_yticks([0, 2, 4, 6, 8, 10])
        axs3.set_yticks([0, 2, 4, 6])
        axs4.set_yticks([0, 1, 2])
        
        tbox = TextArea(
            'A', 
            textprops=dict(color='k', fontsize=18))
    
        anchored_tbox = AnchoredOffsetbox(
            loc='lower left', child=tbox, pad=0., frameon=False,
            bbox_to_anchor=(0.02, 0.84),
            bbox_transform=axs1.transAxes, borderpad=0.)
        
        axs1.add_artist(anchored_tbox)
        
        tbox = TextArea(
            r'C$_\mathrm{SCN^-}-$C$_\mathrm{SCN^-}$', 
            textprops=dict(color='k', fontsize=MEDIUM_SIZE))
    
        anchored_tbox = AnchoredOffsetbox(
            loc='lower left', child=tbox, pad=0., frameon=False,
            bbox_to_anchor=(0.02, 0.64),
            bbox_transform=axs1.transAxes, borderpad=0.)
        
        axs1.add_artist(anchored_tbox)
        
        tbox = TextArea(
            'B', 
            textprops=dict(color='k', fontsize=18))
    
        anchored_tbox = AnchoredOffsetbox(
            loc='lower left', child=tbox, pad=0., frameon=False,
            bbox_to_anchor=(0.02, 0.84),
            bbox_transform=axs2.transAxes, borderpad=0.)
        
        axs2.add_artist(anchored_tbox)
        
        tbox = TextArea(
            r'C$_\mathrm{SCN^-}-$K$^\mathrm{+}$', 
            textprops=dict(color='k', fontsize=MEDIUM_SIZE))
    
        anchored_tbox = AnchoredOffsetbox(
            loc='lower left', child=tbox, pad=0., frameon=False,
            bbox_to_anchor=(0.02, 0.64),
            bbox_transform=axs2.transAxes, borderpad=0.)
        
        axs2.add_artist(anchored_tbox)
        
        tbox = TextArea(
            'C', 
            textprops=dict(color='k', fontsize=18, ha='left'))
    
        anchored_tbox = AnchoredOffsetbox(
            loc='lower left', child=tbox, pad=0., frameon=False,
            bbox_to_anchor=(0.02, 0.84),
            bbox_transform=axs3.transAxes, borderpad=0.)
        
        axs3.add_artist(anchored_tbox)
        
        tbox = TextArea(
            '\n' + r'C$_\mathrm{SCN^-}-$O$_\mathrm{H_2O}$', 
            textprops=dict(color='k', fontsize=MEDIUM_SIZE, ha='right'))
    
        anchored_tbox = AnchoredOffsetbox(
            loc='lower left', child=tbox, pad=0., frameon=False,
            bbox_to_anchor=(0.98, 0.84),
            bbox_transform=axs3.transAxes, borderpad=0.)
        
        axs3.add_artist(anchored_tbox)
        
        tbox = TextArea(
            'D', 
            textprops=dict(color='k', fontsize=18, ha='left'))
    
        anchored_tbox = AnchoredOffsetbox(
            loc='lower left', child=tbox, pad=0., frameon=False,
            bbox_to_anchor=(0.02, 0.84),
            bbox_transform=axs4.transAxes, borderpad=0.)
        
        axs4.add_artist(anchored_tbox)
        
        tbox = TextArea(
            '\n' + r'C$_\mathrm{SCN^-}-$N$_\mathrm{acetamide}$', 
            textprops=dict(color='k', fontsize=MEDIUM_SIZE, ha='right'))
    
        anchored_tbox = AnchoredOffsetbox(
            loc='lower left', child=tbox, pad=0., frameon=False,
            bbox_to_anchor=(0.98, 0.84),
            bbox_transform=axs4.transAxes, borderpad=0.)
        
        axs4.add_artist(anchored_tbox)
    
    
    if isys==(len(info_systems)-1):
        axs2.legend(
            loc=[0.72, 0.22], ncol=1, framealpha=1.0, fontsize=SMALL_SIZE-2)
    
#plt.show()

plt.savefig(
    os.path.join(
        res_maindir, 'paper_gdist_300_VI.png'),
    format='png', dpi=dpi)

            
            

