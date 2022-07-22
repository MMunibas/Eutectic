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

# Miscellaneous
from scipy import ndimage as ndi
from scipy.optimize import curve_fit

#------------
# Parameters
#------------

# Source directory
source = '.'

# Temperatures
temperatures = [300]

# Mixtures
mixtures = [0, 20, 30, 50, 80, 90, 95]
#mixtures = [80, 90, 100]
#mixtures = [0, 10, 20]
#mixtures = [30, 40, 50]
#mixtures = [60, 95]

# Trajectory source
traj_dcdtag = 'dyna.*.dcd'
# first: split condition, second: index for irun
traj_dcdinfo = ['.', 1]
traj_crdfile = 'step1_pdbreader.crd'
traj_psffile = 'step1_pdbreader.psf'

# Result directory
res_maindir = 'results_radang'
res_evaldir = 'evalfiles'

# Parallel evaluation runs
tasks = 20

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
    (1, 0): {
        'SCN': [(1, 0)]}
    }
    
# Time step for averaging in ps
eval_timestep = 1.00

# Maximum time to evaluate in ps
eval_maxtime = 50000.0

# Time step in plots for averaging in ps
plot_timestep = 0.10

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
    list(product(temperatures, mixtures, eval_resids)))

def read_sys(i):
    
    # Begin timer
    start = time.time()
    
    # Data directory
    temp = str(info_systems_resids[i][0])
    mix = str(info_systems_resids[i][1])
    resid = info_systems_resids[i][2]
    datadir = os.path.join(source, temp, mix + "_0")
    gdatdir = os.path.join(source, temp, mix + "_*")
    
    # System tag
    tag = temp + '_' + mix
    
    # Read dcd files and get atom distances
    #---------------------------------------
    
    # Get dcd files
    dcdfiles = np.array(glob(os.path.join(gdatdir, traj_dcdtag)))
    iruns = np.array([
        int(dcdfile.split('/')[-1].split(traj_dcdinfo[0])[traj_dcdinfo[1]])
        for dcdfile in dcdfiles])
    psffile = os.path.join(datadir, traj_psffile)
    
    # Sort dcd files
    dcdsort = np.argsort(iruns)
    dcdfiles = dcdfiles[dcdsort]
    iruns = iruns[dcdsort]
    
    # Distance-Angular file
    rdngfile = os.path.join(
        res_maindir, res_evaldir, 
        'radang_{:s}_{:s}_{:s}_{:d}.npy'.format(temp, mix, eval_residue, resid))
    
    if not os.path.exists(rdngfile):
        
        # Initialize radial angular dictionary
        # Final build: 
        # Residue vector, Target residue type, ...
        # ... target residue vectors, target residue atom, time
        eval_rdngs = {}
        for (ivector, item) in evaluation.items():
            eval_rdngs[ivector] = {}
            for (resj, jvector) in item.items():
                eval_rdngs[ivector][resj] = []
        
        # Initialize trajectory time counter in ps
        traj_time_dcd = 0.0
        
        # Iterate over dcd files
        for idcd, dcdfile in enumerate(dcdfiles):
            
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
                ipos = positions[atomsint[tag][eval_residue][resid]]
                
                # Iterate over defined eval_residue atoms
                for ires, (ivector, item) in enumerate(evaluation.items()):
                    
                    ipos_a = ipos[ivector[0]]
                    ipos_b = ipos[ivector[1]]
                    
                    # Iterate over defined target residue atoms
                    for jres, (resj, jvectors) in enumerate(item.items()):
                        
                        # Initiate distance arrays
                        distances = np.zeros(
                            [numres[tag][resj], len(jvectors)], 
                            dtype=np.float32)
                        angles = np.zeros(
                            [numres[tag][resj], len(jvectors)], 
                            dtype=np.float32)
                        orientation = np.zeros(
                            [numres[tag][resj], len(jvectors)], 
                            dtype=np.float32)
                        
                        # Iterate over number of target residues
                        for nresj in range(numres[tag][resj]):
                            
                            # Get residue position
                            jpos = positions[atomsint[tag][resj][nresj]]
                            
                            # Get atom distances to evaluation atoms
                            for jv, jvector in enumerate(jvectors):
                                
                                jpos_a = jpos[jvector[0]]
                                jpos_b = jpos[jvector[1]]
                                
                                # Check periodic boundary conditions
                                d = jpos_a - ipos_a
                                shift = np.zeros_like(jpos_a)
                                for i in range(3):
                                    L = cell[i]
                                    shift[i] = (
                                        (d[i] + L/2.) % L - L/2. - d[i])
                                
                                # Apply periodic boundary shift
                                jpos_a += shift
                                jpos_b += shift
                                
                                # Get atom-atom distances
                                r = np.linalg.norm(
                                    jpos_a - ipos_a)
                                distances[nresj, jv] = r
                                
                                # Get vector-atom angle
                                vi = ipos_b - ipos_a
                                vj = jpos_a - ipos_a
                                ri = np.linalg.norm(vi)
                                rj = np.linalg.norm(vj)
                                if rj!=0.0:
                                    cosa = np.dot(vi/ri, vj/rj)
                                    if cosa >= 1.0:
                                        cosa = 1.0
                                    if cosa <= -1.0:
                                        cosa = -1.0
                                    a = np.arccos(cosa)
                                    angles[nresj, jv] = a
                                else:
                                    angles[nresj, jv] = 0.0
                                
                                
                                # Get vector-vector angle
                                vi = ipos_b - ipos_a
                                vj = jpos_b - jpos_a
                                ri = np.linalg.norm(vi)
                                rj = np.linalg.norm(vj)
                                coso = np.dot(vi/ri, vj/rj)
                                if coso >= 1.0:
                                    coso = 1.0
                                if coso <= -1.0:
                                    coso = -1.0
                                o = np.arccos(coso)
                                orientation[nresj, jv] = o
                                
                        rdngs = np.stack([distances, angles, orientation]).T
                        eval_rdngs[ivector][resj].append(rdngs)
                        
                # Set time
                traj_time = traj_time_dcd + Nskip*dt*timewindows[ic + 1]
                
                # Check time
                if traj_time >= eval_maxtime:
                    
                    # List to array
                    for (ivector, item) in evaluation.items():
                        for (resj, jvector) in item.items():
                            eval_rdngs[ivector][resj] = np.array(
                                eval_rdngs[ivector][resj])
                            
                    eval_rdngs = np.array(eval_rdngs)
                    
                    # Save result file of frames
                    np.save(rdngfile, eval_rdngs, allow_pickle=True)
                    
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

#--------------------------------
# Evaluate radial angular distr.
#--------------------------------


# Iterate over systems and resids
smix = [0, 30, 50, 80, 95]
#smix = [100]

info_systems = np.array(
    list(product(temperatures, smix)))

rad_cut = 8.0

rad_lim = (2.00, rad_cut)
rad_N = 60
rad_bins = np.linspace(rad_lim[0], rad_lim[1], num=rad_N + 1)
rad_dist = rad_bins[1] - rad_bins[0]
rad_cent = rad_bins[:-1] + rad_dist/2.

ang_lim = (0.00, 180.0)
ang_N = 90
ang_bins = np.linspace(ang_lim[0], ang_lim[1], num=ang_N + 1)
ang_dist = ang_bins[1] - ang_bins[0]
ang_cent = ang_bins[:-1] + ang_dist/2.

ori_lim = (0.00, 180.0)
ori_N = 90
ori_bins = np.linspace(ori_lim[0], ori_lim[1], num=ori_N + 1)
ori_dist = ori_bins[1] - ori_bins[0]
ori_cent = ori_bins[:-1] + ori_dist/2.

rad_rmesh, ang_rmesh = np.meshgrid(rad_cent, ang_cent)
#rad_rmesh, ang_rmesh = np.meshgrid(ang_cent, rad_cent)
ori_rmesh = ang_rmesh
ang_amesh, ori_amesh = np.meshgrid(ang_cent, ori_cent)

# Evaluation info
ires = 'SCN'
ivector = (1, 0)
jres = 'SCN'
jvector = (1, 0)
jind = 0

# Define number of maxima per mixture
maxmix = [10, 10, 20, 10, 10, 10]

# Prepare array
hist_radangori = np.zeros(
    [len(info_systems), rad_N, ang_N, ori_N], dtype=np.float)
hist_maxcoords = np.zeros(
    [len(info_systems), np.max(maxmix), 3], dtype=np.float)

# Define function
def local_maxima_3D(data, order=1):
    """Detects local maxima in a 3D array

    Parameters
    ---------
    data : 3d ndarray
    order : int
        How many points on each side to use for the comparison

    Returns
    -------
    coords : ndarray
        coordinates of the local maxima
    values : ndarray
        values of the local maxima
    """
    size = 1 + 2 * order
    footprint = np.ones((size, size, size))
    footprint[order, order, order] = 0

    filtered = ndi.maximum_filter(data, footprint=footprint)
    mask_local_maxima = data > filtered
    coords = np.asarray(np.where(mask_local_maxima)).T
    values = data[mask_local_maxima]

    return coords, values

for isys, sysi in enumerate(info_systems):
    
    # Data directory
    temp = str(sysi[0])
    mix = str(sysi[1])
    datadir = os.path.join(source, temp, mix)
    
    for resid in eval_resids:
        
        # Distance-Angular file
        rdngfile = os.path.join(
            res_maindir, res_evaldir, 
            'radang_{:s}_{:s}_{:s}_{:d}.npy'.format(
                temp, mix, eval_residue, resid))
            
        # Load file
        eval_rdngs = np.load(rdngfile, allow_pickle=True).item(0)
        
        # Get data
        hist_data = eval_rdngs[ivector][jres][:, jind, :].reshape(-1, 3)
        
        # Select data
        hist_data = hist_data[hist_data[:, 0]!=0.0]
        hist_data = hist_data[hist_data[:, 0]<=rad_cut]
        
        # Convert data
        hist_data[:, 1] *= 180.0/np.pi
        hist_data[:, 2] *= 180.0/np.pi
        
        # Bin data
        hist_isys3d, _ = np.histogramdd(
            hist_data, 
            bins=[rad_bins, ang_bins, ori_bins])
        
        # Add binned data
        hist_radangori[isys] += np.array(hist_isys3d, dtype=int)
    
    # Normalize distance as in radial distribution function
    V = 4./3.*np.pi*rad_lim[1]**3
    N = np.sum(hist_radangori[isys])
    hist_radangori[isys] = V*hist_radangori[isys]/rad_dist/N
    hist_radangori[isys] = (
        hist_radangori[isys]/(4.0*np.pi*rad_cent.reshape(-1, 1, 1)**2))
    
    # Detect maxima
    coords, values = local_maxima_3D(hist_radangori[isys], order=3)
    
    # Sort maxima
    sortm = np.argsort(values)[::-1]
    coords = coords[sortm, :]
    values = values[sortm]
    
    # Print largest maxima
    print(sysi)
    
    # Save maxima coords
    for im in range(maxmix[isys]):
        hist_maxcoords[isys, im, :] = (
            rad_cent[coords[im, 0]], 
            ang_cent[coords[im, 1]], 
            ori_cent[coords[im, 2]])
        print(im, hist_maxcoords[isys, im, :])
        print(values[im])
    
#----------------------------
# Plot radial angular distr.
#----------------------------

# Fontsize
SMALL_SIZE = 12
MEDIUM_SIZE = 16
BIGGER_SIZE = 18

#plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=10)    # fontsize of the tick labels
plt.rc('ytick', labelsize=10)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

dpi = 200

vmax = np.zeros(3, dtype=float)

for isys, sysi in enumerate(info_systems):
    
    # Data directory
    temp = str(sysi[0])
    mix = str(sysi[1])
    datadir = os.path.join(source, temp, mix)
    
    # Figure - angori
    figsize = (8, 7)
    sfig = float(figsize[0])/float(figsize[1])
    fig = plt.figure(figsize=figsize)
    
    # Add axis
    axs = plt.axes(projection='3d')
    
    # Plot - Contour
    ncontour = 10
    axs.contour(
        ang_amesh, ori_amesh, np.sum(hist_radangori[isys], axis=0).T, 
        zdir='z', offset=rad_lim[0], cmap='Blues', zorder=0)
    axs.contour(
        np.sum(hist_radangori[isys], axis=1).T, ori_rmesh, rad_rmesh,
        zdir='x', offset=ang_lim[0], cmap='Reds', zorder=0)
    axs.contour(
        ang_rmesh, np.sum(hist_radangori[isys], axis=2).T, rad_rmesh,
        zdir='y', offset=ori_lim[1], cmap='Greens', zorder=0)
    
    # Get max value
    for i in range(3):
        vmax_i = np.max(np.sum(hist_radangori[isys], axis=i))
        if vmax_i > vmax[i]:
            vmax[i] = vmax_i
    
    
    axs.set_title(
        r'SCN$^-$-SCN$^-$ ' + 'radial angular distribution' + '\n'
        + 'in {:d}%/{:d}% water + acetamide'.format(int(mix), 100 - int(mix))
        + '\n')
    
    axs.set_xlabel('\n' + r'Orientation $\alpha$ ($^\circ$)', linespacing=3.4)
    axs.set_ylabel('\n' + r'Alignment $\theta$ ($^\circ$)', linespacing=3.4)
    axs.set_zlabel(
        '\n' + r'Distance $r_\mathrm{CC}$ ($\mathrm{\AA}$)', linespacing=3.4)
    
    axs.set_xlim(ang_lim)
    axs.set_ylim(ori_lim)
    axs.set_zlim(rad_lim)
    
    axs.set_xticks([0, 45, 90, 135, 180])
    axs.set_yticks([0, 45, 90, 135, 180])
    
    # Plot - Maxima
    for im in range(maxmix[isys]):
        axs.plot(
            [hist_maxcoords[isys, im, 1], ori_lim[0]],
            [hist_maxcoords[isys, im, 2], hist_maxcoords[isys, im, 2]],
            [hist_maxcoords[isys, im, 0], hist_maxcoords[isys, im, 0]],
            '--k', zorder=1)
        axs.plot(
            [hist_maxcoords[isys, im, 1], hist_maxcoords[isys, im, 1]],
            [hist_maxcoords[isys, im, 2], ori_lim[1]],
            [hist_maxcoords[isys, im, 0], hist_maxcoords[isys, im, 0]],
            '--k', zorder=1)
        axs.plot(
            [hist_maxcoords[isys, im, 1], hist_maxcoords[isys, im, 1]],
            [hist_maxcoords[isys, im, 2], hist_maxcoords[isys, im, 2]],
            [hist_maxcoords[isys, im, 0], rad_lim[0]],
            '--k', zorder=1)
        
        axs.plot(
            [hist_maxcoords[isys, im, 1]], 
            [hist_maxcoords[isys, im, 2]], 
            [hist_maxcoords[isys, im, 0]],
            'o', ms=20, alpha=1.0, mec='black', mfc='white', zorder=2)
        axs.text(
            hist_maxcoords[isys, im, 1], 
            hist_maxcoords[isys, im, 2], 
            hist_maxcoords[isys, im, 0],
            '{:d}'.format(im),
            ha='center', va='center')
        #axs.scatter(
            #hist_maxcoords[isys, im, 1], 
            #hist_maxcoords[isys, im, 2], 
            #hist_maxcoords[isys, im, 0],
            #marker='o', s=20, alpha=1.0, c='black')
        
    
    
    #plt.show()
    plt.savefig(
        os.path.join(
            res_maindir, 'rad_ang_distribution_SCN_{:s}.png'.format(mix)),
        format='png', dpi=dpi)
        
    plt.close()


#------------------------------
# Plot radial angular overview
#------------------------------

# Fontsize
SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 18

#plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
plt.rc('ytick', labelsize=12)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

dpi = 200

# Figure
figsize = (12, 6)
sfig = float(figsize[0])/float(figsize[1])
fig = plt.figure(figsize=figsize)

fig.suptitle(r'Increasing water ratio $\longrightarrow$')

# Overview indices
overind = [0, 1, 2, 3, 4]
selmix = [[0], [2], [5], [2], [0]]

# Alignment
left = 0.02
bottom = 0.50
column = [0.80/len(overind), 0.12/len(overind)]

for io, oi in enumerate(overind):
    
    # Data directory
    temp = str(info_systems[oi][0])
    mix = str(info_systems[oi][1])
    datadir = os.path.join(source, temp, mix)
    
    # Add axis
    axs = fig.add_axes(
        [left + io*np.sum(column), bottom, column[0], column[0]*sfig],
        projection='3d')
    
    # Plot - Contour
    ncontour = 10
    axs.contourf(
        ang_amesh, ori_amesh, np.sum(hist_radangori[oi], axis=0).T, 
        zdir='z', offset=rad_lim[0], cmap='Greens', zorder=0, 
        vmin=0)#, vmax=vmax[0])
    axs.contourf(
        np.sum(hist_radangori[oi], axis=1).T, ori_rmesh, rad_rmesh,
        zdir='x', offset=ang_lim[0], cmap='Blues', zorder=0, 
        vmin=0)#, vmax=vmax[1])
    axs.contourf(
        ang_rmesh, np.sum(hist_radangori[oi], axis=2).T, rad_rmesh,
        zdir='y', offset=ori_lim[1], cmap='Reds', zorder=0, 
        vmin=0)#, vmax=vmax[2])
    
    axs.grid(visible=True, which='major', axis='both', zorder=1)
    
    axs.set_title(r'{:d}%'.format(int(mix)), pad=15)
    
    if io==2:
        #axs.set_xlabel(
            #'\n' + r'$\alpha$ ($^\circ$)', linespacing=3.4)
        plt.figtext(
            0.5, bottom -0.22*column[0]*sfig, r'$\alpha$ ($^\circ$)',
            fontsize=MEDIUM_SIZE)
        
    if io==len(overind) - 1:
        axs.set_ylabel('\n' + r'$\theta$ ($^\circ$)', linespacing=3.4)
        axs.set_zlabel(
            '\n' + r'$r_\mathrm{CC}$ ($\mathrm{\AA}$)', linespacing=1.5)
    
    axs.set_xlim(ang_lim)
    axs.set_ylim(ori_lim)
    axs.set_zlim(rad_lim)
    
    axs.set_xticks([0, 45, 90, 135, 180])
    axs.tick_params(axis='x', pad=-8.0)
    for label in axs.get_xticklabels():
        label.set_rotation(50)
        label.set_ha('right')
    axs.set_yticks([0, 45, 90, 135, 180])
    axs.tick_params(axis='y', pad=-8.0)
    for label in axs.get_yticklabels():
        label.set_rotation(-40)
        label.set_ha('left')        
    axs.set_zticks([3, 5, 7])
    axs.tick_params(axis='z', pad=-2.0)
    
plt.savefig(
    os.path.join(
        res_maindir, '3D_radang_overview.png'),
    format='png', dpi=dpi)
plt.close()    

# Figure
figsize = (12, 6)
sfig = float(figsize[0])/float(figsize[1])
fig = plt.figure(figsize=figsize)

for io, oi in enumerate(overind):
    
    # Data directory
    temp = str(info_systems[oi][0])
    mix = str(info_systems[oi][1])
    datadir = os.path.join(source, temp, mix)
    
    # Add axis
    axs = fig.add_axes(
        [left + io*np.sum(column), bottom, column[0], column[0]*sfig],
        projection='3d')
    axs.set_axis_off()
    
    # Plot - Maxima
    for im in selmix[io]:
        axs.plot(
            [hist_maxcoords[oi, im, 1], ori_lim[0]],
            [hist_maxcoords[oi, im, 2], hist_maxcoords[oi, im, 2]],
            [hist_maxcoords[oi, im, 0], hist_maxcoords[oi, im, 0]],
            '--k', zorder=1)
        axs.plot(
            [hist_maxcoords[oi, im, 1], hist_maxcoords[oi, im, 1]],
            [hist_maxcoords[oi, im, 2], ori_lim[1]],
            [hist_maxcoords[oi, im, 0], hist_maxcoords[oi, im, 0]],
            '--k', zorder=1)
        axs.plot(
            [hist_maxcoords[oi, im, 1], hist_maxcoords[oi, im, 1]],
            [hist_maxcoords[oi, im, 2], hist_maxcoords[oi, im, 2]],
            [hist_maxcoords[oi, im, 0], rad_lim[0]],
            '--k', zorder=1)
        
        axs.plot(
            [hist_maxcoords[oi, im, 1]], 
            [hist_maxcoords[oi, im, 2]], 
            [hist_maxcoords[oi, im, 0]],
            'o', ms=20, alpha=1.0, mec='black', mfc='white', zorder=2)
    
    axs.set_xlim(ang_lim)
    axs.set_ylim(ori_lim)
    axs.set_zlim(rad_lim)
    
plt.savefig(
    os.path.join(
        res_maindir, '3D_radang_aux.png'),
    format='png', dpi=dpi)
plt.close()
"""
#----------------------------
# SCN-SCN ion pair life time
#----------------------------

def moving_average(data_set, periods=9):
    weights = np.ones(periods) / periods
    return np.convolve(data_set, weights, 'same')

# SCN-SCN pair distance maximum
dmax = 5.0
dmin = 5.0

# Iterate over systems and resids
smix = [0, 20, 50, 80, 90, 100]
smix = [100]

info_systems = np.array(
    list(product(temperatures, smix)))

for isys, sysi in enumerate(info_systems):
    
    # Data directory
    temp = str(sysi[0])
    mix = str(sysi[1])
    datadir = os.path.join(source, temp, mix)
    
    sfile = os.path.join(
        res_maindir, 'step_data_{:s}_{:s}.npy'.format(temp, mix))
    lfile = os.path.join(
        res_maindir, 'lftm_data_{:s}_{:s}.npy'.format(temp, mix))
    dfile = os.path.join(
        res_maindir, 'dist_data_{:s}_{:s}.npy'.format(temp, mix))
    gfile = os.path.join(
        res_maindir, 'G_data_{:s}_{:s}.npy'.format(temp, mix))
    
    if not os.path.exists(sfile) or False:
        
        # Result list
        s_data = []
        d_data = []
        l_data = []
        
        for ii, resid in enumerate(eval_resids[:-1]):
            
            #print(mix, resid)
            # Distance file
            rdngfile = os.path.join(
                res_maindir, res_evaldir, 
                'radang_{:s}_{:s}_{:s}_{:d}.npy'.format(
                    temp, mix, eval_residue, resid))
        
            
            # Load data
            # Residue atom, Target residue type, ...
            # ... time, target residue (all),  target residue atom
            eval_rdngs = np.load(rdngfile, allow_pickle=True).item(0)
            
            for jj, resjd in enumerate(eval_resids[(ii + 1):]):
            
                # Get C-C distances between two SCN anions
                dists = eval_rdngs[ivector]['SCN'][:, 0, jj + ii + 1, 0]
                
                # Do binary
                steps = np.zeros(dists.shape, dtype=bool)
                steps[dists <= dmax] = True
                
                if np.any(steps):
                    
                    d_data.append(dists)
                    
                    sequence = [
                        (k, sum(1 for _ in i)) for k, i in groupby(steps)]
                    
                s_data.append(steps)
            
        s_data = np.array(s_data)
        d_data = np.array(d_data)
        
        np.save(sfile, s_data)
        np.save(dfile, d_data)
        
    else:
        
        s_data = np.load(sfile)
        d_data = np.load(dfile)
    
    # Show distances
    #for step_i, dsteps in enumerate(d_data[:3]):
        
        #d = moving_average(d_data[step_i], periods=50)
        #plt.plot(np.arange(len(d))/10., d)
        #plt.xlabel('Time (ps)')
        #plt.ylabel('C-C distance')
        
    #plt.plot(np.arange(len(d))/10., np.ones_like(d)*5.0, '--k')
    #plt.show()
    
    # Radial binning
    rad_cut = 10.0
    rad_lim = (2.00, rad_cut)
    rad_N = 40
    rad_bins = np.linspace(rad_lim[0], rad_lim[1], num=rad_N + 1)
    rad_dist = rad_bins[1] - rad_bins[0]
    rad_cent = rad_bins[:-1] + rad_dist/2.
    
    # Get G(r) quantities
    NTtotal = d_data.shape[1]
    NTmax = int(NTtotal//2.5)
    Tmax = NTmax*eval_timestep
    dt = 1.0
    N = 200
    G = np.zeros([int(Tmax/dt), rad_N], dtype=np.float32)
    if not os.path.exists(gfile):
        
        for ni in range(N):
            
            # Get initial quantities
            istart = ni*NTmax//N
            print(istart, NTmax)
            
            di_data = d_data[:,istart:(istart + NTmax)]
            di_data = di_data[di_data[:,0] <= dmax, :]
            
            # Get time resolved G(r, t)
            V = 4./3.*np.pi*rad_lim[1]**3
            Npair = di_data.shape[0]
            for it, t in enumerate(np.arange(0.0, Tmax, dt)):
                #print(ni, it)
                G[it, :] += (
                    V*np.histogram(di_data[:, it], bins=rad_bins)[0]/rad_dist/Npair
                    /(4.0*np.pi*rad_cent**2))
                
        G = G/N
        
        np.save(gfile, G)
        
    else:
        
        G = np.load(gfile)
        
        
    # Infinite radial distribution function
    #g_hist = np.histogram(d_data.reshape(-1), bins=rad_bins)[0]
    #V = 4./3.*np.pi*rad_lim[1]**3
    #Nall = d_data.shape[0]
    #ginf = (
        #V*g_hist/rad_dist/Nall
        #/(4.0*np.pi*rad_cent**2))
    
        
    plt.plot(rad_cent, G[0,:], label=r'$t=0$ ps')
    plt.plot(rad_cent, G[10,:], label=r'$t=10$ ps')
    plt.plot(rad_cent, G[100,:], label=r'$t=100$ ps')
    plt.plot(rad_cent, G[1000,:], label=r'$t=1000$ ps')
    plt.plot(rad_cent, G[-1,:], label=r'$t=2000$ ps')
    plt.xlabel(r'r$_\mathrm{CC}$ ($\mathrm{\AA}$)')
    plt.ylabel(r'$G(r_\mathrm{CC},t)$')
    plt.title(
        r'SCN$^-$-SCN$^-$ ion pair distribution in ' 
        + '{:s}$\%$ water ratio'.format(mix))
    label = (
        r'$r_{\mathrm{cut},t=0} = $' + '{:.1f} '.format(dmax) 
        + r'$\mathrm{\AA}$')
    plt.legend(loc='upper right', title=label)
    plt.savefig('test_G_{:s}.png'.format(mix), format='png')
    plt.close()
    
    
    # Integrate first peak
    iG = np.zeros(G.shape[0])
    it = np.arange(0.0, G.shape[0]*dt, dt)
    for ii, Gt in enumerate(G):
        
        iG[ii] = np.sum(Gt[rad_cent < dmax])*rad_dist
        
    def func_decay(x, tau0, A0, tau1, A1, delta):
        return A0*np.exp(-x/tau0) + A1*np.exp(-x/tau1) + delta
        
    popt, pcov = curve_fit(
        func_decay, it[it > 200], iG[it > 200], 
        p0=[1000.0, 10.0, 5000.0, 10.0, 1.0])
    
    print(popt)
    
    plt.plot(
        it, iG, 
        label=r'Integral$_0^{5\mathrm{\AA}}$ $G(r_\mathrm{CC},t)$ d$r_\mathrm{CC}$')
    label = (
        r'Fit $\tau_0 = $' + '{:.0f} ps, '.format(popt[0])
        + r'$\tau_1 = $' + '{:.0f} ps'.format(popt[2]) + '\n'
        + r'$A_0 = $' + '{:.1f}, '.format(popt[1])
        + r'$A_1 = $' + '{:.1f}, '.format(popt[3])
        + r'$\Delta = $' + '{:.1f}'.format(popt[4]))
    plt.plot(it, func_decay(it, *popt), label=label)
    plt.xlabel(r't (ps)')
    plt.ylabel(r'Integral$_0^{5\mathrm{\AA}}$ $G(r_\mathrm{CC},t)$ d$r_\mathrm{CC}$')
    plt.title(
        r'SCN$^-$-SCN$^-$ ion pair lifetime in ' 
        + '{:s}$\%$ water ratio'.format(mix))
    plt.legend(loc='upper right')
    plt.savefig('test_iG_{:s}.png'.format(mix), format='png')
    plt.close()
    
    
    
    
    
    
    
    
    # Scan for lifetime
    c = 0
    lifetime = []
    for step_i, steps in enumerate(s_data):
        
        if np.any(steps):
            
            # Count number of ion contact trajectories
            c += 1
            
            # Get sequence length 
            sequence = [(k, sum(1 for _ in i)) for k, i in groupby(steps)]
            
            # Get lifetimes
            for seq in sequence[1:]:
                
                if seq[0]:
                    
                    lifetime.append(seq[1]*eval_timestep)
        
    print(c, s_data.shape)
    print(mix, np.mean(lifetime))
    
    # Put in histogram
    lft_lim = (0.00, np.max(lifetime))
    lft_dlft = 0.1
    lft_bins = np.arange(lft_lim[0], lft_lim[1] + lft_dlft, lft_dlft)
    lft_dist = lft_bins[1] - lft_bins[0]
    lft_cent = lft_bins[:-1] + lft_dist/2.
    
    lft_hist, _ = np.histogram(lifetime, bins=lft_bins)
    
    # Plot  result
    figsize = (8, 7)
    sfig = float(figsize[0])/float(figsize[1])
    fig = plt.figure(figsize=figsize)
    
    axs = fig.add_axes([0.2, 0.2, 0.7, 0.7])
    
    axs.bar(lft_cent, lft_hist)
    
    plt.xscale("log")
    plt.yscale("log")
    
    plt.savefig(
        os.path.join(
            res_maindir, 'liftime_{:s}.png'.format(mix)),
        format='png', dpi=300)
    plt.close()
    """
    
