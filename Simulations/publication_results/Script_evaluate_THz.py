# Basics
import os
import sys
import time
import numpy as np
from itertools import product
from glob import glob
import pickle as pkl

# Covariance function
from statsmodels.tsa.stattools import acovf, ccovf, acf, ccf

# Scipy
from scipy.optimize import curve_fit

# Trajectory reader
import MDAnalysis

# Matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, HPacker, VPacker

# ASE
from ase import Atoms
from ase.io import read, write
from ase.visualize import view
from ase import units

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

show_mixtures = [0, 20, 50, 80, 90, 95]
show_fraction = np.array([0, 2, 5, 8, 9, 10])

# Runs
Nrun = 10

# Color code
color_scheme = ['b', 'r', 'g', 'purple', 'orange', 'magenta', 'brown']*2

# Trajectory source
traj_dcdtag = 'dyna.*.dcd'
# first: split condition, second: index for irun
traj_dcdinfo = ['.', 1]
traj_crdfile = 'step1_pdbreader.crd'
traj_psffile = 'step1_pdbreader.psf'

# Result directory
res_maindir = 'results_THz'
dirs_crdndir = 'results_coordination'

# Maximum time to evaluate in ps
eval_maxtime = 5000.0

# Parallel evaluation runs
tasks = 1

# Residues
all_residues = ['SCN', 'ACEM', 'TIP3', 'POT']
all_resnum = {
    'SCN': 3,
    'ACEM': 9,
    'TIP3': 3,
    'K': 1,
    'POT': 1}

# Cross correlation residues
comp_residues = ['SCN', 'ACEM', 'TIP3', 'POT']
cross_residues = [['SCN', 'TIP3'], 
                  ['SCN', 'ACEM'],
                  ['SCN', 'POT'],
                  ['POT', 'TIP3'],
                  ['POT', 'ACEM'],
                  ['TIP3', 'ACEM']]

# Result files
a_file = os.path.join(
    res_maindir, 'Alist.npy')
f_file = os.path.join(
    res_maindir, 'Afreq.npy')
amean_file = os.path.join(
    res_maindir, 'Amean.npy')
c_file = os.path.join(
    res_maindir, 'Clist.npy')
cf_file = os.path.join(
    res_maindir, 'Cfreq.npy')
cmean_file = os.path.join(
    res_maindir, 'Cmean.npy')
cc_file = os.path.join(
    res_maindir, 'CClist.npy')
ccf_file = os.path.join(
    res_maindir, 'CCfreq.npy')
ccmean_file = os.path.join(
    res_maindir, 'CCmean.npy')

# Conversion factor
A2a0 = 1./units.Bohr

# Fontsize
SMALL_SIZE = 16
MEDIUM_SIZE = 18
BIGGER_SIZE = 20

##plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
#plt.rc('font', size=SMALL_SIZE, weight='bold')  # controls default text sizes
#plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
#plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
#plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
#plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
#plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
#plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

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
        numres[tag][res] = int(len(info_res)/all_resnum[res])
        
    # Get residue atom numbers
    atomsint[tag] = {}
    for ires, res in enumerate(all_residues):
        atomsinfo = np.zeros(
            [numres[tag][res], all_resnum[res]], dtype=int)
        for ir in range(numres[tag][res]):
            ri = all_resnum[res]*ir
            for ia in range(all_resnum[res]):
                info = listres[res][ri + ia].split()
                atomsinfo[ir, ia] = int(info[0]) - 1
        atomsint[tag][res] = atomsinfo
        
# Make result directory
if not os.path.exists(res_maindir):
    os.mkdir(res_maindir)
"""
#---------------------
# Collect system data
#---------------------

# Iterate over systems
info_systems = np.array(list(product(temperatures, mixtures, range(Nrun))))
"""
def read_sepsys(i):
    
    # Begin timer
    start = time.time()
    
    # Data directory
    temp = str(info_systems[i][0])
    mix = str(info_systems[i][1])
    run = str(info_systems[i][2])
    datadir = os.path.join(source, temp, "{:s}_{:s}".format(mix, run))
    
    # System tag
    tag = temp + '_' + mix
    
    # Get dcd files
    dcdfiles = np.array(glob(os.path.join(datadir, traj_dcdtag)))
    iruns = np.array([
        int(dcdfile.split('/')[-1].split(traj_dcdinfo[0])[traj_dcdinfo[1]])
        for dcdfile in dcdfiles])
    psffile = os.path.join(datadir, traj_psffile)
    
    # Sort dcd files
    dcdsort = np.argsort(iruns)
    dcdfiles = dcdfiles[dcdsort]
    
    for tarres in all_residues:
    
        # Dipole file
        dipofile = os.path.join(
            res_maindir, 
            'dipos_{:s}_{:s}_{:s}_{:s}.npy'.format(tarres, temp, run, mix))
        
        # Time step file
        timefile = os.path.join(
            res_maindir, 
            'times_{:s}_{:s}_{:s}_{:s}.npy'.format(tarres, temp, run, mix))
        
        if not os.path.exists(dipofile):
            
            # Initialize dipole and time list
            dipole_list = []
            time_list = []
            
            # Initialize trajectory time counter in ps
            traj_time_dcd = 0.0
            
            # Iterate over dcd files
            finished = False
            for idcd, dcdfile in enumerate(dcdfiles):
                
                # Open dcd file
                dcd = MDAnalysis.Universe(psffile, dcdfile)
                
                # Get residue information
                residues = dcd._topology.resnames.values
                residues = np.repeat(
                    residues, [all_resnum[resi] for resi in residues])
                
                # Get atoms of residue tarres
                select = residues==tarres
                
                # Get atom types and select of correct residue
                atoms = [ai[:1] for ai in dcd._topology.names.values]
                atoms = np.array(atoms)[select]
                
                # Get atom charges
                charges = dcd._topology.charges.values[select]
                
                # Get trajectory parameter
                Nframes = len(dcd.trajectory)
                Natoms = len(atoms)
                Nskip = int(dcd.trajectory.skip_timestep)
                dt = np.round(
                    float(dcd.trajectory._ts_kwargs['dt']), decimals=8)/Nskip
                
                # Iterate over frames
                for ii, frame in enumerate(dcd.trajectory):
                    
                    # Update time
                    traj_time = traj_time_dcd + Nskip*dt*ii
                    
                    if not int(1000*traj_time)%10000:
                        print(temp, mix, idcd, traj_time, eval_maxtime)
                    
                    # Check time
                    if traj_time > eval_maxtime:
                        
                        # Convert dipole list to array
                        dipole_list = np.array(dipole_list)
                        
                        # Save dipole
                        np.save(dipofile, dipole_list)
                        
                        # Save time steps
                        np.save(timefile, time_list)
                        
                        # End timer
                        end = time.time()
                        print(
                        'System {:s}, {:s}, {:s} done for {:.0f} ps in {:4.1f} s'.format(
                            tarres, temp, mix, traj_time_dcd*1000 , end - start))
                        print('Max time reached!')
                        
                        finished = True
                        break
                    
                    # Get positions
                    new_positions = frame._pos[select]
                    
                    # Neglect periodic boundary conditions
                    if traj_time==0.0:
                        
                        positions = new_positions
                    
                    else:
                        
                        # Get cell information
                        cell = frame._unitcell
                        
                        # Neglect periodic boundary conditions
                        chnge = new_positions - old_positions
                        shift = np.zeros_like(chnge)
                        for i in range(3):
                            L = cell[i]
                            shift[:,i] = (
                                (chnge[:,i] + L/2.0) % L - L/2.0 - chnge[:,i])
                        chnge += shift 
                        
                        # Neglect periodic boundary conditions
                        positions += chnge
                    
                    # Calculate dipole
                    dipole = np.dot(charges, positions*A2a0)
                    
                    # Append dipole to list
                    dipole_list.append(dipole)
                    
                    # Append time step to list
                    time_list.append(traj_time)
                    
                    # Append positions to list
                    #system = Atoms(atoms, positions=positions)
                    #positions_list.append(system)
                    
                    # Update old positions
                    old_positions = new_positions.copy()
                    
                if finished:
                    
                    break
                    
                # Update trajectory time
                traj_time_dcd = traj_time
                
            if not finished:
            
                # Convert dipole list to array
                dipole_list = np.array(dipole_list)
                
                # Save dipole
                np.save(dipofile, dipole_list)
                
                # Save time steps
                np.save(timefile, time_list)
            
                # End timer
                end = time.time()
                print(
                    'System {:s}, {:s}, {:s} done for {:.0f} ps in {:4.1f} s'.format(
                        tarres, temp, mix, traj_time_dcd*1000 , end - start))
                print('Out of dcd!')
                
                
        else:
            
            end = time.time()
            print(
                'System {:s}, {:s}, {:s} already done'.format(
                    tarres, temp, mix))
            
    return

if tasks==1:
    for i in range(0, len(info_systems)):
        read_sepsys(i)
else:    
    if __name__ == '__main__':
        pool = Pool(tasks)
        pool.imap(read_sepsys, range(0, len(info_systems)))
        pool.close()
        pool.join()

"""

#---------------------------------------
# Define Functions
#---------------------------------------

def autocorr(x):

    mean=np.mean(x)
    var=np.var(x)
    xp=x-mean
    r2=np.fft.ifft(np.abs(np.fft.fft(xp))**2).real
    c=(r2/xp.shape-np.mean(xp)**2)/np.std(xp)**2
    return c*var
    
def moving_average(data_set, periods=9):
    weights = np.ones(periods) / periods
    return np.convolve(data_set, weights, 'same')



#--------------------------
# Calculate Spectra
#--------------------------

# Iterate over systems
info_systems = np.array(list(product(temperatures, mixtures)))

# Evaluation options
flimit = [13.0, 34.0]
Nave = 80
"""
# List absorption
A_list = []
A_list_original = [] 

# List mean absorption
Amean_list = []
Astdv_list = [] 

# List transmission
T_list = [] 

# List mean transmission
Tmean_list = [] 
Tstdv_list = []

# Component list
c_list = []

# Mean component list
cmean_list = []
cstdv_list = []

# Cross product list
cc_list = []

# Mean cross product list
ccmean_list = []
ccstdv_list = []

# Frequency range
f_list = []
cf_list = []
ccf_list = []

for isys, sysi in enumerate(info_systems):
    
    # Data directory
    temp = str(sysi[0])
    mix = str(sysi[1])
    datadir = os.path.join(source, temp, mix)
    
    print(sysi)
    
    # Conversion parameter
    beta = 1.0/3.1668114e-6/float(sysi[0])
    hbar = 1.0
    cminvtoau = 1.0/2.1947e5
    const = beta*cminvtoau*hbar
    # Time for speed of light in vacuum to travel 1 cm
    jiffy = 33.3564
    
    # Result lists
    f_list_run = []
    A_list_run = []
    A_list_original_run = []
    Amean_list_run = []
    Astdv_list_run = []
    T_list_run = []
    Tmean_list_run = []
    Tstdv_list_run = []
    
    # Components
    c_list_run = []
    cmean_list_run = []
    cstdv_list_run = []
    cf_list_run = []
    
    # Cross Components
    cc_list_run = []
    ccmean_list_run = []
    ccstdv_list_run = []
    ccf_list_run = []
    
    for run in range(Nrun):
        
        # Load dipole file
        print('Full')
        dipofile = os.path.join(
            source, res_maindir, 
            'dipos_{:s}_{:s}_{:s}.npy'.format(temp, mix, str(run)))
        dipos = np.load(dipofile)
        
        # Load time file
        timefile = os.path.join(
            source, res_maindir, 
            'times_{:s}_{:s}_{:s}.npy'.format(temp, mix, str(run)))
        times = np.load(timefile)
        
        # Save dipoles and times as txt file
        time_dipos = np.zeros(
            [4 + 3*len(comp_residues), len(times)], dtype=np.float)
        time_dipos[0] = times
        time_dipos[1:4, :] = dipos.T
        
        # Number of frames and frequency points
        Nframes = len(times)
        Nfreq = int(Nframes/2) + 1
        
        # Time step size
        dtime = times[1] - times[0]
        
        # Prepare spectrum
        spec = np.zeros([9, Nfreq])
        
        # Get IR spectra
        
        acfx = acf(dipos[:,0], nlags=Nframes - 1)
        acfy = acf(dipos[:,1], nlags=Nframes - 1)
        acfz = acf(dipos[:,2], nlags=Nframes - 1)
        
        acfxdt = acf(np.gradient(dipos[:,0]), nlags=Nframes - 1)
        acfydt = acf(np.gradient(dipos[:,1]), nlags=Nframes - 1)
        acfzdt = acf(np.gradient(dipos[:,2]), nlags=Nframes - 1)
        
        acvx = acovf(dipos[:,0], fft=True)
        acvy = acovf(dipos[:,1], fft=True)
        acvz = acovf(dipos[:,2], fft=True)
        
        acvxdt = acovf(np.gradient(dipos[:,0]), fft=True)
        acvydt = acovf(np.gradient(dipos[:,1]), fft=True)
        acvzdt = acovf(np.gradient(dipos[:,2]), fft=True)
        
        acfxdtt = ccf(
            np.gradient(dipos[:,0]), dipos[:,0], adjusted=False, fft=True)
        acfydtt = ccf(
            np.gradient(dipos[:,1]), dipos[:,1], adjusted=False, fft=True)
        acfzdtt = ccf(
            np.gradient(dipos[:,2]), dipos[:,2], adjusted=False, fft=True)
        
        acvxdtt = ccovf(
            np.gradient(dipos[:,0]), dipos[:,0], adjusted=False, fft=True)
        acvydtt = ccovf(
            np.gradient(dipos[:,1]), dipos[:,1], adjusted=False, fft=True)
        acvzdtt = ccovf(
            np.gradient(dipos[:,2]), dipos[:,2], adjusted=False, fft=True)
        
        
        acfxdttr = ccf(
            dipos[:,0], np.gradient(dipos[:,0]), adjusted=False, fft=True)
        acfydttr = ccf(
            dipos[:,1], np.gradient(dipos[:,1]), adjusted=False, fft=True)
        acfzdttr = ccf(
            dipos[:,2], np.gradient(dipos[:,2]), adjusted=False, fft=True)
        
        acvxdttr = ccovf(
            dipos[:,0], np.gradient(dipos[:,0]), adjusted=False, fft=True)
        acvydttr = ccovf(
            dipos[:,1], np.gradient(dipos[:,1]), adjusted=False, fft=True)
        acvzdttr = ccovf(
            dipos[:,2], np.gradient(dipos[:,2]), adjusted=False, fft=True)
        
        sum_acf = acfx + acfy + acfz
        sum_acfdt = acfxdt + acfydt + acfzdt
        sum_acfdtt = acfxdtt + acfydtt + acfzdtt
        sum_acfdttr = acfxdttr + acfydttr + acfzdttr
        sum_acv = acvx + acvy + acvz
        sum_acvdt = acvxdt + acvydt + acvzdt
        sum_acvdtt = acvxdtt + acvydtt + acvzdtt
        sum_acvdttr = acvxdttr + acvydttr + acvzdttr
        
        #acf = acf*np.blackman(Nframes)
        #acv = acv*np.blackman(Nframes)
        #acfdt = acfdt*np.blackman(Nframes)
        #acvdt = acvdt*np.blackman(Nframes)
        
        # Transform spectra
        spec[0, :] = np.arange(Nfreq)/float(Nframes)/dtime*jiffy
        
        spec[1, :] = (
            (np.imag(np.fft.rfftn(sum_acf)))*np.tanh(const*spec[0, :]/2.))
        
        spec[2, :] = (
            (np.imag(np.fft.rfftn(sum_acfdt))))
        
        spec[3, :] = (
            (np.imag(np.fft.rfftn(sum_acfdtt)))*np.tanh(const*spec[0, :]/2.))
        
        spec[4, :] = (
            (np.imag(np.fft.rfftn(sum_acfdttr)))*np.tanh(const*spec[0, :]/2.))
        
        spec[5, :] = (
            (np.imag(np.fft.rfftn(sum_acv)))*np.tanh(const*spec[0, :]/2.))
        
        spec[6, :] = (
            (np.imag(np.fft.rfftn(sum_acvdt))))
        
        spec[7, :] = (
            (np.imag(np.fft.rfftn(sum_acvdtt)))*np.tanh(const*spec[0, :]/2.))
        
        spec[8, :] = (
            (np.imag(np.fft.rfftn(sum_acvdttr)))*np.tanh(const*spec[0, :]/2.))
        
        
        # Apply moving average
        spec[1, :] = moving_average(spec[1, :], Nave)
        spec[2, :] = moving_average(spec[2, :], Nave)
        spec[3, :] = moving_average(spec[3, :], Nave)
        spec[4, :] = moving_average(spec[4, :], Nave)
        spec[5, :] = moving_average(spec[5, :], Nave)
        spec[6, :] = moving_average(spec[6, :], Nave)
        spec[7, :] = moving_average(spec[7, :], Nave)
        spec[8, :] = moving_average(spec[8, :], Nave)
        mvavg = (spec[0, 1] - spec[0, 0])*Nave
        
        # Get scaling between results
        frange = np.logical_and(spec[0, :] > flimit[0], spec[0, :] < flimit[1])
        smax1 = np.max(spec[1, :][frange])
        smax2 = np.max(spec[2, :][frange])
        smax3 = np.max(spec[3, :][frange])
        smax4 = np.max(spec[4, :][frange])
        smax5 = np.max(spec[5, :][frange])
        smax6 = np.max(spec[6, :][frange])
        smax7 = np.max(spec[7, :][frange])
        smax8 = np.max(spec[8, :][frange])
        smin1 = np.min(spec[1, :][frange])
        smin2 = np.min(spec[2, :][frange])
        smin3 = np.min(spec[3, :][frange])
        smin4 = np.min(spec[4, :][frange])
        smin5 = np.min(spec[5, :][frange])
        smin6 = np.min(spec[6, :][frange])
        smin7 = np.min(spec[7, :][frange])
        smin8 = np.min(spec[8, :][frange])
        
        
        scal1 = 1./(smax1 - smin1)
        scal2 = 1./(smax2 - smin2)
        scal3 = 1./(smax3 - smin3)
        scal4 = 1./(smax4 - smin4)
        scal5 = 1./(smax5 - smin5)
        scal6 = 1./(smax6 - smin6)
        scal7 = 1./(smax7 - smin7)
        scal8 = 1./(smax8 - smin8)
        
        # Get absorption spectra between 13 and 34 cm^-1
        frange = np.logical_and(spec[0, :] > flimit[0], spec[0, :] < flimit[1])
        chose = 6
        Arange = spec[chose, :][frange]
        Arange_original = spec[1, :][frange]
        
        # Convert to transmission spectra
        Trange = 10**(-Arange)
        
        # Get mean
        Amean = np.mean(Arange)
        Astdv = np.std(Arange)
        Tmean = np.mean(Trange)
        Tstdv = np.std(Trange)
        
        # Add to list
        f_list_run.append(spec[0, :][frange])
        A_list_run.append(Arange)
        A_list_original_run.append(Arange_original)
        Amean_list_run.append(Amean)
        Astdv_list_run.append(Astdv)
        T_list_run.append(Trange)
        Tmean_list_run.append(Tmean)
        Tstdv_list_run.append(Tstdv)
        
        # Components
        
        ci_list_run = []
        cimean_list_run = []
        cistdv_list_run = []
        cfi_list_run = []
        for it, tarres in enumerate(comp_residues):
            
            print(tarres)
            
            # Load dipole file
            dipofile = os.path.join(
                source, res_maindir, 
                'dipos_{:s}_{:s}_{:s}_{:s}.npy'.format(tarres, temp, str(run), mix))
            dipos = np.load(dipofile)
            
            time_dipos[4 + 3*it:4 + 3*(it + 1), :] = dipos.T
            
            # Load time file
            timefile = os.path.join(
                source, res_maindir, 
                'times_{:s}_{:s}_{:s}_{:s}.npy'.format(tarres, temp, str(run), mix))
            times = np.load(timefile)
            
            # Number of frames and frequency points
            Nframes = len(times)
            Nfreq = int(Nframes/2) + 1
            
            # Time step size
            dtime = times[1] - times[0]
            
            # Prepare spectrum
            spec = np.zeros([9, Nfreq])
            
            # Get IR spectra
            
            acfx = acf(dipos[:,0], nlags=Nframes - 1)
            acfy = acf(dipos[:,1], nlags=Nframes - 1)
            acfz = acf(dipos[:,2], nlags=Nframes - 1)
            
            acfxdt = acf(np.gradient(dipos[:,0]), nlags=Nframes - 1)
            acfydt = acf(np.gradient(dipos[:,1]), nlags=Nframes - 1)
            acfzdt = acf(np.gradient(dipos[:,2]), nlags=Nframes - 1)
            
            acvx = acovf(dipos[:,0], fft=True)
            acvy = acovf(dipos[:,1], fft=True)
            acvz = acovf(dipos[:,2], fft=True)
            
            acvxdt = acovf(np.gradient(dipos[:,0]), fft=True)
            acvydt = acovf(np.gradient(dipos[:,1]), fft=True)
            acvzdt = acovf(np.gradient(dipos[:,2]), fft=True)
            
            acfxdtt = ccf(
                np.gradient(dipos[:,0]), dipos[:,0], adjusted=False, fft=True)
            acfydtt = ccf(
                np.gradient(dipos[:,1]), dipos[:,1], adjusted=False, fft=True)
            acfzdtt = ccf(
                np.gradient(dipos[:,2]), dipos[:,2], adjusted=False, fft=True)
            
            acvxdtt = ccovf(
                np.gradient(dipos[:,0]), dipos[:,0], adjusted=False, fft=True)
            acvydtt = ccovf(
                np.gradient(dipos[:,1]), dipos[:,1], adjusted=False, fft=True)
            acvzdtt = ccovf(
                np.gradient(dipos[:,2]), dipos[:,2], adjusted=False, fft=True)
            
            
            acfxdttr = ccf(
                dipos[:,0], np.gradient(dipos[:,0]), adjusted=False, fft=True)
            acfydttr = ccf(
                dipos[:,1], np.gradient(dipos[:,1]), adjusted=False, fft=True)
            acfzdttr = ccf(
                dipos[:,2], np.gradient(dipos[:,2]), adjusted=False, fft=True)
            
            acvxdttr = ccovf(
                dipos[:,0], np.gradient(dipos[:,0]), adjusted=False, fft=True)
            acvydttr = ccovf(
                dipos[:,1], np.gradient(dipos[:,1]), adjusted=False, fft=True)
            acvzdttr = ccovf(
                dipos[:,2], np.gradient(dipos[:,2]), adjusted=False, fft=True)
            
            sum_acf = acfx + acfy + acfz
            sum_acfdt = acfxdt + acfydt + acfzdt
            sum_acfdtt = acfxdtt + acfydtt + acfzdtt
            sum_acfdttr = acfxdttr + acfydttr + acfzdttr
            sum_acv = acvx + acvy + acvz
            sum_acvdt = acvxdt + acvydt + acvzdt
            sum_acvdtt = acvxdtt + acvydtt + acvzdtt
            sum_acvdttr = acvxdttr + acvydttr + acvzdttr
            
            #acf = acf*np.blackman(Nframes)
            #acv = acv*np.blackman(Nframes)
            #acfdt = acfdt*np.blackman(Nframes)
            #acvdt = acvdt*np.blackman(Nframes)
            
            # Transform spectra
            spec[0, :] = np.arange(Nfreq)/float(Nframes)/dtime*jiffy
            
            spec[1, :] = (
                (np.imag(np.fft.rfftn(sum_acf)))*np.tanh(const*spec[0, :]/2.))
            
            spec[2, :] = (
                (np.imag(np.fft.rfftn(sum_acfdt))))
            
            spec[3, :] = (
                (np.imag(np.fft.rfftn(sum_acfdtt)))*np.tanh(const*spec[0, :]/2.))
            
            spec[4, :] = (
                (np.imag(np.fft.rfftn(sum_acfdttr)))*np.tanh(const*spec[0, :]/2.))
            
            spec[5, :] = (
                (np.imag(np.fft.rfftn(sum_acv)))*np.tanh(const*spec[0, :]/2.))
            
            spec[6, :] = (
                (np.imag(np.fft.rfftn(sum_acvdt))))
            
            spec[7, :] = (
                (np.imag(np.fft.rfftn(sum_acvdtt)))*np.tanh(const*spec[0, :]/2.))
            
            spec[8, :] = (
                (np.imag(np.fft.rfftn(sum_acvdttr)))*np.tanh(const*spec[0, :]/2.))
            
            
            # Apply moving average
            spec[1, :] = moving_average(spec[1, :], Nave)
            spec[2, :] = moving_average(spec[2, :], Nave)
            spec[3, :] = moving_average(spec[3, :], Nave)
            spec[4, :] = moving_average(spec[4, :], Nave)
            spec[5, :] = moving_average(spec[5, :], Nave)
            spec[6, :] = moving_average(spec[6, :], Nave)
            spec[7, :] = moving_average(spec[7, :], Nave)
            spec[8, :] = moving_average(spec[8, :], Nave)
            mvavg = (spec[0, 1] - spec[0, 0])*Nave
            
            # Get scaling between results
            frange = np.logical_and(spec[0, :] > flimit[0], spec[0, :] < flimit[1])
            smax1 = np.max(spec[1, :][frange])
            smax2 = np.max(spec[2, :][frange])
            smax3 = np.max(spec[3, :][frange])
            smax4 = np.max(spec[4, :][frange])
            smax5 = np.max(spec[5, :][frange])
            smax6 = np.max(spec[6, :][frange])
            smax7 = np.max(spec[7, :][frange])
            smax8 = np.max(spec[8, :][frange])
            smin1 = np.min(spec[1, :][frange])
            smin2 = np.min(spec[2, :][frange])
            smin3 = np.min(spec[3, :][frange])
            smin4 = np.min(spec[4, :][frange])
            smin5 = np.min(spec[5, :][frange])
            smin6 = np.min(spec[6, :][frange])
            smin7 = np.min(spec[7, :][frange])
            smin8 = np.min(spec[8, :][frange])
            
            
            scal1 = 1./(smax1 - smin1)
            scal2 = 1./(smax2 - smin2)
            scal3 = 1./(smax3 - smin3)
            scal4 = 1./(smax4 - smin4)
            scal5 = 1./(smax5 - smin5)
            scal6 = 1./(smax6 - smin6)
            scal7 = 1./(smax7 - smin7)
            scal8 = 1./(smax8 - smin8)
            
            # Get absorption spectra between 13 and 34 cm^-1
            frange = np.logical_and(spec[0, :] > flimit[0], spec[0, :] < flimit[1])
            chose = 6
            cArange = spec[chose, :][frange]
            
            # Get mean
            cAmean = np.mean(cArange)
            cAstdv = np.std(cArange)
            
            # Append data
            ci_list_run.append(cArange)
            cimean_list_run.append(cAmean)
            cistdv_list_run.append(cAstdv)
            cfi_list_run.append(spec[0, :][frange])
            
        # Append component section data
        c_list_run.append(ci_list_run)
        cmean_list_run.append(cimean_list_run)
        cstdv_list_run.append(cistdv_list_run)
        cf_list_run.append(cfi_list_run)
        
        # Cross correlation
        
        cci_list_run = []
        ccimean_list_run = []
        ccistdv_list_run = []
        ccfi_list_run = []
        for tarres in cross_residues:
            
            print(tarres)
            
            # Load dipole file
            dipofile0 = os.path.join(
                source, res_maindir, 
                'dipos_{:s}_{:s}_{:s}_{:s}.npy'.format(tarres[0], temp, str(run), mix))
            dipofile1 = os.path.join(
                source, res_maindir, 
                'dipos_{:s}_{:s}_{:s}_{:s}.npy'.format(tarres[1], temp, str(run), mix))
            dipos0 = np.load(dipofile0)
            dipos1 = np.load(dipofile1)
            
            # Load time file
            timefile = os.path.join(
                source, res_maindir, 
                'times_{:s}_{:s}_{:s}_{:s}.npy'.format(tarres[0], temp, str(run), mix))
            times = np.load(timefile)
            
            # Number of frames and frequency points
            Nframes = len(times)
            Nfreq = int(Nframes/2) + 1
            
            # Time step size
            dtime = times[1] - times[0]
            
            # Prepare spectrum
            cspec = np.zeros([2, Nfreq])
            
            # Get Cross spectra
            
            #acfxdtt = ccf(
                #np.gradient(dipos0[:,0]), dipos1[:,0], adjusted=False, fft=True)
            #acfydtt = ccf(
                #np.gradient(dipos0[:,1]), dipos1[:,1], adjusted=False, fft=True)
            #acfzdtt = ccf(
                #np.gradient(dipos0[:,2]), dipos1[:,2], adjusted=False, fft=True)
            
            #acvxdtt = ccovf(
                #np.gradient(dipos0[:,0]), dipos1[:,0], adjusted=False, fft=True)
            #acvydtt = ccovf(
                #np.gradient(dipos0[:,1]), dipos1[:,1], adjusted=False, fft=True)
            #acvzdtt = ccovf(
                #np.gradient(dipos0[:,2]), dipos1[:,2], adjusted=False, fft=True)
            
            acvxdtt = ccovf(
                np.gradient(dipos0[:,0]), np.gradient(dipos1[:,0]), adjusted=False, fft=True)
            acvydtt = ccovf(
                np.gradient(dipos0[:,1]), np.gradient(dipos1[:,1]), adjusted=False, fft=True)
            acvzdtt = ccovf(
                np.gradient(dipos0[:,2]), np.gradient(dipos1[:,2]), adjusted=False, fft=True)
            
            
            #sum_acfdtt = acfxdtt + acfydtt + acfzdtt
            sum_acvdtt = acvxdtt + acvydtt + acvzdtt
            
            cspec[0, :] = np.arange(Nfreq)/float(Nframes)/dtime*jiffy
            #cspec[1, :] = (
                #(np.imag(np.fft.rfftn(sum_acfdtt)))*np.tanh(const*spec[0, :]/2.))
            #cspec[1, :] = (
                #(np.imag(np.fft.rfftn(sum_acvdtt)))*np.tanh(const*spec[0, :]/2.))
            cspec[1, :] = (
                (np.imag(np.fft.rfftn(sum_acvdtt))))
            
            # Apply moving average
            cspec[1, :] = moving_average(cspec[1, :], Nave)
            
            # Get scaling between results
            frange = np.logical_and(cspec[0, :] > flimit[0], cspec[0, :] < flimit[1])
            smax1 = np.max(cspec[1, :][frange])
            
            # Get absorption spectra between 13 and 34 cm^-1
            frange = np.logical_and(
                cspec[0, :] > flimit[0], cspec[0, :] < flimit[1])
            cArange = cspec[1, :][frange]
            
            # Get mean
            cmean = np.mean(cArange)
            cstdv = np.std(cArange)
            
            # Append data
            cci_list_run.append(cArange)
            ccimean_list_run.append(cmean)
            ccistdv_list_run.append(cstdv)
            ccfi_list_run.append(spec[0, :][frange])
            
        # Append cross section data
        cc_list_run.append(cci_list_run)
        ccmean_list_run.append(ccimean_list_run)
        ccstdv_list_run.append(ccistdv_list_run)
        ccf_list_run.append(ccfi_list_run)
        
    # Result lists
    f_list.append(np.mean(np.array(f_list_run), axis=0))
    A_list.append(np.mean(np.array(A_list_run), axis=0))
    A_list_original.append(np.mean(np.array(A_list_original_run), axis=0))
    Amean_list.append(np.mean(np.array(Amean_list_run), axis=0))
    Astdv_list.append(np.mean(np.array(Astdv_list_run), axis=0))
    T_list.append(np.mean(np.array(T_list_run), axis=0))
    Tmean_list.append(np.mean(np.array(Tmean_list_run), axis=0))
    Tstdv_list.append(np.mean(np.array(Tstdv_list_run), axis=0))
    
    # Components
    c_list.append(
        np.array([
            np.mean(
                np.array(
                    [c_list_run[ii][jj] for ii in range(Nrun)]),
                axis=0)
            for jj in range(len(c_list_run[0]))]))
    print(c_list[-1].shape)
    cmean_list.append(
        np.array([
            np.mean(
                np.array(
                    [cmean_list_run[ii][jj] for ii in range(Nrun)]),
                axis=0)
            for jj in range(len(cmean_list_run[0]))]))
    cstdv_list.append(
        np.array([
            np.mean(
                np.array(
                    [cstdv_list_run[ii][jj] for ii in range(Nrun)]),
                axis=0)
            for jj in range(len(cstdv_list_run[0]))]))
    cf_list.append(
        np.array([
            np.mean(
                np.array(
                    [cf_list_run[ii][jj] for ii in range(Nrun)]),
                axis=0)
            for jj in range(len(cf_list_run[0]))]))
    
    # Cross Components
    cc_list.append(
        np.array([
            np.mean(
                np.array(
                    [cc_list_run[ii][jj] for ii in range(Nrun)]),
                axis=0)
            for jj in range(len(cc_list_run[0]))]))
    ccmean_list.append(
        np.array([
            np.mean(
                np.array(
                    [ccmean_list_run[ii][jj] for ii in range(Nrun)]),
                axis=0)
            for jj in range(len(ccmean_list_run[0]))]))
    ccstdv_list.append(
        np.array([
            np.mean(
                np.array(
                    [ccstdv_list_run[ii][jj] for ii in range(Nrun)]),
                axis=0)
            for jj in range(len(ccstdv_list_run[0]))]))
    ccf_list.append(
        np.array([
            np.mean(
                np.array(
                    [ccf_list_run[ii][jj] for ii in range(Nrun)]),
                axis=0)
            for jj in range(len(ccf_list_run[0]))]))
    

# Save lists
np.save(a_file, A_list, allow_pickle=True)
np.save(f_file, f_list, allow_pickle=True)
np.save(amean_file, Amean_list, allow_pickle=True)
np.save(c_file, c_list, allow_pickle=True)
np.save(cf_file, cf_list, allow_pickle=True)
np.save(cmean_file, cmean_list, allow_pickle=True)
np.save(cc_file, cc_list, allow_pickle=True)
np.save(ccf_file, ccf_list, allow_pickle=True)
np.save(ccmean_file, ccmean_list, allow_pickle=True)
"""
# Load lists
A_list = np.load(a_file, allow_pickle=True)
f_list = np.load(f_file, allow_pickle=True)
Amean_list = np.load(amean_file, allow_pickle=True)
c_list = np.load(c_file, allow_pickle=True)
cf_list = np.load(cf_file, allow_pickle=True)
cmean_list = np.load(cmean_file, allow_pickle=True)
cc_list = np.load(cc_file, allow_pickle=True)
ccf_list = np.load(ccf_file, allow_pickle=True)
ccmean_list = np.load(ccmean_file, allow_pickle=True)

# Get Intensity Integral
df = [fi[1] - fi[0] for fi in f_list]
Amean_list = np.array([np.sum(A_list[im])*df[im] for im in range(len(Amean_list))])

df = [fi[0, 1] - fi[0, 0] for fi in cf_list]
cmean_list = np.array([np.sum(c_list[im], axis=1)*df[im] for im in range(len(cmean_list))])

df = [fi[0, 1] - fi[0, 0] for fi in ccf_list]
ccmean_list = np.array([np.sum(cc_list[im], axis=1)*df[im] for im in range(len(ccmean_list))])

# Reduce Spectra
redind = np.array([0, 2, 5, 8, 9, 10])
A_list = [A_list[im] for im in redind]
f_list = [f_list[im] for im in redind]
c_list = [c_list[im] for im in redind]
cf_list = [cf_list[im] for im in redind]
cc_list = [cc_list[im] for im in redind]
ccf_list = [ccf_list[im] for im in redind]





#----------------------------
# Final Plot
#----------------------------

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


# Load experimental data
file_ExpA = open(os.path.join("source", 'DES_THzSpectra_forKT.txt'), 'r')
lines_ExpA = file_ExpA.readlines()
file_ExpA.close()

# Allocate data
for il, line in enumerate(lines_ExpA):
    
    if il==0:
        
        fraction_str = line.split(',')[1:]
        fraction_num = [float(fstr.split('%')[0]) for fstr in fraction_str]
        
        Expf = np.zeros(len(lines_ExpA) - 1, dtype=np.float)
        ExpA = np.zeros(
            [len(fraction_num), len(lines_ExpA) - 1], dtype=np.float)

    else:
        
        result_str = line.split(',')
        Expf[il - 1] = float(result_str[0])
        ExpA[:, il - 1] = np.array(result_str[1:], dtype=np.float)

# Invert water ratio order 
fraction_num = np.array(fraction_num[::-1])
ExpA = ExpA[::-1]

frange = np.logical_and(Expf > flimit[0], Expf < flimit[1])
Nrange = len(ExpA[0][frange])

# Load experimental averages
file_AvgA = open(
    os.path.join("source", 'DES_THzAvgAbsroption_forKT.txt'), 'r')
lines_AvgA = file_AvgA.readlines()
file_AvgA.close()

# Allocate data
Avgm = []
AvgA = []
for il, line in enumerate(lines_AvgA):
    
    if il!=0:
        
        result_str = line.split(',')
        Avgm.append(float(result_str[0]))
        AvgA.append(float(result_str[1]))
        
Avgm = np.array(Avgm, dtype=np.float)
AvgA = np.array(AvgA, dtype=np.float)

# Integrated Intensity
Avgm = fraction_num
df = Expf[1] - Expf[0]
AvgA = np.array([np.sum(ExpA_mix[frange])*df for ExpA_mix in ExpA])


eval_reslab = 'TIP3'
eval_resctr = 0
eval_resdst = [3.0, 3.5, 4.0, 4.5, 5.0, 5.5]
evel_nummax = 48
eval_pltlab = r'H_2O'

plot_idxdst = [0, 3]
crdnN = np.zeros([len(plot_idxdst), len(mixtures), evel_nummax + 1])
average_list = [[] for _ in plot_idxdst]
stddvtn_list = [[] for _ in plot_idxdst]

for isys, sys in enumerate(info_systems):
    
    # Data directory
    temp = sys[0]
    mix = sys[1]
    tag = str(temp) + '_' + str(mix)
    
    crdnfile = os.path.join(
        dirs_crdndir, 'evalfiles', 
        'crdn_{:d}_{:d}_{:s}.npy'.format(temp, mix, eval_reslab))
    
    coordination = pkl.load(open(crdnfile, "rb"))
    
    for ir, idxr in enumerate(plot_idxdst):
        
        crdnN[ir, np.where(mixtures==mix)[0][0], :] = \
            coordination['N'][idxr]/np.sum(coordination['N'][idxr])
        
        rcut = eval_resdst[idxr]
        
        # Average
        ncrdntn = crdnN[ir, np.where(mixtures==mix)[0][0], :32]
        N = np.arange(len(ncrdntn))
        avrg = np.sum(ncrdntn*N)
        average_list[ir].append(avrg)
        
        # Standard deviation
        stdv = np.sqrt(np.sum(ncrdntn*(N - avrg)**2))
        stddvtn_list[ir].append(stdv)

average_list = np.array(average_list)
stddvtn_list = np.array(stddvtn_list)

# Figure
figsize = (12, 6)
sfig = float(figsize[0])/float(figsize[1])
fig = plt.figure(figsize=figsize)

# Alignment
left = 0.10
bottom = 0.15
column = [0.20, 0.05]
row = [0.30, 0.15]

line_scheme = [
    'solid', 'dotted', 'dashed', 'dashdot', (0, (3, 1, 3, 1, 1, 1)), 
    (0, (3, 1, 1, 1, 1, 1)), (0, (3, 1, 1, 1, 1, 1, 1, 1))]

label_sim = [
    '0%',
    '20%',
    '50%',
    '80%',
    '90%',
    '95%']

label_exp = [
    '0%',
    '20%',
    '50%',
    '80%',
    '90%',
    '100%']

idx_exp_sim = []
idx_avg_sim = []
for mix in mixtures:
    if len(np.where(mix==np.round(fraction_num, decimals=-1))[0]):
        idx_exp_sim.append(
            np.where(mix==np.round(fraction_num, decimals=-1))[0][0])
        idx_avg_sim.append(np.where(mix==np.round(Avgm, decimals=-1))[0][0])
idx_exp_sim = np.array(idx_exp_sim)
idx_avg_sim = np.array(idx_avg_sim)


axs1 = fig.add_axes([
    left, bottom + 1*sum(row), column[0], row[0]])
axs2 = fig.add_axes([
    left + 1*np.sum(column), bottom + 1*sum(row), column[0], row[0]])
axs3 = fig.add_axes([
    left + 2*np.sum(column), bottom + 1*sum(row), column[0], row[0]])
axs4l = fig.add_axes([
    left, bottom, column[0]*1.5, row[0]])
axs4r = axs4l.twinx()

axs5l = fig.add_axes([
    left + 2.0*np.sum(column), bottom, column[0]*1.5, row[0]])
axs5r = axs5l.twinx()




# Experiment
Amax_exp = 0.0
im = 0
for ia in show_fraction:
    
    if ia in idx_exp_sim:
        
        axs1.plot(
            Expf[frange], ExpA[ia][frange], 
            color=color_scheme[im], ls=line_scheme[im])
        
        if np.max(ExpA[ia][frange]) > Amax_exp:
            Amax_exp = np.max(ExpA[ia][frange])
            
        im += 1
        
    if ia==10:
        axs1.plot(
            Expf[frange], ExpA[ia][frange], 
            color=color_scheme[im + 1], ls=line_scheme[im + 1])
            
        if np.max(ExpA[ia][frange]) > Amax_exp:
            Amax_exp = np.max(ExpA[ia][frange])
    
        
dlimit = flimit[1] - flimit[0]
axs1.set_xlim(flimit[0] - dlimit*0.02, flimit[1] + dlimit*0.02)
axs1.set_ylim(0.0, Amax_exp*1.2)

axs1.set_title(
    'Experiment', fontweight=font)

#axs1.set_xlabel(r'Frequency (cm$^{-1}$)')
#axs1.get_xaxis().set_label_coords(1.0 + column[1]/column[0]/2., -0.20)
axs1.set_ylabel(r'Absorption (a.u.)', fontweight=font)
axs1.get_yaxis().set_label_coords(-0.22, 0.50)

axs1.set_xticks([15, 20, 25, 30])

tbox = TextArea('A', textprops=dict(
    color='k', fontsize=MEDIUM_SIZE))

anchored_tbox = AnchoredOffsetbox(
    loc='upper left', child=tbox, pad=0., frameon=False,
    bbox_to_anchor=(0.02*1.4, 0.95),
    bbox_transform=axs1.transAxes, borderpad=0.)

axs2.add_artist(anchored_tbox)




# Simulation
Amax_sim = 0.0
for im, mix in enumerate(show_mixtures):
    
    Nsim = int(len(A_list[im])/Nrange) + 1
    
    axs2.plot(
        f_list[im][::Nsim], A_list[im][::Nsim], 
        color=color_scheme[im], ls=line_scheme[im],
        label=label_sim[im])
    
    if A_list[im][-1] > Amax_sim:
        Amax_sim = A_list[im][-1]
        
axs2.plot(
        f_list[-1][::Nsim], A_list[-1][::Nsim] + 1000.0, 
        color=color_scheme[im + 1], ls=line_scheme[im],
        label=label_exp[-1])
    
dlimit = flimit[1] - flimit[0]
axs2.set_xlim(flimit[0] - dlimit*0.02, flimit[1] + dlimit*0.02)
axs2.set_ylim(0.0, Amax_sim*1.2)
#axs2.set_ylim(0.0, Amax_exp*1.2)

axs2.set_title(
    'Simulation', fontweight=font)

axs2.set_xlabel(r'Frequency (cm$^{-1}$)')
axs2.get_xaxis().set_label_coords(0.5, -0.20)

axs2.set_xticks([15, 20, 25, 30])
#axs2.set_yticklabels([])

#axs2.legend(loc=[1.10, 0.0], title='Water ratio', framealpha=1.0)

tbox = TextArea('B', textprops=dict(
    color='k', fontsize=MEDIUM_SIZE))

anchored_tbox = AnchoredOffsetbox(
    loc='upper left', child=tbox, pad=0., frameon=False,
    bbox_to_anchor=(0.02*1.4, 0.95),
    bbox_transform=axs2.transAxes, borderpad=0.)

axs2.add_artist(anchored_tbox)





# Water
Amax_wat = 0.0
for im, mix in enumerate(show_mixtures):
    
    Nsim = int(len(c_list[im][2])/Nrange) + 1
    
    axs3.plot(
        cf_list[im][2][::Nsim], c_list[im][2][::Nsim], 
        color=color_scheme[im], ls=line_scheme[im],
        label=label_sim[im])
    
    if A_list[im][-1] > Amax_sim:
        Amax_sim = A_list[im][-1]

axs3.plot(
    cf_list[-1][2][::Nsim], c_list[-1][2][::Nsim] + 1000.0, 
    color=color_scheme[im + 1], ls=line_scheme[im + 1],
    label=label_exp[-1])

dlimit = flimit[1] - flimit[0]
axs3.set_xlim(flimit[0] - dlimit*0.02, flimit[1] + dlimit*0.02)
axs3.set_ylim(0.0, Amax_sim*1.2)

axs3.set_title(
    r'Component H$_2$O', fontweight=font)

#axs3.set_xlabel(r'Frequency (cm$^{-1}$)')
#axs3.get_xaxis().set_label_coords(0.5, -0.20)

axs3.set_xticks([15, 20, 25, 30])
#axs3.set_yticklabels([])

handles, labels = axs3.get_legend_handles_labels()
order = np.arange(len(handles))[::-1]
axs3.legend(
    [handles[idx] for idx in order], [labels[idx] for idx in order],
    loc=[1.10, -0.1], title='Water ratio', framealpha=1.0)

tbox = TextArea('C', textprops=dict(
    color='k', fontsize=MEDIUM_SIZE))

anchored_tbox = AnchoredOffsetbox(
    loc='upper left', child=tbox, pad=0., frameon=False,
    bbox_to_anchor=(0.02*1.4, 0.95),
    bbox_transform=axs3.transAxes, borderpad=0.)

axs3.add_artist(anchored_tbox)






# Mean Adsorption

# Scaling factor
#def compare(x, sc, sh):
    #return np.sum(x[0] - sc*(x[1] + sh))
#Amean_list = np.array(Amean_list)
#Astdv_list = np.array(Astdv_list)
#popt, pcov = curve_fit(
    #compare, [AvgA[idx_exp_sim], Amean_list], np.zeros(len(idx_exp_sim)),
    #p0=[0.75, 0.10])
#scale, shift = popt[0], popt[1]

# Simulation
axs4r.plot(mixtures, Amean_list, 'r-o', ms=5.0)

# Experiment
axs4l.plot(Avgm, AvgA, 'b-s', ms=5.0, label='Experiment')
axs4l.plot(mixtures, Amean_list + 100.0, 'r-o', label='Simulation')

"""
# Coordination number
label = r'$\bar{n}_\mathrm{H_2O}$'

diff1 = Amean_list[-1] - Amean_list[0]
diff2 = average_list[1][-1] - average_list[1][0]

avrg_list = average_list[1][:]
avrg_list -= avrg_list[0]
avrg_list *= (diff1/diff2)
avrg_list += Amean_list[0]

stdv_list = stddvtn_list[1][:]
stdv_list *= (diff1/diff2)
    
axs4r.errorbar(
    mixtures, avrg_list, yerr=stdv_list, fmt='m-x', ms=5.0)
axs4l.plot(
    mixtures, avrg_list + 100.0, 'm-x', ms=5.0, label=label)
"""

axs4l.set_xlim(-10, 110)
axs4r.set_xlim(-10, 110)
AvgA_min = np.min(AvgA)
AvgA_max = np.max(AvgA)
dAvgA = AvgA_max - AvgA_min
AvgAs_min = np.min(Amean_list)
AvgAs_max = np.max(Amean_list)
dAvgAs = AvgAs_max - AvgAs_min

axs4l.set_ylim(AvgA_min - dAvgA*0.2, AvgA_max + dAvgA*0.2)
axs4l.tick_params(axis='y', colors='blue')
axs4r.set_ylim(AvgAs_min - dAvgAs*0.2, AvgAs_max + dAvgAs*0.2)
##axsr.set_ylim(0.5 + shift, AvgA_max*1.2/scale)
#axs4r.set_ylim(0.5, AvgA_max*1.1)
axs4r.tick_params(axis='y', colors='red')

axs4l.set_xticks([0, 20, 40, 60, 80, 100])
axs4l.set_xticklabels(['0', '20', '40', '60', '80', '100'])
axs4r.set_xticks([0, 20, 40, 60, 80, 100])
axs4r.set_xticklabels(['0', '20', '40', '60', '80', '100'])

axs4l.set_yticks([15, 25, 35])
axs4r.set_yticks([30, 50, 70])
#axs4r.set_yticks([25, 50, 75, 100])
#axs4r.set_yticklabels(["1.0", "4.0", "7.0"])

axs4l.set_xlabel(r'Mixing Ratio (%)', fontweight=font)
axs4l.get_xaxis().set_label_coords(0.5, -0.25)
axs4l.set_ylabel('Integrated\nAbsorption (a.u.)', fontweight=font)
axs4l.get_yaxis().set_label_coords(-0.15, 0.50)

axs4l.legend(
    loc=[0.02, 0.5], framealpha=0.0, frameon=False)

tbox = TextArea('D', textprops=dict(
    color='k', fontsize=MEDIUM_SIZE))

anchored_tbox = AnchoredOffsetbox(
    loc='upper left', child=tbox, pad=0., frameon=False,
    bbox_to_anchor=(0.02, 0.95),
    bbox_transform=axs4l.transAxes, borderpad=0.)

axs4l.add_artist(anchored_tbox)






# Component

# Coordination number
label = r'$\left< n_\mathrm{H_2O} \right>$'

avrg_list = average_list[1][:]
stdv_list = stddvtn_list[1][:]
    
axs5r.errorbar(
    mixtures, avrg_list, yerr=stdv_list, fmt='m-x', ms=5.0, zorder=5)
#axs5r.plot(
    #mixtures, avrg_list, 'm-x', ms=5.0, zorder=1)
#axs5l.plot(
    #mixtures, avrg_list + 1000.0, 'm-x', ms=5.0, label=label)
axs5l.errorbar(
    mixtures, avrg_list + 1000.0, yerr=stdv_list, fmt='m-x', ms=5.0, 
    label=label)
axs5r.tick_params(axis='y', colors='m')

# Simulation
axs5l.plot(mixtures, Amean_list, 'r-o', ms=5.0, label='Simulation', zorder=3)

# Water
axs5l.plot(
    mixtures, [cimean[2] for cimean in cmean_list], 
    '-o', mec='darkcyan', mfc='None', ms=5.0, label=r'H$_2$O', zorder=2)


AvgAs_min = np.min(Amean_list)
AvgAs_max = np.max(Amean_list)
dAvgAs = AvgAs_max - AvgAs_min
axs5l.set_ylim(0.0, AvgAs_max*1.2)
axs5r.set_ylim(0.0, np.max(avrg_list)*1.3)

axs5l.set_xlim(-10, 110)
axs5r.set_xlim(-10, 110)

axs5l.set_xticks([0, 20, 40, 60, 80, 100])
axs5l.set_xticklabels(['0', '20', '40', '60', '80', '100'])
axs5r.set_xticks([0, 20, 40, 60, 80, 100])
axs5r.set_xticklabels(['0', '20', '40', '60', '80', '100'])

axs5l.set_yticks([0, 30, 50, 70])
#axs5l.set_yticks([25, 50, 75, 100])
#axs5l.set_yticklabels(["0", "50", "100", "150"])

axs5l.set_xlabel(r'Mixing Ratio (%)', fontweight=font)
axs5l.get_xaxis().set_label_coords(0.5, -0.25)
axs5l.set_ylabel('Integrated\nAbsorption (a.u.)', fontweight=font)
axs5l.get_yaxis().set_label_coords(-0.20, 0.50)

axs5r.set_ylabel(
    'Coordination\nNumber {:s}'.format(label), 
    fontweight=font, color='m')
axs5r.get_yaxis().set_label_coords(1.15, 0.50)


axs5l.legend(
    loc=[0.10, 0.45], framealpha=0.0, frameon=False)

tbox = TextArea('E', textprops=dict(
    color='k', fontsize=MEDIUM_SIZE))

anchored_tbox = AnchoredOffsetbox(
    loc='upper left', child=tbox, pad=0., frameon=False,
    bbox_to_anchor=(0.02, 0.95),
    bbox_transform=axs5l.transAxes, borderpad=0.)

axs5l.add_artist(anchored_tbox)


#plt.show()
plt.savefig(
    os.path.join(
        res_maindir, 'paper_THz_{:d}_final_int.png'.format(temperatures[0])),
    format='png', dpi=dpi)
plt.close()





























#----------------------------
# SI Plot
#----------------------------

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

# Figure
figsize = (12, 12)
sfig = float(figsize[0])/float(figsize[1])
fig = plt.figure(figsize=figsize)

# Alignment
left = 0.10
bottom = 0.15
column = [0.18, 0.03]
row = [0.14, 0.08]

line_scheme = [
    'solid', 'dotted', 'dashed', 'dashdot', (0, (3, 1, 3, 1, 1, 1)), 
    (0, (3, 1, 1, 1, 1, 1))]

label_sim = [
    '0%',
    '20%',
    '50%',
    '80%',
    '90%',
    '95%']

# Determine maximum
Amax = 0.0
for im, mix in enumerate(show_mixtures):
    for ir, resi in enumerate(comp_residues):
        Amaxi = np.max(c_list[im][ir])
        if np.max(c_list[im][ir]) > Amax:
            Amax = Amaxi


# Plot component spectra

#comp_residues = ['SCN', 'ACEM', 'TIP3', 'POT']

label_row = ['A', 'B', 'C', 'D']
title_row = [r'SCN$^-$', r'K$^+$', 'Acetamide', r'H$_2$O']
order_row = np.array([0, 3, 1, 2])
for ia, ir in enumerate(order_row):
        
    axs = fig.add_axes([
        left + ia*np.sum(column), bottom + 3*sum(row), column[0], row[0]])
    
    for im, mix in enumerate(show_mixtures):
    
        
        Nsim = int(len(c_list[im][ir])/Nrange) + 1
        
        axs.plot(
            cf_list[im][ir][::Nsim], c_list[im][ir][::Nsim], 
            color=color_scheme[im], ls=line_scheme[im],
            label=label_sim[im])
        
    dlimit = flimit[1] - flimit[0]
    axs.set_xlim(flimit[0] - dlimit*0.02, flimit[1] + dlimit*0.02)
    axs.set_ylim(0.0, Amax)
    
    axs.set_xticks([15, 20, 25, 30])
    axs.set_yticks([0, 2, 4])
    #axs.set_yticks([0, 2, 4, 6])
    if ia!=0:
        axs.set_yticklabels([])
    
    if ia==0:
        axs.set_ylabel('Partial\nAbsorption (a.u.)', fontweight=font)
        axs.get_yaxis().set_label_coords(-0.25, 0.50)
    elif ia==1:
        axs.set_xlabel(r'Frequency (cm$^{-1}$)')
        axs.get_xaxis().set_label_coords(
            1.0 + column[1]/column[0]/2., -0.20)

    axs.set_title(
        title_row[ia], fontweight=font)
    
    
    tbox = TextArea(label_row[ia], textprops=dict(
        color='k', fontsize=MEDIUM_SIZE))

    anchored_tbox = AnchoredOffsetbox(
        loc='upper left', child=tbox, pad=0., frameon=False,
        bbox_to_anchor=(0.03, 0.95),
        bbox_transform=axs.transAxes, borderpad=0.)

    axs.add_artist(anchored_tbox)

        
    
# Plot cross product spectra

#cross_residues = [['SCN', 'TIP3'], 
                  #['SCN', 'ACEM'],
                  #['SCN', 'POT'],
                  #['POT', 'TIP3'],
                  #['POT', 'ACEM'],
                  #['TIP3', 'ACEM']]

label_row = ['E', 'F', 'G']
label_res = {
    'SCN': r'SCN$^-$',
    'POT': r'K$^+$',
    'K': r'K$^+$',
    'TIP3': r'H$_2$O',
    'ACEM': 'Acetamide'}
iselect = [2, 1, 0]
for ir, (resi, resj) in enumerate(
    [cross_residues[2], cross_residues[1], cross_residues[0]]):
    
    axs = fig.add_axes([
        left + ir*np.sum(column), bottom + 2.*sum(row) + row[0]/3., 
        column[0], row[0]/2.0])
    
    titlei = '{:s} x {:s}'.format(label_res[resi], label_res[resj])
    
    for im, mix in enumerate(show_mixtures):
        
        Nsim = int(len(cc_list[im][iselect[ir]])/Nrange) + 1
        
        axs.plot(
            ccf_list[im][iselect[ir]][::Nsim], cc_list[im][iselect[ir]][::Nsim], 
            color=color_scheme[im], ls=line_scheme[im],
            label=label_sim[im])
    
    dlimit = flimit[1] - flimit[0]
    axs.set_xlim(flimit[0] - dlimit*0.02, flimit[1] + dlimit*0.02)
    axs.set_ylim(-1.0, 1.0)
    
    axs.set_xticks([15, 20, 25, 30])
    axs.set_xticklabels([])
    axs.set_yticks([-1, 0, 1])
    if ir!=0:
        axs.set_yticklabels([])
    
    #if ir==0:
        #axs.set_ylabel('Cross\nCorrelation (a.u.)', fontweight=font)
        #axs.get_yaxis().set_label_coords(-0.25, 0.50)
    #elif ir==1:
        #axs.set_xlabel(r'Frequency (cm$^{-1}$)')
        #axs.get_xaxis().set_label_coords(0.5, -0.20)
    #elif ir==2:
        #axs.legend(loc=[1.20, 0.0], title='Water ratio', framealpha=1.0)

    axs.set_title(
        titlei, fontweight=font)
    
    tbox = TextArea(label_row[ir], textprops=dict(
        color='k', fontsize=MEDIUM_SIZE))

    anchored_tbox = AnchoredOffsetbox(
        loc='upper left', child=tbox, pad=0., frameon=False,
        bbox_to_anchor=(0.03, 0.95),
        bbox_transform=axs.transAxes, borderpad=0.)

    axs.add_artist(anchored_tbox)
    
    
    
# Plot cross product spectra

#cross_residues = [['SCN', 'TIP3'], 
                  #['SCN', 'ACEM'],
                  #['SCN', 'POT'],
                  #['POT', 'TIP3'],
                  #['POT', 'ACEM'],
                  #['TIP3', 'ACEM']]

label_row = ['H', 'I', 'K']
label_res = {
    'SCN': r'SCN$^-$',
    'POT': r'K$^+$',
    'K': r'K$^+$',
    'TIP3': r'H$_2$O',
    'ACEM': 'Acetamide'}
iselect = [4, 3, 5]
for ir, (resi, resj) in enumerate(
    [cross_residues[4], cross_residues[3], cross_residues[5][::-1]]):
    
    axs = fig.add_axes([
        left + ir*np.sum(column), bottom + 1.7*sum(row), 
        column[0], row[0]/2.])
    
    titlei = '{:s} x {:s}'.format(label_res[resi], label_res[resj])
    
    for im, mix in enumerate(show_mixtures):
        
        Nsim = int(len(cc_list[im][iselect[ir]])/Nrange) + 1
        
        axs.plot(
            ccf_list[im][iselect[ir]][::Nsim], cc_list[im][iselect[ir]][::Nsim], 
            color=color_scheme[im], ls=line_scheme[im],
            label=label_sim[im])
    
    dlimit = flimit[1] - flimit[0]
    axs.set_xlim(flimit[0] - dlimit*0.02, flimit[1] + dlimit*0.02)
    axs.set_ylim(-1.0, 1.0)
    
    axs.set_xticks([15, 20, 25, 30])
    axs.set_yticks([-1, 0, 1])
    if ir!=0:
        axs.set_yticklabels([])
    
    if ir==0:
        axs.set_ylabel('Cross\nCorrelation (a.u.)', fontweight=font)
        axs.get_yaxis().set_label_coords(-0.25, 1.50)
    elif ir==1:
        axs.set_xlabel(r'Frequency (cm$^{-1}$)')
        axs.get_xaxis().set_label_coords(0.5, -0.20*2)
    elif ir==2:
        handles, labels = axs.get_legend_handles_labels()
        order = np.arange(len(handles))[::-1]
        axs.legend(
            [handles[idx] for idx in order], [labels[idx] for idx in order],
            loc=[1.20, 0.5], title='Water ratio', framealpha=1.0)

    axs.set_title(
        titlei, fontweight=font)
    
    tbox = TextArea(label_row[ir], textprops=dict(
        color='k', fontsize=MEDIUM_SIZE))

    anchored_tbox = AnchoredOffsetbox(
        loc='upper left', child=tbox, pad=0., frameon=False,
        bbox_to_anchor=(0.03, 0.95),
        bbox_transform=axs.transAxes, borderpad=0.)

    axs.add_artist(anchored_tbox)
    



sb = 1.0

# Determine maximum
for im, mix in enumerate(mixtures):
    Amax = np.max(Amean_list[im])

# Plot component mean absorption 
axs = fig.add_axes([
    left, bottom + 0.7*np.sum(row), 
    column[0]*sb, row[0]])

#comp_residues = ['SCN', 'ACEM', 'TIP3', 'POT']

title_row = [r'SCN$^-$', r'K$^+$', 'Acetamide', r'H$_2$O']
cross_line_scheme = [
    'dotted', 'dashed', 'dashdot', (0, (3, 1, 3, 1, 1, 1))]
marker_scheme = [
    's', 'd', 'v', 'P']
color_scheme = ['purple', 'orange', 'blue', 'magenta']

#axs.plot(mixtures, Amean_list, 'r-o', ms=5.0, label='Full')
order_row = np.array([0, 3, 1, 2])
for ia, ir in enumerate(order_row):
    
    axs.plot(
        mixtures, [cmean_list[im][ir] for im in range(len(mixtures))], 
        color=color_scheme[ir], ls=cross_line_scheme[ir],
        marker=marker_scheme[ir],
        label=title_row[ia], ms=5.0)
    
axs.set_xlim(-10, 110)
axs.set_ylim(0.0, np.max(Amean_list)*1.05)

axs.set_xticks([0, 20, 40, 60, 80, 100])
axs.set_xticklabels(['0', '20', '40', '60', '80', '100'])
#axs.set_yticks([0, 2, 4, 6])
axs.set_yticks([0, 30, 50, 70])
#axs.set_yticks([0, 25, 50, 75, 100])


axs.set_xlabel(r'Mixing Ratio (%)', fontweight=font)
axs.get_xaxis().set_label_coords(0.5, -0.25)
axs.set_ylabel('Integrated\nAbsorption (a.u.)', fontweight=font)
axs.get_yaxis().set_label_coords(-0.25/sb, 0.50)

axs.legend(
    loc=[1.10, 0.0], framealpha=1.0, frameon=True, title='Single Components')


tbox = TextArea('L', textprops=dict(
        color='k', fontsize=MEDIUM_SIZE))

anchored_tbox = AnchoredOffsetbox(
    loc='upper left', child=tbox, pad=0., frameon=False,
    bbox_to_anchor=(0.03/sb, 0.95),
    bbox_transform=axs.transAxes, borderpad=0.)

axs.add_artist(anchored_tbox)



# Plot  mean cross correlation
axs = fig.add_axes([
    left + 2.25*np.sum(column), bottom + 0.7*np.sum(row), 
    column[0]*sb, row[0]])


#cross_residues = [['SCN', 'TIP3'], 
                  #['SCN', 'ACEM'],
                  #['SCN', 'POT'],
                  #['POT', 'TIP3'],
                  #['POT', 'ACEM'],
                  #['TIP3', 'ACEM']]


cross_line_scheme = [
    'dashed', 'dashdot', (0, (3, 1, 3, 1, 1, 1)), 
    (0, (2, 2, 2, 2, 2, 2)), (0, (1, 1, 1, 1, 1, 1)), (0, (2, 1, 1, 1, 2, 1))]
marker_scheme = [
    'd', 'v', 'P', '>', 'D', 'p']
color_scheme = ['orange', 'blue', 'magenta', 'cyan', 'r', 'g', ]

label_res = {
    'SCN': r'SCN$^-$',
    'POT': r'K$^+$',
    'K': r'K$^+$',
    'TIP3': r'H$_2$O',
    'ACEM': 'Acetamide'}
iselect = [2, 1, 0, 4, 3, 5]
for ir, (resi, resj) in enumerate([
    cross_residues[2], cross_residues[1], cross_residues[0],
    cross_residues[4], cross_residues[3], cross_residues[5]]):
    
    titlei = '{:s} x {:s}'.format(label_res[resi], label_res[resj])
    #titlei = '{:s}'.format(label_res[resj])
    
    axs.plot(
        mixtures, [ccmean_list[im][iselect[ir]] for im in range(len(mixtures))], 
        color=color_scheme[ir], ls=cross_line_scheme[ir],
        marker=marker_scheme[ir],
        label=titlei, ms=5.0, mfc='None')
    
axs.set_xlim(-10, 110)
#axs.set_ylim(0.0, np.max(Amean_list)*1.05/3.)
#axs.set_ylim(-1.0, 1.0)

axs.set_xticks([0, 20, 40, 60, 80, 100])
axs.set_xticklabels(['0', '20', '40', '60', '80', '100'])
#axs.set_yticks([0, 2, 4, 6])
#axs.set_yticks([0, 1, 2])
axs.set_yticks([-10, 0, 10])

axs.set_xlabel(r'Mixing Ratio (%)', fontweight=font)
axs.get_xaxis().set_label_coords(0.5, -0.25)
axs.set_ylabel('Integrated Cross\nCorrelation (a.u.)', fontweight=font)
axs.get_yaxis().set_label_coords(-0.25/sb, 0.50)

axs.legend(
    loc=[1.10, 0.0], framealpha=1.0, frameon=True, 
    title=r'Cross components')


tbox = TextArea('M', textprops=dict(
        color='k', fontsize=MEDIUM_SIZE))

anchored_tbox = AnchoredOffsetbox(
    loc='upper left', child=tbox, pad=0., frameon=False,
    bbox_to_anchor=(0.03/sb, 0.95),
    bbox_transform=axs.transAxes, borderpad=0.)

axs.add_artist(anchored_tbox)




# Plot summation

axs = fig.add_axes([
    left + 0.25*np.sum(column), bottom - 0.3*np.sum(row), 
    column[0]*1.5*sb, row[0]])
axsr = axs.twinx()



# Plot experiment

# Load experimental data
file_ExpA = open(os.path.join("source", 'DES_THzSpectra_forKT.txt'), 'r')
lines_ExpA = file_ExpA.readlines()
file_ExpA.close()

# Allocate data
for il, line in enumerate(lines_ExpA):
    
    if il==0:
        
        fraction_str = line.split(',')[1:]
        fraction_num = [float(fstr.split('%')[0]) for fstr in fraction_str]
        
        Expf = np.zeros(len(lines_ExpA) - 1, dtype=np.float)
        ExpA = np.zeros(
            [len(fraction_num), len(lines_ExpA) - 1], dtype=np.float)

    else:
        
        result_str = line.split(',')
        Expf[il - 1] = float(result_str[0])
        ExpA[:, il - 1] = np.array(result_str[1:], dtype=np.float)

# Invert water ratio order 
fraction_num = fraction_num[::-1]
ExpA = ExpA[::-1]

frange = np.logical_and(Expf > flimit[0], Expf < flimit[1])
Nrange = len(ExpA[0][frange])

# Load experimental averages
file_AvgA = open(
    os.path.join("source", 'DES_THzAvgAbsroption_forKT.txt'), 'r')
lines_AvgA = file_AvgA.readlines()
file_AvgA.close()

# Allocate data
Avgm = []
AvgA = []
for il, line in enumerate(lines_AvgA):
    
    if il!=0:
        
        result_str = line.split(',')
        Avgm.append(float(result_str[0]))
        AvgA.append(float(result_str[1]))
        
Avgm = np.array(Avgm, dtype=np.float)
AvgA = np.array(AvgA, dtype=np.float)

# Integrated Intensity
Avgm = fraction_num
df = Expf[1] - Expf[0]
AvgA = np.array([np.sum(ExpA_mix[frange])*df for ExpA_mix in ExpA])


axs.plot(Avgm, AvgA, 'b-s', ms=5.0, label='Experiment')






# Plot simulated spectra
axsr.plot(mixtures, np.array(Amean_list), 'r-o', ms=5.0, label='Simulation')
axs.plot(
    mixtures, np.array(Amean_list) + 1000.0, 'r-o', ms=5.0, label='Simulation')

# Plot summed spectra
Smean_list = np.array([
    np.sum(cmean_list[im], axis=0) + np.sum(ccmean_list[im], axis=0) 
    for im in range(len(mixtures))])
axsr.plot(
    mixtures, Smean_list, '--r', marker='P',
    label=r'$\Sigma$ Components', ms=5.0)
axs.plot(
    mixtures, Smean_list + 1000.0, '--r', marker='P',
    label=r'$\Sigma$ Components', ms=5.0)
    

axs.set_xlim(-10, 110)
axsr.set_xlim(-10, 110)


AvgA_min = np.min(AvgA)
AvgA_max = np.max(AvgA)
dAvgA = AvgA_max - AvgA_min
AvgAs_min = np.min(Amean_list)
AvgAs_max = np.max(Amean_list)
dAvgAs = AvgAs_max - AvgAs_min

axs.set_ylim(AvgA_min - dAvgA*0.2, AvgA_max + dAvgA*0.2)
axs.tick_params(axis='y', colors='blue')
axsr.set_ylim(AvgAs_min - dAvgAs*0.2, AvgAs_max + dAvgAs*0.2)

axs.set_xticks([0, 20, 40, 60, 80, 100])
axs.set_xticklabels(['0', '20', '40', '60', '80', '100'])

axsr.tick_params(axis='y', colors='red')

axs.set_xlabel(r'Mixing Ratio (%)', fontweight=font)
axs.get_xaxis().set_label_coords(0.5, -0.25)
axs.set_ylabel('Integrated\nAbsorption (a.u.)', fontweight=font)
axs.get_yaxis().set_label_coords(-0.20/sb, 0.50)

axs.set_yticks([15, 25, 35])
axsr.set_yticks([30, 50, 70])
#axsr.set_yticks([25, 50, 75, 100])


axs.legend(
    loc=[1.20, 0.0], framealpha=1.0, frameon=True)


tbox = TextArea('N', textprops=dict(
        color='k', fontsize=MEDIUM_SIZE))

anchored_tbox = AnchoredOffsetbox(
    loc='upper left', child=tbox, pad=0., frameon=False,
    bbox_to_anchor=(0.03/sb, 0.95),
    bbox_transform=axs.transAxes, borderpad=0.)

axs.add_artist(anchored_tbox)

plt.savefig(
    os.path.join(
        res_maindir, 'paper_THz_{:d}_si_int.png'.format(temperatures[0])),
    format='png', dpi=dpi)
plt.close()
