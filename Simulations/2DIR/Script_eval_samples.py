# Basics
import os
import sys
import numpy as np

# Interpolation
from scipy.interpolate import interp1d, interp2d

# Matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, HPacker, VPacker

# Miscellaneous
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

# Data parameter order [separator, [temp, mix, #run, #sample]]
data_sepinf = ['_', [1, 2, 3, 4]]

# Output file
data_outfle = 'out_{:s}_{:s}_{:s}_{:s}_SCN.txt'

# Linear spectra result file
data_slnfle = 'spec_lin_{:s}.dat'

# 2D spectra result file
data_s2dfle = 'spec_2D_{:s}.dat'

# Reference data evaluation directory
data_evldir = "eval_results_samples"

# Reference frequencies plot
data_evlfrq = "refs_frequencies_{:s}_{:s}_{:s}_{:s}.png"

# 1D reference frequencies spectra plot
data_evlspc = "spec_frequencies_{:s}_{:s}_{:s}_{:s}.png"

# 2D simulated spectra plot
data_evls2d = "2Dspec_{:s}_{:s}_{:s}_{:s}.png"

# Mean reference frequencies plot
data_evlavg = "mean_frequencies_{:s}_{:s}_{:s}.png"

# Reference data evaluation directory
data_fitdir = "fitted_results_samples"

# 1D simulated spectra plot
data_avgdln = "lnspec_{:s}_{:s}.png"

# 2D simulated spectra plot
data_avgd2d = "2Dspec_{:s}_{:s}.png"

# Averaged 1D simulated spectra file
data_aslnfl = 'spec_lin_{:s}.dat'

# Averaged 2D simulated spectra file
data_as2dfl = 'spec_2D_{:s}.dat'

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


#---------------------------------
# Reading Section
#---------------------------------

# Get sample list
data_frqlst = glob(
    os.path.join(data_smpdir, data_frqtag.format('*', '*', '*', '*')))

# Get system parameter
data_params = {}
for frqfle in data_frqlst:
    
    # Extract information
    temp, mix, irun, isample = [
        frqfle.split('/')[-1].split(data_sepinf[0])[ii] 
        for ii in data_sepinf[1]]
    
    # Add to parameter list
    if not temp in data_params.keys():
        
        data_params[temp] = {}
        
    if not mix in data_params[temp].keys():
        
        if not os.path.exists(os.path.join(temp, mix)):
            
            continue
        
        data_params[temp][mix] = {}
        
    if not irun in data_params[temp][mix].keys():
        
        data_params[temp][mix][irun] = []

    if not isample in data_params[temp][mix][irun]:
        
        if not os.path.exists(
            os.path.join(
                temp, 
                mix, 
                "{:s}_{:s}".format(irun, isample),
                data_outfle.format(temp, mix, irun, isample))):
            
            continue
        
        data_params[temp][mix][irun].append(isample)
        
# Iterate over system parameter
data_specfr = {}
data_specln = {}
data_spec2d = {}
data_dtmdly = {}
data_mnmxfr = {}
data_mnmxln = {}
data_mnmx2d = {}
for temp in data_params.keys():
    data_specfr[temp] = {}
    data_specln[temp] = {}
    data_spec2d[temp] = {}
    data_dtmdly[temp] = {}
    data_mnmxfr[temp] = {}
    data_mnmxln[temp] = {}
    data_mnmx2d[temp] = {}
    for mix in data_params[temp].keys():
        data_specfr[temp][mix] = {}
        data_specln[temp][mix] = {}
        data_spec2d[temp][mix] = {}
        data_dtmdly[temp][mix] = {}
        data_mnmxfr[temp][mix] = {}
        data_mnmxln[temp][mix] = {}
        data_mnmx2d[temp][mix] = {}
        for irun in data_params[temp][mix].keys():
            data_specfr[temp][mix][irun] = []
            data_specln[temp][mix][irun] = []
            data_spec2d[temp][mix][irun] = []
            data_dtmdly[temp][mix][irun] = []
            data_mnmxfr[temp][mix][irun] = [np.inf, 0.0]
            data_mnmxln[temp][mix][irun] = [np.inf, 0.0]
            data_mnmx2d[temp][mix][irun] = [np.inf, 0.0]
            for isample in data_params[temp][mix][irun]:
            
                # Working directory
                wrkdir = os.path.join(temp, mix, "{:s}_{:s}".format(
                    irun, isample))
                
                # Output file
                outfle = data_outfle.format(temp, mix, irun, isample)
                
                # linear spectra result file
                slnfle = data_slnfle.format(mix)

                # 2D spectra result file
                s2dfle = data_s2dfle.format(mix)
                
                # Read results
                with open(os.path.join(wrkdir, slnfle), 'r') as f:
                    slnlns = f.readlines()
                with open(os.path.join(wrkdir, s2dfle), 'r') as f:
                    s2dlns = f.readlines()
                
                # Get linear spectra
                frqarr = []
                slnarr = []
                for lne in slnlns:
                    frqarr.append(float(lne.split()[0]))
                    slnarr.append(float(lne.split()[1]))
                
                # Get 2D spectra parameters
                Npts = len(frqarr) 
                Ndtm = len(s2dlns[1].split()) - 2
                
                # Get 2D spectra
                s2darr = np.zeros([Npts, Npts, Ndtm], dtype=float)
                data_dtmdly[temp][mix][irun].append(float(s2dlns[0]))
                for lne in s2dlns[1:]:
                    vals = np.array(lne.split(), dtype=float)
                    ii = frqarr.index(vals[0])
                    jj = frqarr.index(vals[1])
                    s2darr[ii, jj, :] = vals[2:]
                    
                # Add results
                data_specfr[temp][mix][irun].append(frqarr)
                data_specln[temp][mix][irun].append(slnarr)
                data_spec2d[temp][mix][irun].append(s2darr)
                
            # Convert to array
            data_specfr[temp][mix][irun] = np.array(
                data_specfr[temp][mix][irun], dtype=float)
            data_specln[temp][mix][irun] = np.array(
                data_specln[temp][mix][irun], dtype=float)
            data_spec2d[temp][mix][irun] = np.array(
                data_spec2d[temp][mix][irun], dtype=float)
        

#---------------------------------
# Evaluation Section
#---------------------------------    

# Create evaluation directory
if not os.path.exists(data_evldir):
    os.mkdir(data_evldir)

# Check INM frequencies
data_refstm = {}
data_refsfr = {}
data_avgdfr = {}
data_stdvfr = {}
data_mintme = np.inf
data_maxtme = 0.0
data_minfrq = np.inf
data_maxfrq = 0.0
for temp in data_params.keys():
    data_refstm[temp] = {}
    data_refsfr[temp] = {}
    data_avgdfr[temp] = {}
    data_stdvfr[temp] = {}
    for mix in data_params[temp].keys():
        data_refstm[temp][mix] = {}
        data_refsfr[temp][mix] = {}
        data_avgdfr[temp][mix] = {}
        data_stdvfr[temp][mix] = {}
        for irun in data_params[temp][mix].keys():
            data_refstm[temp][mix][irun] = {}
            data_refsfr[temp][mix][irun] = {}
            data_avgdfr[temp][mix][irun] = []
            data_stdvfr[temp][mix][irun] = []
            for isample in data_params[temp][mix][irun]:
                
                # Get reference frequency file
                frqfle = os.path.join(
                    data_smpdir, data_frqtag.format(temp, mix, irun, isample))
                
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
                
                # Get minimum maximum values of time array
                mintme, maxtme = np.nanmin(refstm), np.nanmax(refstm)
                if mintme < data_mintme:
                    data_mintme = mintme
                if maxtme > data_maxtme:
                    data_maxtme = maxtme
                if np.any(np.isnan(refstm)):
                    print("NaN in reference file: {:s}".format(frqfle))
                    
                # Get minimum maximum values of frequency array
                minfrq, maxfrq = np.nanmin(refsfr), np.nanmax(refsfr)
                if minfrq < data_minfrq:
                    data_minfrq = minfrq
                if maxfrq > data_maxfrq:
                    data_maxfrq = maxfrq
                if np.any(np.isnan(refsfr)):
                    print("NaN in reference file: {:s}".format(frqfle))
                
                # Assign reference data
                data_refstm[temp][mix][irun][isample] = refstm
                data_refsfr[temp][mix][irun][isample] = refsfr
                data_avgdfr[temp][mix][irun].append(np.mean(refsfr))
                data_stdvfr[temp][mix][irun].append(np.std(refsfr))

# Figure parameters
figsize = (10, 6)
left = 0.15
bottom = 0.15
column = [0.70, 0.00]
row = [0.70, 0.00]

for temp in data_params.keys():
    
    # Create figure and axis system
    cfig = plt.figure(num=0, figsize=figsize)
    caxs = cfig.add_axes([left, bottom, column[0], row[0]])
    cclr = [
        'blue', 'red', 'purple', 'green', 'orange', 'cyan', 
        'magenta', 'darkgreen', 'grey', 'yellow']
    dx = 0.1
    
    for mi, mix in enumerate(data_params[temp].keys()):
        
        for irun in data_params[temp][mix].keys():
        
            for smpi, isample in enumerate(data_params[temp][mix][irun]):
                
                # Plot frequency sequence
                
                # Create figure and axis system
                fig = plt.figure(num=1, figsize=figsize)
                axs = fig.add_axes([left, bottom, column[0], row[0]])
                
                # Plot reference frequencies
                for refsfr in data_refsfr[temp][mix][irun][isample]:
                    axs.plot(data_refstm[temp][mix][irun][isample], refsfr)
                
                # Axis label
                axs.set_title(
                    "Reference frequencies in {:s}, {:s}, {:s}, {:s}".format(
                        temp, mix, irun, isample))
                axs.set_xlabel(r"Time (ps)")
                axs.set_ylabel(r"Frequency (cm$^{-1}$)")
                
                # Axis limits
                axs.set_xlim(data_mintme, data_maxtme)
                axs.set_ylim(data_minfrq, data_maxfrq)
                
                # Save figure
                fig.savefig(
                    os.path.join(data_evldir, data_evlfrq.format(
                        temp, mix, irun, isample)),
                    format='png', dpi=100)
                plt.close(1)
                
                # Plot frequency spectra
                
                # Create figure and axis system
                fig = plt.figure(num=2, figsize=figsize)
                axs = fig.add_axes([left, bottom, column[0], row[0]])
                
                # Bin frequencies
                specfr = data_specfr[temp][mix][irun][smpi]
                specdf = np.diff(specfr)[0]
                specbn = np.array(
                    [
                        specfr[0] - specdf*ifr for ifr in range(1, 100) 
                        if (specfr[0] - specdf*ifr) > (data_minfrq - specdf)
                        ][::-1]
                    + list(specfr)
                    + [
                        specfr[-1] + specdf*ifr for ifr in range(1, 100) 
                        if (specfr[-1] + specdf*ifr) < (data_maxfrq + specdf)
                        ]
                    )
                speccn = specbn[1:] - specdf/2.
                spechf, _ = np.histogram(
                    data_refsfr[temp][mix][irun][isample].reshape(-1),
                    bins=specbn)
                
                # Scale frequency spectra
                nscale = (
                    np.max(data_specln[temp][mix][irun][smpi])/np.max(spechf))
                spechf = np.array(spechf, dtype=float)*nscale
                
                # Plot binned frequency
                axs.plot(speccn, spechf, '-b', label="INM(Ref.)")
                
                # Plot 1D spectra
                axs.plot(
                    specfr, data_specln[temp][mix][irun][smpi], 
                    '-r', label="Spec(Sim.)")
                
                # Axis label
                axs.set_title("1D spectra {:s}, {:s}, {:s}, {:s}".format(
                    temp, mix, irun, isample))
                axs.set_xlabel(r"Frequency (cm$^{-1}$)")
                axs.set_ylabel(r"Intensity")
                
                # Axis limits
                axs.set_xlim(
                    np.min([data_minfrq, np.min(data_specfr[temp][mix][irun])]), 
                    data_maxfrq)
                axs.set_ylim(
                    0.0, np.max(data_specln[temp][mix][irun][smpi])*1.1)
                
                # Add legend
                axs.legend(loc='upper right')
                
                # Save figure
                fig.savefig(
                    os.path.join(data_evldir, data_evlspc.format(
                        temp, mix, irun, isample)),
                    format='png', dpi=100)
                plt.close(2)
                
                # Plot 2DIR spectra samples
                
                # Create figure and axis system
                fig = plt.figure(num=3, figsize=figsize)
                s = figsize[1]/figsize[0]
                axs0 = fig.add_axes([left, bottom, row[0]*s, row[0]])
                axs1 = fig.add_axes([left + row[0]*s, bottom, row[0]*s, row[0]])
                
                # Plot 2D spectra
                fx, fy = np.meshgrid(
                    data_specfr[temp][mix][irun][smpi], 
                    data_specfr[temp][mix][irun][smpi])
                cmax = np.nanmax(
                    np.abs(data_spec2d[temp][mix][irun][smpi, :, :, 0]))
                axs0.contourf(
                    fx, fy, 
                    data_spec2d[temp][mix][irun][smpi, :, :, 0].T, 20,
                    vmin=-cmax, vmax=cmax, cmap="seismic")
                cmax = np.nanmax(
                    np.abs(data_spec2d[temp][mix][irun][smpi, :, :, 15]))
                axs1.contourf(
                    fx, fy, 
                    data_spec2d[temp][mix][irun][smpi, :, :, 15].T, 20,
                    vmin=-cmax, vmax=cmax, cmap="seismic")
                #print(
                    #temp, mix, irun, isample, 
                    #np.max(np.abs(data_spec2d[temp][mix][smpi, :, :, 0])))
                # Axis label
                fig.suptitle("2D spectra {:s}, {:s}, {:s}, {:s}".format(
                    temp, mix, irun, isample))
                axs0.set_title("0ps")
                axs1.set_title("15ps")
                axs0.set_xlabel(r"Pump Frequency (cm$^{-1}$)")
                axs1.set_xlabel(r"Pump Frequency (cm$^{-1}$)")
                axs0.set_ylabel(r"Probe Frequency (cm$^{-1}$)")
                #axs1.set_ylabel(r"Pump Frequency (cm$^{-1}$)")
                axs1.set_yticks([])
                
                # Axis limits
                axs.set_xlim(np.min(fx), np.max(fx))
                axs.set_ylim(np.min(fx), np.max(fx))
                
                # Save figure
                fig.savefig(
                    os.path.join(data_evldir, data_evls2d.format(
                        temp, mix, irun, isample)),
                    format='png', dpi=100)
                plt.close(3)
                
            # Plot frequency mean per sample
            
            # Create figure and axis system
            fig = plt.figure(num=4, figsize=figsize)
            axs = fig.add_axes([left, bottom, column[0], row[0]])
            
            # Plot frequency mean and standard deviation
            axs.errorbar(
                np.arange(len(data_avgdfr[temp][mix][irun])), 
                data_avgdfr[temp][mix][irun],
                yerr=data_stdvfr[temp][mix][irun])
            
            # Axis label
            axs.set_title(
                "Mean reference frequencies in {:s}, {:s}, {:s}".format(
                    temp, mix, irun))
            axs.set_xlabel(r"Sample number")
            axs.set_ylabel(r"Frequency (cm$^{-1}$)")
            
            # Axis limits
            axs.set_ylim(data_minfrq, data_maxfrq)
            
            # Save figure
            fig.savefig(
                os.path.join(data_evldir, data_evlavg.format(
                    temp, mix, irun)),
                format='png', dpi=100)
            plt.close(4)
            
            # Plot frequency mean and standard deviation
            nmix = len(data_avgdfr[temp].keys())
            if mi==0:
                caxs.errorbar(
                    np.arange(
                        len(data_avgdfr[temp][mix][irun])) - dx*(nmix/2. - mi), 
                    data_avgdfr[temp][mix][irun],
                    yerr=data_stdvfr[temp][mix][irun],
                    #color=cclr[mi],
                    #ecolor=cclr[mi],
                    label=mix)
            else:
                caxs.errorbar(
                    np.arange(
                        len(data_avgdfr[temp][mix][irun])) - dx*(nmix/2. - mi), 
                    data_avgdfr[temp][mix][irun],
                    yerr=data_stdvfr[temp][mix][irun],
                    #color=cclr[mi],
                    #ecolor=cclr[mi],
                    label=mix)
            
        # Axis label
        caxs.set_title("Mean reference frequencies in {:s}".format(
            temp))
        caxs.set_xlabel(r"Sample number")
        caxs.set_ylabel(r"Frequency (cm$^{-1}$)")
        
        # Axis limits
        caxs.set_ylim(data_minfrq, data_maxfrq)
        
        # Add legend
        caxs.legend(loc='upper right', ncol=3)
            
        # Save figure
        cfig.savefig(
            os.path.join(data_evldir, data_evlavg.format(
                temp, mix, "all")),
            format='png', dpi=100)
        plt.close(0)

#---------------------------------
# Fitting Section
#---------------------------------    

if not os.path.exists(data_fitdir):
    os.mkdir(data_fitdir)

# Figure parameters
figsize = (10, 6)
left = 0.10
bottom = 0.15
column = [0.70, 0.03]
row = [0.70, 0.00]

# Fit frequency grid
avrg_specfr = {}
avrg_specln = {}
avrg_spec2d = {}
stdv_specfr = {}
stdv_specln = {}
stdv_spec2d = {}
for temp in data_params.keys():
    avrg_specfr[temp] = {}
    avrg_specln[temp] = {}
    avrg_spec2d[temp] = {}
    stdv_specfr[temp] = {}
    stdv_specln[temp] = {}
    stdv_spec2d[temp] = {}
    for mix in data_params[temp].keys():
        print(temp, mix)
        Nspc = 0
        for ir, irun in enumerate(data_params[temp][mix].keys()):
            
            if data_spec2d[temp][mix][irun].shape[0]==0:
                continue
            
            (_, Npts, _, Ndtm) = data_spec2d[temp][mix][irun].shape
            
            for ispc, spln in enumerate(data_specln[temp][mix][irun]):
                
                Nspc += 1
        
        # Reference grid
        grid_specfr = np.mean(
            [
                spci
                for irun in data_params[temp][mix].keys()
                for spci in data_specfr[temp][mix][irun]],
            axis=0)
        grid_specln = np.zeros([Nspc, Npts])
        grid_spec2d = np.zeros([Nspc, Npts, Npts, Ndtm])
        
        ii = 0
        for ir, irun in enumerate(data_params[temp][mix].keys()):
            
            if data_spec2d[temp][mix][irun].shape[0]==0:
                continue
            
            for ispc, spln in enumerate(data_specln[temp][mix][irun]):
                
                # Fit linear spectra to reference grid
                dfri = np.diff(data_specfr[temp][mix][irun][ispc])[0]
                sfri = np.concatenate((
                    [data_specfr[temp][mix][irun][ispc][0] - dfri], 
                    data_specfr[temp][mix][irun][ispc],
                    [data_specfr[temp][mix][irun][ispc][-1] + dfri]))
                slni = np.concatenate(([spln[0]], spln, [spln[-1]]))
                fln = interp1d(sfri, slni, kind='cubic')
                grid_specln[ii, :] = fln(grid_specfr)
                
                # Fit 2D spectra to reference grid
                for idtm in range(Ndtm):
                    f2d = interp2d(
                        data_specfr[temp][mix][irun][ispc], 
                        data_specfr[temp][mix][irun][ispc], 
                        data_spec2d[temp][mix][irun][ispc][:, :, idtm].reshape(
                            -1),
                        kind='cubic')
                    grid_spec2d[ii, :, :, idtm] = f2d(
                        grid_specfr, grid_specfr)
                    
                ii += 1
                
        avrg_specfr[temp][mix] = grid_specfr.copy()
        avrg_specln[temp][mix] = np.mean(grid_specln, axis=0)
        avrg_spec2d[temp][mix] = np.mean(grid_spec2d, axis=0)
        stdv_specfr[temp][mix] = np.std(
            [
                spfri 
                for irun in data_specfr[temp][mix].keys()
                for spfri in data_specfr[temp][mix][irun]
            ],
            axis=0)
        stdv_specln[temp][mix] = np.std(grid_specln, axis=0)
        stdv_spec2d[temp][mix] = np.std(grid_spec2d, axis=0)
        
        # Plot 1D average spectra
        fmax = 0.10
        
        # Bin frequencies
        specfr = avrg_specfr[temp][mix]
        specdf = np.diff(specfr)[0]
        specbn = np.zeros(specfr.shape[0] + 1, dtype=float)
        specbn[:-1] = specfr - specdf/2.
        specbn[-1] = specfr[-1] + specdf/2.
        spechf = np.zeros_like(specfr)
        for irun in data_params[temp][mix].keys():
            for smpi, isample in enumerate(data_params[temp][mix][irun]):
                specdf = np.diff(specfr)[0]
                speccn = specbn[1:] - specdf/2.
                spechf[:] += np.histogram(
                    data_refsfr[temp][mix][irun][isample].reshape(-1),
                    bins=specbn)[0]
            
        # Scale frequency spectra
        nscale = fmax/np.max(spechf)
        spechf = spechf*nscale
        
        # Create figure and axis system
        fig = plt.figure(num=2, figsize=figsize)
        axs = fig.add_axes([left, bottom, column[0], row[0]])
        
        # Plot binned frequency
        axs.plot(speccn, spechf, '-b', label="INM(Ref.)")
        
        # Plot 1D spectra
        axs.plot(
            avrg_specfr[temp][mix], avrg_specln[temp][mix], 
            '-r', label="Spec(Sim.)")
        
        # Axis label
        axs.set_title("1D spectra {:s}K, {:s}%".format(
            temp, mix, isample))
        axs.set_xlabel(r"Frequency (cm$^{-1}$)")
        axs.set_ylabel(r"Intensity")
        
        # Axis limits
        axs.set_xlim(avrg_specfr[temp][mix][0], avrg_specfr[temp][mix][-1])
        axs.set_ylim(0.0, np.max(avrg_specln[temp][mix])*1.1)
        axs.set_ylim(0.0, fmax)
        
        # Add legend
        axs.legend(loc='upper left')
        
        # Save figure
        fig.savefig(
            os.path.join(data_fitdir, data_avgdln.format(
                temp, mix, isample)),
            format='png', dpi=100)
        plt.close(2)
        
        # Plot 2DIR average spectra 
        
        # Create figure and axis system
        fig = plt.figure(num=3, figsize=figsize)
        s = figsize[1]/figsize[0]
        axs0 = fig.add_axes([left, bottom, row[0]*s, row[0]])
        axs1 = fig.add_axes([left + row[0]*s + column[1], bottom, row[0]*s, row[0]])
        
        # Plot 2D spectra
        fx, fy = np.meshgrid(
            avrg_specfr[temp][mix], 
            avrg_specfr[temp][mix])
        cmax = np.nanmax(np.abs(avrg_spec2d[temp][mix][:, :, 0]))
        axs0.contourf(
            fx, fy, 
            avrg_spec2d[temp][mix][:, :, 0], 20,
            vmin=-cmax, vmax=cmax, cmap="seismic")
        cmax = np.nanmax(np.abs(avrg_spec2d[temp][mix][:, :, 15]))
        axs1.contourf(
            fx, fy, 
            avrg_spec2d[temp][mix][:, :, 15], 20,
            vmin=-cmax, vmax=cmax, cmap="seismic")
        # Axis label
        fig.suptitle("Average 2D spectra {:s}K, {:s}%".format(
            temp, mix))
        axs0.set_title("0ps")
        axs1.set_title("15ps")
        axs0.set_xlabel(r"Pump Frequency (cm$^{-1}$)")
        axs1.set_xlabel(r"Pump Frequency (cm$^{-1}$)")
        axs0.set_ylabel(r"Probe Frequency (cm$^{-1}$)")
        #axs1.set_ylabel(r"Pump Frequency (cm$^{-1}$)")
        axs1.set_yticks([])
        
        # Axis limits
        axs0.set_xlim(np.min(fx), np.max(fx))
        axs0.set_ylim(np.min(fx), np.max(fx))
        axs1.set_xlim(np.min(fx), np.max(fx))
        axs1.set_ylim(np.min(fx), np.max(fx))
        
        # Save figure
        fig.savefig(
            os.path.join(data_fitdir, data_avgd2d.format(
                temp, mix, isample)),
            format='png', dpi=100)
        plt.close(3)
        
        # Write average results to file
        
        # 1D spectra
        np.savetxt(
            os.path.join(data_fitdir, data_aslnfl.format(mix)),
            np.stack((avrg_specfr[temp][mix], avrg_specln[temp][mix])).T,
            fmt=['%5.1f', '%8.4f'])
        
        # 2D spectra
        dtmdly = np.mean(
            [
                tmdli
                for irun in data_dtmdly[temp][mix].keys()
                for tmdli in data_dtmdly[temp][mix][irun]
            ])
        outs2d = "{:4.1f}\n".format(dtmdly)
        for ifx, fxi in enumerate(avrg_specfr[temp][mix]):
            for ify, fyi in enumerate(avrg_specfr[temp][mix]):
                outs2d += "{:4.1f} {:4.1f} ".format(fxi, fyi)
                outs2d += ("{: 7.5f} "*avrg_spec2d[temp][mix].shape[2]).format(
                    *avrg_spec2d[temp][mix][ifx, ify, :])
                outs2d += "\n"
        with open(os.path.join(data_fitdir, data_as2dfl.format(mix)), 'w') as f:
            f.write(outs2d[:-1])
                
        
        
        
