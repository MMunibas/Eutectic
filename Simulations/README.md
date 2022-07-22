===============================================================================
Author:
Kai Töpfer (2022), kai.toepfer@unibas.ch

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

===============================================================================

This compilation of Scripts and files are required to reproduce the theoretical
results reported in the publication https://doi.org/10.1021/jacs.2c04169 .

# Requirements

The Simulation are performed with the CHARMM program package c47a2.

The Python Scripts requires Python 3.6 or higher with following dependencies:

- numpy 
- scipy
- Matplotlib
- Atomic Simulation Environment (3.20 and higher)
- MDAnalysis 1.0

To generate the CHARMM input files, the 'packmol' program 
(http://leandro.iqm.unicamp.br/m3g/packmol/home.shtml) 
is required and  the executable must be added to the $PATH environmental 
variable.

To run the CHARMM simulations, the template/run_temp.sh has to be adopted to
to load the specific modules and link to the CHARMM executable of your system.

The C Script to compute the 2D IR spectra in '2DIR/templates' written by Prof. 
Peter Hamm requires a C compiler and the Python Script starting the 
compilation is set to use the gcc compiler.

# Instructions

## CHARMM Simulations

The Python Script 'Script_start.py' generates working directories with the 
input files to run CHARMM simulations of different ratios of KSCN, acetamide 
and water. To run the simulation on system with the 'slurm' management system,
you can execute the generated 'start.sh' script in the main directory.
It iterates over all the working directories and start the Python Scripts
'observe.py' and 'inm.py' in the background. You can also go to the working 
directories and start the simulations manually.

'observe.py' is a convenience tool that start the CHARMM simulation 
('sbatch run.sh') and observe the respective slurm file for error messages to
cancel and restart the job if necessary.

'inm.py' runs in the background and start the Instantaneous Normal Mode (INM)
analysis of all SCN anion from the requested trajectories if the respective 
dcd trajectory files are written completely.

Further in the background can run the Python Script 'Script_prep2DIR.py' that 
frequently check the working directories for finished INM analysis and 
prepares the input files for the 2D IR computation and store the in the 
generated directory 'results_samples'.


## 2D IR computation

The directory '2DIR' contains all files to compute the 2D IR spectra of the
CN stretch vibration in SCN^- in the respective simulation environment.
It requires the INM results in the 'results_samples' directory.

To run the computation of the 2D IR spectra you have to run 'sbatch run_2dir.sh'
that will run the Python script 'Script_run_2DIR.py'. You have to adjust the
shell script to load the gcc compiler if necessary.

The Python script 'Script_run_2DIR.py' will repeatedly check the 
'results_samples' directory for new input files (make sure to put the correct 
path to 'results_samples' in the Python script) and create a working directory 
with the compiled 2DIR script and the input files and starts the computation.

When complete or even before, you can run the Python script 
'Script_eval_samples.py' to evaluate the 2D IR computations and get an 
averaged result of trajectories of the same system compilation. Also adopt
the path to the input files in this script.

The files with the averaged results in 'fitted_results_samples' are plotted
by the Script to evaluate the experimental results.


## Evaluation

When all CHARMM simulations are done, the evaluation of the results to 
reproduce the figures in the publication.
To run over all evaluation scripts, you can execute 'run_evaluation.sh' via
'sbatch run_evaluation.sh'. Note that the current configuration is for one
node with 20 CPUs. 

The radial distribution function are computed by the Python script
'Script_evaluate_rdf.py'. The current configuration (see parameter 'tasks') runs
20 parallel tasks to compute the requested distances and compute the histogram
for the subsequent computation of the radial distribution functions.
This script can run for a long time.

To compute the THz spectra from the trajectories, you have to run first the 
Python Script 'Script_evaluate_cnum.py' that computes the coordination number
of water cluster for different mixing ratios. Some of these results are shown
together with the THz spectra.

The Python Script 'Script_evaluate_THz.py' compute the THz spectra,  
partial spectra and cross correlations and plot these results. For one
plot, the average coordination number of water clusters is computed for the 
results of the 'Script_evaluate_cnum.py' script. This could cause errors if the 
given path is not correct.

The Python Script 'Script_evaluate_radang.py' computes the radial-angular 
distribution between SCN^- anions and generates figure fragments to 
build the radial-angular distribution plot from the publication.
For the publication, further snapshots of SCN^- anion pairs at different 
conformations are rendered by VMD. All pieces were finally put together 
with the gimp program 2.10. The order of the high density points which are
shown in plot can change with different trajectories. The one that are shown
are specified in the Script at line 682.


## Published Results

In the directory 'published_results' are the python Scripts to generate
the figures for the radial distribution functions and the THz spectra.
The respective subdirectories contains files with the final results and part of 
the Scripts are commented out, that the execution of the Scripts just perform 
the final part generating the plots. This could not be provided for the 
radial-angular distribution because of the size of the files with the results
of several tenth of GB. However the gimp .xcf file of the final figure is 
provided.
