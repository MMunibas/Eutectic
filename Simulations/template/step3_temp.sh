#!/bin/bash
#SBATCH --job-name=JBNM
#SBATCH --partition=vshort
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=1400
#SBATCH --exclude=node[41-53,109-124]

#---------
# Modules
#---------

module load intel/2019-compiler-intel-mpi
my_charmm="/home/toepfer/Programs/intel_c45a2/build/cmake/charmm"
ulimit -s 10420

#------------
# Run CHARMM
#------------

srun $my_charmm -i INPF -o OUTF

#----------------
# Run Evaluation
#----------------

python EVAF OUTF RESF FLGF

rm -f INPF EVAF OUTF
