#!/bin/bash
#SBATCH --job-name=2DIR
#SBATCH --partition=infinite
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --mem-per-cpu=1400
#SBATCH --exclude=node[65-74]

module load gcc/gcc-9.2.0-openmpi-3.1.4

python Script_run_2DIR.py 

# We succeeded, reset trap and clean up normally.
trap - EXIT
exit 0
