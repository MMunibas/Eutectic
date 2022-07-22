#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=20
#SBATCH --mem-per-cpu=1000

python Script_evaluate_rdf.py
python Script_evaluate_cnum.py
python Script_evaluate_THz.py
python Script_evaluate_radang.py

# We succeeded, reset trap and clean up normally.
trap - EXIT
exit 0
