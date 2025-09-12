#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=110:00:00
#SBATCH --mem=16G 
#SBATCH --ntasks=1
#SBATCH --job-name=dsc
#SBATCH --output=dsc.%j.out

export PYTHONUNBUFFERED=1
module purge

#module load Python/3.11.3-GCCcore-12.3.0
source /projects/disruptsc/miniforge3/bin/activate /projects/disruptsc/miniforge3/envs/rewiring
python /projects/disruptsc/rewiring/init_ntw_launcher.py
