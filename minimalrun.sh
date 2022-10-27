#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH -J Name
#SBATCH --output=R-%x_%j.log
#SBATCH --gres=gpu:8
#SBATCH -p alpha
#SBATCH --mem=0
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=6
#SBATCH --gpus-per-task=1 
#SBATCH --constraint=fs_beegfs
#SBATCH --hint=nomultithread

###
module load
source venv
exit

####
# Fail on any error.
set -e

DATETIME=$(date "+%Y%m%d-%H%M%S")
PRAEFIX=${SLURM_JOB_ID}_${DATETIME}



commitID=$(git rev-parse --short HEAD)
gitdescript=$(git describe --abbrev=4 HEAD)
echo "Run Commit ${commitID} : ${gitdescript}"

echo 'Training Begin at'
date

# Display commands being run.
set -x

srun python -u ./main_DDP_minimal.py

echo 'Training Ended at'
date
