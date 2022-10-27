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
source ./setupenv/scorep_GCC113/scoreprc #adds scorep to path
module load
source venv #already with scorep-python-bindings
exit

####
# Fail on any error.
set -e

DATETIME=$(date "+%Y%m%d-%H%M%S")
PRAEFIX=${SLURM_JOB_ID}_${DATETIME}

export SCOREP_ENABLE_PROFILING=false
export SCOREP_ENABLE_TRACING=true
export SCOREP_CUDA_ENABLE=yes
export SCOREP_CUDA_BUFFER=60000000
export SCOREP_TOTAL_MEMORY=4G
export SCOREP_EXPERIMENT_DIRECTORY=./RUNS/${PRAEFIX}_${SLURM_JOB_NAME}_scorep


commitID=$(git rev-parse --short HEAD)
gitdescript=$(git describe --abbrev=4 HEAD)
echo "Run Commit ${commitID} : ${gitdescript}"

echo 'Training Begin at'
date

# Display commands being run.
set -x

srun python -m scorep --cuda --mpp=mpi ./main_DDP_minimal.py

echo 'Training Ended at'
date
