#!/bin/bash

#SBATCH -N 1  # number of nodes
#SBATCH -n 1  # number of "tasks" (default: allocates 1 core per task)
#SBATCH -t 0-01:00:00   # time in d-hh:mm:ss
#SBATCH -p publicgpu    # partition
#SBATCH -q wildfire     # QOS
#SBATCH --gres=gpu:2    # Request two GPUs
#SBATCH -o slurm.%j.out # file to save job's STDOUT (%j = JobId)
#SBATCH -e slurm.%j.err # file to save job's STDERR (%j = JobId)
#SBATCH --mail-type=ALL # Send an e-mail when a job starts, stops, or fails
#SBATCH --mail-user=%u@asu.edu # Mail-to address

# Always purge modules to ensure consistent environments
module purge
# Load required modules for job's environment
module load anaconda/py3
# Activating conda env                                                          
source activate keras-gpu-2.4.3
# checking for GPU
nvidia-smi
# Run python script
srun python deepweed_agave.py
