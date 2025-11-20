#!/bin/bash

#SBATCH -p gpu                  # Partition (queue)
#SBATCH -t 0:15:00              # Time limit: 15 minutes (very low GPU time)
#SBATCH --gpus=2                # Number of GPUs (need 2 for communication test)
#SBATCH --constraint=a100       # Request specific GPU type
#SBATCH --mem=16G               # Memory per node
#SBATCH -o slurm-%j.out         # Standard output log
#SBATCH -e slurm-%j.err         # Standard error log
#SBATCH --mail-type=BEGIN,END,FAIL  # Email notification
#SBATCH --job-name=gpu_comm_test    # Job name

# Load required modules
module load conda/latest
module load cuda/12.6
module load cudnn/8.9.7.29-12-cuda12.6

# Activate conda environment
conda activate changemamba

# Run the GPU communication test with 2 GPUs
torchrun --nproc_per_node=2 test_gpu_communication.py
