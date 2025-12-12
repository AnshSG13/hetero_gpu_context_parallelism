#!/bin/bash

# This script launches the ring attention split comparison benchmark.
# It simulates a heterogeneous environment using CUDA MPS.

# --- Configuration ---
# You can edit these default values or pass them as arguments to the script.

# GPUs to use, e.g., "0,1" for two GPUs.
GPUS="0,1"

# Corresponding MPS percentages for each GPU.
# This example makes GPU 1 half the speed of GPU 0.
MPS_PERCENTAGES="100,50"

# --- Script Logic ---

# Allow overriding defaults with command-line arguments
# Usage: ./run_compare_ring_attn_splits.sh [GPUS] [MPS_PERCENTAGES]
# Example: ./run_compare_ring_attn_splits.sh "0,1,2,3" "100,75,50,25"

if [ ! -z "$1" ]; then
    GPUS=$1
fi

if [ ! -z "$2" ]; then
    MPS_PERCENTAGES=$2
fi

# Determine the number of GPUs from the comma-separated list
IFS=',' read -ra ADDR <<< "$GPUS"
NPROC_PER_NODE=${#ADDR[@]}

echo "Running benchmark on GPUs: $GPUS"
echo "With MPS percentages: $MPS_PERCENTAGES"
echo "Number of processes: $NPROC_PER_NODE"
echo "--------------------------------------------------"

# Set the environment variables and launch the benchmark with torchrun
CUDA_VISIBLE_DEVICES=$GPUS \
CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$MPS_PERCENTAGES \
torchrun --nproc_per_node=$NPROC_PER_NODE "$(dirname "$0")/compare_ring_attn_splits.py"
