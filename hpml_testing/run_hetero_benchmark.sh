#!/bin/bash

# This script runs a benchmark to compare ring attention latency with
# even and uneven workload splitting on two GPUs, where one GPU is
# artificially slowed down using CUDA MPS.

# --- Parameters ---
SEQ_LEN=8192
EMB_DIM=4096
N_HEADS=32
WORLD_SIZE=2
# Percentage of threads the slow GPU can use. 50 means it's 50% as fast.
# This simulates a GPU with half the compute capability.
SLOWDOWN_PERCENTAGE=50
# The slowdown factor used in the python script must match the percentage.
# This is slowdown_percentage / 100
SLOWDOWN_FACTOR=0.5

# --- Script setup ---
set -e # Exit immediately if a command exits with a non-zero status.

# Add the project root to the python path to allow imports of 'fms'
export PYTHONPATH=$(pwd):$PYTHONPATH


# Specify which GPUs to use
export CUDA_VISIBLE_DEVICES=0,1

echo "Starting CUDA MPS daemon..."
# The user might need to run this script with sudo for this to work
nvidia-cuda-mps-control -d
if [ $? -ne 0 ]; then
    echo "Failed to start MPS daemon. You may need to run this script with sudo."
    exit 1
fi
echo "MPS daemon started."

# --- Experiment 1: Even Split (Baseline) ---

echo "=========================================================="
echo "  Running Experiment 1: EVEN split with 1 slow GPU"
echo "=========================================================="

# Launch the two python processes in the background.
# Rank 1 will be the "slow" GPU.
# We apply the MPS slowdown factor ONLY to the rank 1 process.

# Rank 0 (Fast GPU)
python3 hpml_testing/benchmark_hetero_latency.py \
    --rank 0 \
    --world-size $WORLD_SIZE \
    --seq-len $SEQ_LEN \
    --emb-dim $EMB_DIM \
    --n-heads $N_HEADS \
    --split-type even & \
PID_RANK_0=$!

# Rank 1 (Slow GPU) - Apply MPS slowdown
CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$SLOWDOWN_PERCENTAGE python3 hpml_testing/benchmark_hetero_latency.py \
    --rank 1 \
    --world-size $WORLD_SIZE \
    --seq-len $SEQ_LEN \
    --emb-dim $EMB_DIM \
    --n-heads $N_HEADS \
    --split-type even & \
PID_RANK_1=$!

# Wait for both processes to complete
wait $PID_RANK_0
wait $PID_RANK_1

echo "Experiment 1 (Even Split) finished."
echo ""
sleep 2 # Give a moment for things to settle

# --- Experiment 2: Uneven Split (Balanced) ---

echo "=========================================================="
echo "  Running Experiment 2: UNEVEN split with 1 slow GPU"
echo "=========================================================="

# Launch again, but this time with the "uneven" split type.
# The python script will calculate the appropriate uneven split based
# on the slowdown factor.

# Rank 0 (Fast GPU)
python3 hpml_testing/benchmark_hetero_latency.py \
    --rank 0 \
    --world-size $WORLD_SIZE \
    --seq-len $SEQ_LEN \
    --emb-dim $EMB_DIM \
    --n-heads $N_HEADS \
    --split-type uneven \
    --slowdown-factor $SLOWDOWN_FACTOR & \
PID_RANK_0=$!

# Rank 1 (Slow GPU) - Apply MPS slowdown
CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$SLOWDOWN_PERCENTAGE python3 hpml_testing/benchmark_hetero_latency.py \
    --rank 1 \
    --world-size $WORLD_SIZE \
    --seq-len $SEQ_LEN \
    --emb-dim $EMB_DIM \
    --n-heads $N_HEADS \
    --split-type uneven \
    --slowdown-factor $SLOWDOWN_FACTOR & \
PID_RANK_1=$!

# Wait for both processes to complete
wait $PID_RANK_0
wait $PID_RANK_1

echo "Experiment 2 (Uneven Split) finished."
echo ""

# --- Experiment 3: LUT-based Uneven Split (Refined) ---

echo "=========================================================="
echo "  Running Experiment 3: LUT-based split with 1 slow GPU"
echo "=========================================================="

# Launch again, this time with the "lut" split type.
# The python script will use the performance profile to calculate
# the refined uneven split.

# Rank 0 (Fast GPU)
python3 hpml_testing/benchmark_hetero_latency.py \
    --rank 0 \
    --world-size $WORLD_SIZE \
    --seq-len $SEQ_LEN \
    --emb-dim $EMB_DIM \
    --n-heads $N_HEADS \
    --split-type lut \
    --use-perf-profile hpml_testing/results/matmul_mps_sweep.csv \
    --rank-mps "100,${SLOWDOWN_PERCENTAGE}" & \
PID_RANK_0=$!

# Rank 1 (Slow GPU) - Apply MPS slowdown
CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$SLOWDOWN_PERCENTAGE python3 hpml_testing/benchmark_hetero_latency.py \
    --rank 1 \
    --world-size $WORLD_SIZE \
    --seq-len $SEQ_LEN \
    --emb-dim $EMB_DIM \
    --n-heads $N_HEADS \
    --split-type lut \
    --use-perf-profile hpml_testing/results/matmul_mps_sweep.csv \
    --rank-mps "100,${SLOWDOWN_PERCENTAGE}" & \
PID_RANK_1=$!

# Wait for both processes to complete
wait $PID_RANK_0
wait $PID_RANK_1

echo "Experiment 3 (LUT-based Split) finished."
echo ""


# --- Cleanup ---
echo "Stopping CUDA MPS daemon..."
echo "quit" | nvidia-cuda-mps-control
echo "Benchmark complete."
