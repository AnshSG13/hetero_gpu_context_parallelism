#!/bin/bash

# This script runs a profiling comparison for the three different
# rebalancing strategies at a fixed sequence length and slowdown.
# It generates three separate PyTorch Profiler trace files for analysis.

# --- Parameters ---
SLOWDOWN_PERCENTAGE=50
SLOWDOWN_FACTOR=$(awk "BEGIN {print $SLOWDOWN_PERCENTAGE/100}")
RESULTS_DIR="hpml_testing/results"

# --- Script setup ---
set -e
export PYTHONPATH=$(pwd):$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1

# Create results directory if it doesn't exist
mkdir -p $RESULTS_DIR

echo "This script will generate three profiler trace files for a fixed 8k sequence length."
echo "Assumes the CUDA MPS daemon is already running."
sleep 2

# --- 1. Profile Even Split (Baseline Heterogeneous) ---
echo "=========================================================="
echo "  Profiling: EVEN split with 1 slow GPU"
echo "=========================================================="
python3 hpml_testing/profile_rebalancing.py \
    --rank 0 \
    --world-size 2 \
    --split-type even \
    --output-file "$RESULTS_DIR/profile_even.json" &
PID_RANK_0=$!

CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$SLOWDOWN_PERCENTAGE python3 hpml_testing/profile_rebalancing.py \
    --rank 1 \
    --world-size 2 \
    --split-type even \
    --output-file "$RESULTS_DIR/profile_even.json" &
PID_RANK_1=$!

wait $PID_RANK_0
wait $PID_RANK_1
echo "Finished profiling even split."
echo ""

# --- 2. Profile Uneven Split (Theoretical) ---
echo "=========================================================="
echo "  Profiling: UNEVEN split with 1 slow GPU"
echo "=========================================================="
python3 hpml_testing/profile_rebalancing.py \
    --rank 0 \
    --world-size 2 \
    --split-type uneven \
    --slowdown-factor $SLOWDOWN_FACTOR \
    --output-file "$RESULTS_DIR/profile_uneven.json" &
PID_RANK_0=$!

CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$SLOWDOWN_PERCENTAGE python3 hpml_testing/profile_rebalancing.py \
    --rank 1 \
    --world-size 2 \
    --split-type uneven \
    --slowdown-factor $SLOWDOWN_FACTOR \
    --output-file "$RESULTS_DIR/profile_uneven.json" &
PID_RANK_1=$!

wait $PID_RANK_0
wait $PID_RANK_1
echo "Finished profiling uneven split."
echo ""

# --- 3. Profile LUT Split (Empirical) ---
echo "=========================================================="
echo "  Profiling: LUT-based split with 1 slow GPU"
echo "=========================================================="
python3 hpml_testing/profile_rebalancing.py \
    --rank 0 \
    --world-size 2 \
    --split-type lut \
    --rank-mps "100,$SLOWDOWN_PERCENTAGE" \
    --output-file "$RESULTS_DIR/profile_lut.json" &
PID_RANK_0=$!

CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$SLOWDOWN_PERCENTAGE python3 hpml_testing/profile_rebalancing.py \
    --rank 1 \
    --world-size 2 \
    --split-type lut \
    --rank-mps "100,$SLOWDOWN_PERCENTAGE" \
    --output-file "$RESULTS_DIR/profile_lut.json" &
PID_RANK_1=$!

wait $PID_RANK_0
wait $PID_RANK_1
echo "Finished profiling LUT-based split."
echo ""

echo "Profiling complete. Traces saved to $RESULTS_DIR/"
echo "You can now analyze these traces using a tool like TensorBoard:"
echo "pip install tensorboard"
echo "tensorboard --logdir $RESULTS_DIR"
