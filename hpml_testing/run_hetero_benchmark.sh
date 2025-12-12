#!/bin/bash

# --- Parameters ---
SEQ_LEN=${SWEEP_SEQ_LEN:-8192}
EMB_DIM=4096
N_HEADS=32
WORLD_SIZE=2
SLOWDOWN_PERCENTAGE=${SWEEP_SLOWDOWN_PCT:-50}
SLOWDOWN_FACTOR=$(echo "$SLOWDOWN_PERCENTAGE / 100" | bc -l)

set -e 
export PYTHONPATH=$(pwd):$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1

echo "=========================================================="
echo "  Running Experiment 1: EVEN split with 1 slow GPU"
echo "=========================================================="

python3 hpml_testing/benchmark_hetero_latency.py \
    --rank 0 \
    --world-size $WORLD_SIZE \
    --seq-len $SEQ_LEN \
    --emb-dim $EMB_DIM \
    --n-heads $N_HEADS \
    --split-type even & \
PID_RANK_0=$!

CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$SLOWDOWN_PERCENTAGE python3 hpml_testing/benchmark_hetero_latency.py \
    --rank 1 \
    --world-size $WORLD_SIZE \
    --seq-len $SEQ_LEN \
    --emb-dim $EMB_DIM \
    --n-heads $N_HEADS \
    --split-type even & \
PID_RANK_1=$!

wait $PID_RANK_0
wait $PID_RANK_1

echo "Experiment 1 (Even Split) finished."
echo ""
sleep 2 

echo "=========================================================="
echo "  Running Experiment 2: UNEVEN split with 1 slow GPU"
echo "=========================================================="

python3 hpml_testing/benchmark_hetero_latency.py \
    --rank 0 \
    --world-size $WORLD_SIZE \
    --seq-len $SEQ_LEN \
    --emb-dim $EMB_DIM \
    --n-heads $N_HEADS \
    --split-type uneven \
    --slowdown-factor $SLOWDOWN_FACTOR & \
PID_RANK_0=$!

CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$SLOWDOWN_PERCENTAGE python3 hpml_testing/benchmark_hetero_latency.py \
    --rank 1 \
    --world-size $WORLD_SIZE \
    --seq-len $SEQ_LEN \
    --emb-dim $EMB_DIM \
    --n-heads $N_HEADS \
    --split-type uneven \
    --slowdown-factor $SLOWDOWN_FACTOR & \
PID_RANK_1=$!

wait $PID_RANK_0
wait $PID_RANK_1

echo "Experiment 2 (Uneven Split) finished."
echo ""

echo "=========================================================="
echo "  Running Experiment 3: LUT-based split with 1 slow GPU"
echo "=========================================================="

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

wait $PID_RANK_0
wait $PID_RANK_1

echo "Experiment 3 (LUT-based Split) finished."
echo ""
