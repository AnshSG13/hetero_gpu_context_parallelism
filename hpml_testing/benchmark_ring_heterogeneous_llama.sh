#!/bin/bash
set -e
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT

# Hetero MPS percentages (weak vs strong GPU)
STRONG_GPU_PERCENTAGE=${STRONG_GPU_PERCENTAGE:-10}
WEAK_GPU_PERCENTAGE=${WEAK_GPU_PERCENTAGE:-100}
SEQ_LEN=${SEQ_LEN:-4096}
# 512 1024 4096 8192 16384 32768 65536

# LLaMA model config — adjust to your paths
NUM_DECODE_TOKENS=${NUM_DECODE_TOKENS:-0}
BATCH_SIZE=${BATCH_SIZE:-1}
ARCH="llama"
VARIANT="3.2-1b"
MODEL_PATH="/workspace/.hf_home/hub/models--meta-llama--Llama-3.2-1B/snapshots/4e20de362430cd3b72f300e6b0f18e50e7166e08"   # <-- change if needed

export WORLD_SIZE=2
export MASTER_ADDR='localhost'
export MASTER_PORT='29500'

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
export PYTHONPATH="$SCRIPT_DIR/../:$PYTHONPATH"

PYTHON="python3"
BENCH="$SCRIPT_DIR/benchmark_uneven_split.py"

echo "Using STRONG_GPU_PERCENTAGE=$STRONG_GPU_PERCENTAGE, WEAK_GPU_PERCENTAGE=$WEAK_GPU_PERCENTAGE"
echo "SEQ_LEN=$SEQ_LEN, NUM_DECODE_TOKENS=$NUM_DECODE_TOKENS"
echo

run_mode () {
    local shard_mode=$1

    echo "--------------------- ${shard_mode^^} sharding (hetero) ---------------------"
    echo "Starting benchmark with total sequence length: $SEQ_LEN"
    echo "Rank 0 MPS: $STRONG_GPU_PERCENTAGE%"
    echo "Rank 1 MPS: $WEAK_GPU_PERCENTAGE%"
    echo "--------------------------------------------------------------------"

    # Rank 0 (e.g. weak GPU)
    export RANK=0
    export LOCAL_RANK=0
    SHARD_MODE=$shard_mode CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$STRONG_GPU_PERCENTAGE \
    $PYTHON "$BENCH" \
        --device_type cuda \
        --architecture "$ARCH" \
        --variant "$VARIANT" \
        --model_path "$MODEL_PATH" \
        --batch_size $BATCH_SIZE \
        --num_tokens $SEQ_LEN \
        --num_decode_tokens $NUM_DECODE_TOKENS \
        # --disable_flash \
        --summary_csv "$SCRIPT_DIR/llama_hetero_${shard_mode}.csv" &

    # Rank 1 (strong GPU)
    export RANK=1
    export LOCAL_RANK=1
    SHARD_MODE=$shard_mode CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$WEAK_GPU_PERCENTAGE \
    $PYTHON "$BENCH" \
        --device_type cuda \
        --architecture "$ARCH" \
        --variant "$VARIANT" \
        --model_path "$MODEL_PATH" \
        --batch_size $BATCH_SIZE \
        --num_tokens $SEQ_LEN \
        --num_decode_tokens $NUM_DECODE_TOKENS \
        # --disable_flash \
        --summary_csv "$SCRIPT_DIR/llama_hetero_${shard_mode}.csv" &

    wait
    echo "--------------------------------------------------------------------"
    echo "Benchmark ${shard_mode} (hetero) finished."
    echo
}

# 1) even sharding with hetero GPUs
run_mode "even"

# 2) proportional sharding with same hetero GPUs
run_mode "proportional"