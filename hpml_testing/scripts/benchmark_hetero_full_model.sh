#!/bin/bash
# Benchmark heterogeneous ring attention on full Llama model
# Launches ranks manually to apply different MPS throttling per GPU

set -e

module load conda/latest
conda activate context_parallelism
module load cuda/12.6

# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
# export TORCH_DISTRIBUTED_DEBUG=DETAIL
# MPS Setup (for GPU throttling)
export CUDA_MPS_PIPE_DIRECTORY="${CUDA_MPS_PIPE_DIRECTORY:-/tmp/mps_$USER}"
export CUDA_MPS_LOG_DIRECTORY="${CUDA_MPS_LOG_DIRECTORY:-/tmp/mps_$USER}"
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"

# Start MPS daemon if not already running
if ! pgrep -u "$USER" -f "nvidia-cuda-mps-control" >/dev/null 2>&1; then
    echo "Starting MPS daemon..."
    nvidia-cuda-mps-control -d
    sleep 2
fi
echo "MPS directories: $CUDA_MPS_PIPE_DIRECTORY"

# Configuration
MODEL_PATH="/datasets/ai/llama3/hub/models--meta-llama--Llama-3.2-1B/snapshots/4e20de362430cd3b72f300e6b0f18e50e7166e08"
WORLD_SIZE=2
DTYPE="bfloat16"
NUM_DECODE=0
OUTPUT_CSV="hetero_full_model_results.csv"

# MPS settings for heterogeneous GPUs (GPU0=100%, GPU1=throttled)
MPS_FAST=100   # GPU0 at full speed
MPS_SLOW=50    # GPU1 throttled to 50%
RANK_MPS="${MPS_FAST},${MPS_SLOW}"
SLOWDOWN_FACTOR=$(awk "BEGIN {print $MPS_SLOW/$MPS_FAST}")

# Sequence lengths to test 
SEQ_LENGTHS=(4096 8192 16384)

# Split strategies to compare
# Note: "lut" requires a performance profile CSV - skipping if not available
SPLIT_TYPES=("even" "uneven" "formula")

# Performance profile for LUT-based split
PERF_PROFILE="hpml_testing/results/matmul_mps_sweep.csv"

# Distributed setup
MASTER_ADDR="localhost"
MASTER_PORT=29500

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
export PYTHONPATH="$SCRIPT_DIR/../:$PYTHONPATH"

echo "Heterogeneous Full Model Benchmark"
echo "Model: $MODEL_PATH"
echo "World size: $WORLD_SIZE"
echo "MPS config: GPU0=${MPS_FAST}%, GPU1=${MPS_SLOW}%"

for SEQ_LEN in "${SEQ_LENGTHS[@]}"; do
    for SPLIT in "${SPLIT_TYPES[@]}"; do
        echo ""
        echo ">>> Testing: seq_len=$SEQ_LEN, split=$SPLIT"
        echo "-------------------------------------------"

        # Use random port to avoid conflicts
        MASTER_PORT=$((29500 + RANDOM % 1000))

        # Build common args
        COMMON_ARGS="--model_path $MODEL_PATH \
            --num_tokens $SEQ_LEN \
            --num_decode_tokens $NUM_DECODE \
            --split-type $SPLIT \
            --rank-mps $RANK_MPS \
            --slowdown-factor $SLOWDOWN_FACTOR \
            --dtype $DTYPE \
            --summary_csv $OUTPUT_CSV"

        if [ "$SPLIT" == "lut" ]; then
            COMMON_ARGS="$COMMON_ARGS --use-perf-profile $PERF_PROFILE"
        fi

        # Launch Rank 0 (GPU0 at full speed)
        CUDA_VISIBLE_DEVICES=0 \
        MASTER_ADDR=$MASTER_ADDR \
        MASTER_PORT=$MASTER_PORT \
        RANK=0 \
        LOCAL_RANK=0 \
        WORLD_SIZE=$WORLD_SIZE \
        python benchmarks/hetero_full_model.py $COMMON_ARGS &
        PID_RANK0=$!

        # Launch Rank 1 (GPU1 throttled via MPS)
        CUDA_VISIBLE_DEVICES=1 \
        CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$MPS_SLOW \
        MASTER_ADDR=$MASTER_ADDR \
        MASTER_PORT=$MASTER_PORT \
        RANK=1 \
        LOCAL_RANK=0 \
        WORLD_SIZE=$WORLD_SIZE \
        python benchmarks/hetero_full_model.py $COMMON_ARGS &
        PID_RANK1=$!

        # Wait for both ranks
        wait $PID_RANK0
        wait $PID_RANK1

        echo "-------------------------------------------"
        sleep 2
    done
done

echo ""
echo "Results saved to: $OUTPUT_CSV"
echo "Done!"
