#!/bin/bash
#SBATCH -p gpu-preempt
#SBATCH -t 0:45:00
#SBATCH --gpus-per-node=2
#SBATCH --constraint=a100
#SBATCH --mem=26G
#SBATCH -o ../logs/slurm-%j.out
#SBATCH -e ../logs/slurm-%j.err
#SBATCH --job-name=ring_vs_regular
#SBATCH --nodes=1

set -e

MODEL_PATH="/datasets/ai/llama3/hub/models--meta-llama--Llama-3.2-1B/snapshots/4e20de362430cd3b72f300e6b0f18e50e7166e08"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="benchmark_results"
NUM_GPUS=2

module load conda/latest
conda activate context_parallelism
module load cuda/12.6

#export NCCL_DEBUG=INFO

SUMMARY_CSV="$RESULTS_DIR/summary_${TIMESTAMP}.csv"
LOG_FILE="$RESULTS_DIR/run_${TIMESTAMP}.log"

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
export PYTHONPATH="$SCRIPT_DIR/../:$PYTHONPATH"

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi
mkdir -p "$RESULTS_DIR"

# Check NVLink topology
nvidia-smi topo -m

# Cleanup function to kill any lingering processes
cleanup_processes() {
    pkill -9 -f "torchrun.*benchmark_ring.py" 2>/dev/null || true
    pkill -9 -f "benchmark_ring.py" 2>/dev/null || true
    sleep 2
}

# Run ring attention benchmark
run_ring_benchmark() {
    local num_tokens=$1
    local num_gpus=$2

    # Cleanup before running
    cleanup_processes

    # Use random port to avoid conflicts
    local master_port=$((29500 + RANDOM % 1000))

    echo "[RING] tokens=$num_tokens gpus=$num_gpus"
    MASTER_PORT=$master_port PYTHONPATH="$SCRIPT_DIR/../:$PYTHONPATH" torchrun \
        --nproc_per_node=$num_gpus \
        --master_port=$master_port \
        benchmark_ring.py \
        --architecture hf_pretrained \
        --model_path "$MODEL_PATH" \
        --device_type cuda \
        --num_tokens $num_tokens \
        --num_decode_tokens 0 \
        --batch_size 1 \
        --run_ring_first \
        --summary_csv "$SUMMARY_CSV"

    # Cleanup after running
    cleanup_processes
}

# Run regular attention benchmark (single GPU)
run_regular_benchmark() {
    local num_tokens=$1

    # Cleanup before running
    cleanup_processes

    echo "[REGULAR] tokens=$num_tokens gpu=0"
    CUDA_VISIBLE_DEVICES=0 python benchmark_regular.py \
        --architecture hf_pretrained \
        --model_path "$MODEL_PATH" \
        --device_type cuda \
        --num_tokens $num_tokens \
        --num_decode_tokens 0 \
        --batch_size 1 \
        --dtype float16
}

echo "Benchmark (Ring then Regular) - $(date)" | tee "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Test various context lengths
for num_tokens in 256 512 1024 2048 4096 8192 16384 32768 65536 131072 262144; do
    echo "tokens=$num_tokens START" | tee -a "$LOG_FILE"

    echo "[ring] tokens=$num_tokens gpus=$NUM_GPUS" | tee -a "$LOG_FILE"
    run_ring_benchmark "$num_tokens" "$NUM_GPUS" 2>&1 | tee -a "$LOG_FILE"

    echo "[regular] tokens=$num_tokens gpus=1" | tee -a "$LOG_FILE"
    run_regular_benchmark "$num_tokens" 2>&1 | tee -a "$LOG_FILE"

    echo "tokens=$num_tokens DONE" | tee -a "$LOG_FILE"
done


echo "Benchmark complete" | tee -a "$LOG_FILE"
echo "results=$SUMMARY_CSV" | tee -a "$LOG_FILE"
echo "log=$LOG_FILE" | tee -a "$LOG_FILE"

# Final cleanup
cleanup_processes

# Show results if CSV exists
if [ -f "$SUMMARY_CSV" ]; then
    echo "" | tee -a "$LOG_FILE"
    echo "Results Summary:" | tee -a "$LOG_FILE"
    cat "$SUMMARY_CSV" | tee -a "$LOG_FILE"
fi