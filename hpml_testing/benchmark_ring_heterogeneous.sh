#!/bin/bash
#SBATCH -p gpu-preempt
#SBATCH -t 0:10:00
#SBATCH --gpus-per-node=2
#SBATCH --constraint=a100
#SBATCH --mem=26G
#SBATCH -o ../logs/slurm-%j.out
#SBATCH -e ../logs/slurm-%j.err
#SBATCH --job-name=ring_hetero
#SBATCH --nodes=1

set -e

# Model path (same as run_all_benchmarks.sh)
MODEL_PATH="/datasets/ai/llama3/hub/models--meta-llama--Llama-3.2-1B/snapshots/4e20de362430cd3b72f300e6b0f18e50e7166e08"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="benchmark_results"
NUM_GPUS=2

# Configurable parameters
SEQ_LEN=${SEQ_LEN:-16384}

module load conda/latest
conda activate context_parallelism
module load cuda/12.6

mkdir -p "$RESULTS_DIR"
LOG_FILE="$RESULTS_DIR/hetero_run_${TIMESTAMP}.log"

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi
nvidia-smi topo -m

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

echo "==================================================================" | tee "$LOG_FILE"
echo "Ring Attention Heterogeneous Benchmark - $(date)" | tee -a "$LOG_FILE"
echo "Sequence Length: $SEQ_LEN" | tee -a "$LOG_FILE"
echo "==================================================================" | tee -a "$LOG_FILE"

########################
# 1) EVEN SHARDING (homogeneous baseline)
########################
echo "" | tee -a "$LOG_FILE"
echo "----------------------- Even Sharding (Baseline) -----------------------" | tee -a "$LOG_FILE"

torchrun --nproc_per_node=$NUM_GPUS "$SCRIPT_DIR/test_ring_prefill.py" \
    --total_seq_len $SEQ_LEN 2>&1 | tee -a "$LOG_FILE"

echo "Even sharding complete." | tee -a "$LOG_FILE"

########################
# 2) HETEROGENEOUS SHARDING (more tokens to rank 0)
########################
echo "" | tee -a "$LOG_FILE"
echo "----------------------- Heterogeneous Sharding -----------------------" | tee -a "$LOG_FILE"
echo "Rank 0 gets 75% tokens, Rank 1 gets 25% tokens" | tee -a "$LOG_FILE"

# Calculate block lengths for heterogeneous split (75/25)
RANK0_TOKENS=$((SEQ_LEN * 3 / 4))
RANK1_TOKENS=$((SEQ_LEN - RANK0_TOKENS))

BLOCK_LENS="${RANK0_TOKENS},${RANK1_TOKENS}" torchrun --nproc_per_node=$NUM_GPUS "$SCRIPT_DIR/test_ring_prefill.py" \
    --total_seq_len $SEQ_LEN 2>&1 | tee -a "$LOG_FILE"

echo "Heterogeneous sharding complete." | tee -a "$LOG_FILE"

########################
# 3) INVERSE HETEROGENEOUS (more tokens to rank 1)
########################
echo "" | tee -a "$LOG_FILE"
echo "----------------------- Inverse Heterogeneous -----------------------" | tee -a "$LOG_FILE"
echo "Rank 0 gets 25% tokens, Rank 1 gets 75% tokens" | tee -a "$LOG_FILE"

BLOCK_LENS="${RANK1_TOKENS},${RANK0_TOKENS}" torchrun --nproc_per_node=$NUM_GPUS "$SCRIPT_DIR/test_ring_prefill.py" \
    --total_seq_len $SEQ_LEN 2>&1 | tee -a "$LOG_FILE"

echo "Inverse heterogeneous sharding complete." | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "==================================================================" | tee -a "$LOG_FILE"
echo "All benchmarks complete. Results saved to: $LOG_FILE" | tee -a "$LOG_FILE"
echo "==================================================================" | tee -a "$LOG_FILE"
