#!/usr/bin/env bash
#
# Sweep single-GPU matmul performance across matrix sizes and MPS throttling.
# Requires CUDA and the benchmark at hpml_testing/benchmark_single_gpu_matmul.py.

set -euo pipefail

# Configurable via environment:
#   SIZES: space-separated square sizes (e.g., "2048 4096 8192")
#   MPS_PCTS: space-separated MPS percentages (e.g., "100 80 60 40 20")
#   DTYPE: matmul dtype (float16 | bfloat16 | float32)
#   NUM_ITERS: timed iterations per run
#   WARMUP_ITERS: warmup iterations per run
#   BG_STREAMS: background streams to add contention
#   OUT_CSV: output CSV path

SIZES=${SIZES:-"64 128 256 512 1024 2048 4096 8192 16384 32768"}
MPS_PCTS=${MPS_PCTS:-"100 80 60 40 20 15 10 5 2.5 1"}
DTYPE=${DTYPE:-float16}
NUM_ITERS=${NUM_ITERS:-20}
WARMUP_ITERS=${WARMUP_ITERS:-5}
BG_STREAMS=${BG_STREAMS:-0}
OUT_CSV=${OUT_CSV:-"results/matmul_mps_sweep_$(date +%Y%m%d_%H%M%S).csv"}

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
RESULTS_DIR=$(dirname "$OUT_CSV")
mkdir -p "$RESULTS_DIR"

echo "mps_pct,size,dtype,bg_streams,avg_ms,tflops" > "$OUT_CSV"

for pct in $MPS_PCTS; do
  export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$pct
  for n in $SIZES; do
    echo "=== MPS=${pct}% size=${n} ==="
    # Capture the data line for this size (filter out headers/blank lines).
    line=$(python3 "$SCRIPT_DIR/benchmark_single_gpu_matmul.py" \
      --sizes "$n" \
      --dtype "$DTYPE" \
      --num-iters "$NUM_ITERS" \
      --warmup-iters "$WARMUP_ITERS" \
      --background-streams "$BG_STREAMS" \
      | grep -E '^[[:space:]]*[0-9]+' | tail -n 1)

    if [[ -z "$line" ]]; then
      echo "No benchmark output captured for MPS=${pct} size=${n}" >&2
      continue
    fi

    # Expected format: "     size |    dtype | bg_streams |     avg_ms |     TFLOPs"
    # Use awk to extract fields by position.
    size_field=$(echo "$line" | awk '{print $1}')
    dtype_field=$(echo "$line" | awk '{print $3}')
    bg_field=$(echo "$line" | awk '{print $5}')
    avg_ms_field=$(echo "$line" | awk '{print $7}')
    tflops_field=$(echo "$line" | awk '{print $9}')

    echo "${pct},${size_field},${dtype_field},${bg_field},${avg_ms_field},${tflops_field}" >> "$OUT_CSV"
  done
done

echo "Saved sweep to $OUT_CSV"
