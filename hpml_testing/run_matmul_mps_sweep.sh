#!/usr/bin/env bash
SIZES=${SIZES:-"64 128 256 512 1024 2048 4096 8192 16384 32768"}
MPS_PCTS=${MPS_PCTS:-"1 10 20 30 40 50 60 70 80 90 100"}
DTYPE=${DTYPE:-float16}
NUM_ITERS=${NUM_ITERS:-5}
WARMUP_ITERS=${WARMUP_ITERS:-5}
BG_STREAMS=${BG_STREAMS:-0}
OUT_CSV=${OUT_CSV:-"./hpml_testing/results/matmul_mps_sweep.csv"}

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
RESULTS_DIR=$(dirname "$OUT_CSV")
mkdir -p "$RESULTS_DIR"

echo "mps_pct,size,dtype,bg_streams,avg_ms,tflops" > "$OUT_CSV"

for pct in $MPS_PCTS; do
  export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$pct
  for n in $SIZES; do
    echo "=== MPS=${pct}% size=${n} ==="
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

    size_field=$(echo "$line" | awk '{print $1}')
    dtype_field=$(echo "$line" | awk '{print $3}')
    bg_field=$(echo "$line" | awk '{print $5}')
    avg_ms_field=$(echo "$line" | awk '{print $7}')
    tflops_field=$(echo "$line" | awk '{print $9}')

    echo "${pct},${size_field},${dtype_field},${bg_field},${avg_ms_field},${tflops_field}" >> "$OUT_CSV"
  done
done

echo "Saved sweep to $OUT_CSV"
