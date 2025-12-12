#!/bin/bash
  #SBATCH -p gpu-preempt
  #SBATCH -t 0:30:00
  #SBATCH --gpus-per-node=2
  #SBATCH --constraint=a100
  #SBATCH --mem=26G
  #SBATCH -o ../logs/slurm-%j.out
  #SBATCH -e ../logs/slurm-%j.err
  #SBATCH --job-name=hetero_q_mps_test
  #SBATCH --nodes=1

  set -e
  trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT

  MODEL_PATH="/datasets/ai/llama3/hub/models--meta-llama--Llama-3.2-1B/snapshots/4e20de362430cd3b72f300e6b0f18e50e7166e08"
  TIMESTAMP=$(date +%Y%m%d_%H%M%S)
  RESULTS_DIR="benchmark_results"

  module load conda/latest
  conda activate context_parallelism
  module load cuda/12.6

  SUMMARY_CSV="$RESULTS_DIR/hetero_q_mps_${TIMESTAMP}.csv"
  LOG_FILE="$RESULTS_DIR/hetero_q_mps_${TIMESTAMP}.log"

  SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
  export PYTHONPATH="$SCRIPT_DIR/../:$PYTHONPATH"

  mkdir -p "$RESULTS_DIR"

  # MPS Setup
  export CUDA_MPS_PIPE_DIRECTORY="${CUDA_MPS_PIPE_DIRECTORY:-/tmp/mps_$USER}"
  export CUDA_MPS_LOG_DIRECTORY="${CUDA_MPS_LOG_DIRECTORY:-/tmp/mps_$USER}"
  mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"

  # Start MPS daemon if needed
  if ! pgrep -u "$USER" -f "nvidia-cuda-mps-control" >/dev/null 2>&1; then
      echo "Starting MPS daemon..."
      nvidia-cuda-mps-control -d
      sleep 2
  fi

  export NCCL_DEBUG=WARN

  # Common distributed settings
  export WORLD_SIZE=2
  export MASTER_ADDR='localhost'

  # Cleanup function
  cleanup_processes() {
      pkill -9 -f "benchmark_hetero_q.py" 2>/dev/null || true
      sleep 2
  }

  # Run heterogeneous benchmark with MPS throttling
  # Args: num_tokens, gpu0_tokens, gpu1_tokens, gpu0_mps, gpu1_mps
  run_hetero_mps_benchmark() {
      local num_tokens=$1
      local gpu0_tokens=$2
      local gpu1_tokens=$3
      local gpu0_mps=$4
      local gpu1_mps=$5

      cleanup_processes

      local master_port=$((29500 + RANDOM % 1000))
      export MASTER_PORT=$master_port

      echo ""
      echo "[HETERO Q + MPS] tokens=$num_tokens"
      echo "  GPU 0: $gpu0_tokens tokens, MPS=${gpu0_mps}%"
      echo "  GPU 1: $gpu1_tokens tokens, MPS=${gpu1_mps}%"

      # Launch Rank 0 (GPU 0)
      (
          export RANK=0
          export LOCAL_RANK=0
          export CUDA_VISIBLE_DEVICES=0
          export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$gpu0_mps
          python3 "$SCRIPT_DIR/benchmark_hetero_q.py" \
              --architecture hf_pretrained \
              --model_path "$MODEL_PATH" \
              --device_type cuda \
              --num_tokens $num_tokens \
              --gpu0_tokens $gpu0_tokens \
              --gpu1_tokens $gpu1_tokens \
              --num_decode_tokens 0 \
              --batch_size 1 \
              --summary_csv "$SUMMARY_CSV"
      ) &
      PID0=$!

      # Launch Rank 1 (GPU 1)
      (
          export RANK=1
          export LOCAL_RANK=1
          export CUDA_VISIBLE_DEVICES=1
          export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$gpu1_mps
          python3 "$SCRIPT_DIR/benchmark_hetero_q.py" \
              --architecture hf_pretrained \
              --model_path "$MODEL_PATH" \
              --device_type cuda \
              --num_tokens $num_tokens \
              --gpu0_tokens $gpu0_tokens \
              --gpu1_tokens $gpu1_tokens \
              --num_decode_tokens 0 \
              --batch_size 1
      ) &
      PID1=$!

      # Wait for both
      wait $PID0
      wait $PID1

      cleanup_processes
  }

  echo "Heterogeneous Q + MPS Throttling Test - $(date)" | tee "$LOG_FILE"
  echo "" | tee -a "$LOG_FILE"

  nvidia-smi
  nvidia-smi topo -m

  # Test parameters
  TEST_TOKENS=8192
  GPU0_MPS=100  # Full speed
  GPU1_MPS=50   # Throttled to 50%

  echo "" | tee -a "$LOG_FILE"
  echo "Testing with GPU0=${GPU0_MPS}% MPS, GPU1=${GPU1_MPS}% MPS" | tee -a "$LOG_FILE"

  # Test 1: Even split (should be unbalanced due to MPS)
  echo "" | tee -a "$LOG_FILE"
  echo "--- Test 1: Even split (50/50 tokens) with uneven MPS ---" | tee -a "$LOG_FILE"
  gpu0_tokens=$((TEST_TOKENS / 2))
  gpu1_tokens=$((TEST_TOKENS - gpu0_tokens))
  run_hetero_mps_benchmark $TEST_TOKENS $gpu0_tokens $gpu1_tokens $GPU0_MPS $GPU1_MPS 2>&1 | tee -a "$LOG_FILE"

  # Test 2: Proportional split (should balance load)
  # GPU0 is 2x faster, so give it 2x tokens: 66.7% / 33.3%
  echo "" | tee -a "$LOG_FILE"
  echo "--- Test 2: Proportional split (~67/33 tokens) matching MPS ratio ---" | tee -a "$LOG_FILE"
  gpu0_tokens=$(python3 -c "print(int($TEST_TOKENS * $GPU0_MPS / ($GPU0_MPS + $GPU1_MPS)))")
  gpu1_tokens=$((TEST_TOKENS - gpu0_tokens))
  run_hetero_mps_benchmark $TEST_TOKENS $gpu0_tokens $gpu1_tokens $GPU0_MPS $GPU1_MPS 2>&1 | tee -a "$LOG_FILE"

  # Test 3: Different MPS ratios
  echo "" | tee -a "$LOG_FILE"
  echo "--- Test 3: More extreme MPS difference (100% vs 30%) ---" | tee -a "$LOG_FILE"
  GPU1_MPS=30
  # Even split
  gpu0_tokens=$((TEST_TOKENS / 2))
  gpu1_tokens=$((TEST_TOKENS - gpu0_tokens))
  run_hetero_mps_benchmark $TEST_TOKENS $gpu0_tokens $gpu1_tokens $GPU0_MPS $GPU1_MPS 2>&1 | tee -a "$LOG_FILE"

  # Proportional split for 100/30
  gpu0_tokens=$(python3 -c "print(int($TEST_TOKENS * $GPU0_MPS / ($GPU0_MPS + $GPU1_MPS)))")
  gpu1_tokens=$((TEST_TOKENS - gpu0_tokens))
  echo "" | tee -a "$LOG_FILE"
  echo "--- Test 4: Proportional split for 100/30 MPS ---" | tee -a "$LOG_FILE"
  run_hetero_mps_benchmark $TEST_TOKENS $gpu0_tokens $gpu1_tokens $GPU0_MPS $GPU1_MPS 2>&1 | tee -a "$LOG_FILE"

  echo "" | tee -a "$LOG_FILE"
  echo "========================================" | tee -a "$LOG_FILE"
  echo "Test Complete!" | tee -a "$LOG_FILE"
  echo "Results saved to: $SUMMARY_CSV" | tee -a "$LOG_FILE"

  cleanup_processes

  if [ -f "$SUMMARY_CSV" ]; then
      echo "" | tee -a "$LOG_FILE"
      echo "Results Summary:" | tee -a "$LOG_FILE"
      cat "$SUMMARY_CSV" | tee -a "$LOG_FILE"
  fi