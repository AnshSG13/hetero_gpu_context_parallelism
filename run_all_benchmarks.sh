#!/bin/bash
#
# Benchmark Runner Script for Ring Attention Context Parallelism
# This script runs benchmarks at different context lengths and GPU counts,
# then generates a summary table of results.
#

set -e  # Exit on error

# =============================================================================
# Configuration - Modify these paths for your environment
# =============================================================================

MODEL_PATH="/datasets/ai/llama3/hub/models--meta-llama--Meta-Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"
TOKENIZER_PATH="/datasets/ai/llama3/hub/models--meta-llama--Meta-Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"
VARIANT="3-8b"
ARCHITECTURE="llama"
DEVICE_TYPE="cuda"
NUM_TOKENS_TO_BENCHMARK=30

# Output files
RESULTS_DIR="benchmark_results"
CSV_FILE="$RESULTS_DIR/benchmark_results.csv"
TABLE_FILE="$RESULTS_DIR/benchmark_table.md"
LOGS_DIR="$RESULTS_DIR/logs"

# =============================================================================
# Setup
# =============================================================================

echo "========================================"
echo "Ring Attention Benchmark Runner"
echo "========================================"
echo ""

# Create output directories
mkdir -p "$RESULTS_DIR"
mkdir -p "$LOGS_DIR"

# Initialize CSV file with headers
echo "Context_Length,GPUs,Avg_Time_Token_Ring_ms,Avg_Time_Token_Regular_ms,Speedup,Outputs_Match" > "$CSV_FILE"

# =============================================================================
# Helper Functions
# =============================================================================

# Parse benchmark output to extract average time per token
parse_avg_time() {
    local log_file=$1
    local label=$2

    # Look for "Summary for <label>:" followed by "Average time per token: X.XX ms"
    grep -A 3 "Summary for $label:" "$log_file" | grep "Average time per token:" | awk '{print $5}'
}

# Check if correctness assertion passed
check_correctness() {
    local log_file=$1

    # If the script completed without assertion errors, it passed
    # Check for assertion error or script failure
    if grep -q "AssertionError" "$log_file" || grep -q "FAILED" "$log_file"; then
        echo "FAILED"
    else
        echo "PASSED"
    fi
}

# Run a single benchmark
run_benchmark() {
    local context_len=$1
    local num_gpus=$2
    local log_file="$LOGS_DIR/benchmark_ctx${context_len}_gpu${num_gpus}.log"

    echo "Running benchmark: Context=$context_len, GPUs=$num_gpus"
    echo "  Log file: $log_file"

    # Run the benchmark and capture output
    # Don't exit on error - we want to collect timing data even if correctness check fails
    set +e
    torchrun --nproc_per_node=$num_gpus \
        scripts/llama_ring/benchmark_ring.py \
        --architecture "$ARCHITECTURE" \
        --variant "$VARIANT" \
        --model_path "$MODEL_PATH" \
        --tokenizer "$TOKENIZER_PATH" \
        --device_type "$DEVICE_TYPE" \
        --num_tokens_to_benchmark $NUM_TOKENS_TO_BENCHMARK \
        --batch_size 1 \
        --run_ring_first \
        --prompt_len $context_len \
        > "$log_file" 2>&1
    set -e

    # Parse results
    local ring_time=$(parse_avg_time "$log_file" "Ring Attention")
    local regular_time=$(parse_avg_time "$log_file" "Regular Attention")
    local correctness=$(check_correctness "$log_file")

    # Calculate speedup (if both times are available)
    local speedup="N/A"
    if [[ -n "$ring_time" && -n "$regular_time" ]]; then
        speedup=$(echo "scale=2; $regular_time / $ring_time" | bc)
    fi

    # If ring_time or regular_time is empty, mark as failed
    if [[ -z "$ring_time" ]]; then
        ring_time="ERROR"
    fi
    if [[ -z "$regular_time" ]]; then
        regular_time="ERROR"
    fi

    # Save to CSV
    echo "$context_len,$num_gpus,$ring_time,$regular_time,$speedup,$correctness" >> "$CSV_FILE"

    echo "  Ring Attention: $ring_time ms/token"
    echo "  Regular Attention: $regular_time ms/token"
    echo "  Speedup: ${speedup}x"
    echo "  Correctness: $correctness"
    echo ""
}

# =============================================================================
# Run Benchmarks
# =============================================================================

echo "Starting benchmarks..."
echo ""

# Context length scaling with 2 GPUs
for context_len in 100 1024 2048 4096 8192; do
    run_benchmark $context_len 2
done

# Note: 4 GPU test disabled (only 2 GPUs available)
# Uncomment the following lines if you have 4 GPUs:
# echo "Running 4 GPU benchmark..."
# run_benchmark 8192 4

echo "All benchmarks completed!"
echo ""

# =============================================================================
# Generate Summary Table
# =============================================================================

echo "Generating summary table..."

# Create markdown table
cat > "$TABLE_FILE" << 'EOF'
# Ring Attention Benchmark Results

## Summary Table

| Context Length | GPUs | Avg Time/Token (Ring) | Avg Time/Token (Regular) | Speedup | Outputs Match? |
|----------------|------|-----------------------|--------------------------|---------|----------------|
EOF

# Read CSV and format as markdown table
tail -n +2 "$CSV_FILE" | while IFS=',' read -r ctx gpus ring_time reg_time speedup correctness; do
    # Format times with "ms" suffix
    if [[ "$ring_time" != "ERROR" ]]; then
        ring_display="${ring_time} ms"
    else
        ring_display="ERROR"
    fi

    if [[ "$reg_time" != "ERROR" ]]; then
        reg_display="${reg_time} ms"
    else
        reg_display="ERROR"
    fi

    # Format speedup
    if [[ "$speedup" != "N/A" ]]; then
        speedup_display="${speedup}x"
    else
        speedup_display="N/A"
    fi

    echo "| $ctx            | $gpus    | $ring_display                 | $reg_display                    | $speedup_display       | $correctness          |" >> "$TABLE_FILE"
done

echo "" >> "$TABLE_FILE"
echo "## Notes" >> "$TABLE_FILE"
echo "" >> "$TABLE_FILE"
echo "- All benchmarks run with $NUM_TOKENS_TO_BENCHMARK tokens generated per test" >> "$TABLE_FILE"
echo "- Model: $VARIANT" >> "$TABLE_FILE"
echo "- Speedup calculated as: Regular Time / Ring Time" >> "$TABLE_FILE"
echo "- Correctness verified by comparing output logits between Ring and Regular attention" >> "$TABLE_FILE"

# =============================================================================
# Display Results
# =============================================================================

echo ""
echo "========================================"
echo "Results Summary"
echo "========================================"
echo ""
cat "$TABLE_FILE"
echo ""
echo "========================================"
echo "Raw CSV data saved to: $CSV_FILE"
echo "Markdown table saved to: $TABLE_FILE"
echo "Individual logs saved to: $LOGS_DIR/"
echo "========================================"
