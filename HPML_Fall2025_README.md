# HPML Fall 2025 

## New Contributions

### 1. GPU Interconnect Speed Measurement (`test_gpu_communication.py`)

Measures GPU-to-GPU communication bandwidth using ring shift operations to characterize interconnect performance.

**Usage:**
```bash
torchrun --nproc_per_node=2 test_gpu_communication.py
```

**Output:**
- Per-GPU timing statistics
- Measured bandwidth vs theoretical limits
- Interconnect type identification

---

### 2. Comprehensive Benchmark Suite (`run_all_benchmarks.sh`)

Automated benchmarking script that compares Ring Attention vs Regular Attention across varying context lengths.

**Usage:**
```bash
bash run_all_benchmarks.sh
```

**Output Files:**
- `benchmark_results/benchmark_results.csv` - Raw data
- `benchmark_results/benchmark_table.md` - Formatted results table
- `benchmark_results/logs/` - Individual run logs

**Metrics Captured:**
- Number Count (input prompt parameter)
- Input Tokens (actual tokenized length)
- GPUs used
- Average time/token for Ring Attention (ms)
- Average time/token for Regular Attention (ms)
- Speedup (Regular/Ring ratio)
- Output correctness validation

---

## Model Configuration

Both benchmarks use:
- **Model:** Meta-Llama-3.1-8B
- **Architecture:** LLaMA with 32 attention heads
- **Precision:** FP16
- **Backend:** NCCL (CUDA)

## Things to be Improved

### Current Benchmark Limitations

1. **KV Cache Not Enabled**

2. **Measuring Prefill + Decode Together (Not Just Decode)**


