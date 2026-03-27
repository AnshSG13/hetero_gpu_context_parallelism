# HPML Testing

Benchmarks, analysis, and plotting for heterogeneous ring attention research.

## Directory Structure

```
hpml_testing/
├── utils.py              Shared utilities (distributed helpers, split strategies, benchmarking)
├── benchmarks/           Benchmark scripts
├── plotting/             Plotting and analysis scripts
├── tests/                Ring attention correctness tests
├── scripts/              Shell launch scripts and model download
├── results/              Benchmark result CSVs
├── benchmark_results/    Additional benchmark outputs
└── plots/                Generated plot images
```

## Primary Workflow

1. **Generate performance profile**
   ```bash
   python3 hpml_testing/benchmarks/generate_ring_profile.py
   ```

2. **Run full benchmark sweep** (logs to Weights & Biases)
   ```bash
   python3 hpml_testing/benchmarks/run_sweep.py --profile-path hpml_testing/results/ring_attention_profile.csv
   ```

3. **Plot sweep results** (auto-called by run_sweep.py, or standalone)
   ```bash
   python3 -m hpml_testing.plotting.sweep_results
   ```

4. **Generate paper figures**
   ```bash
   python3 -m hpml_testing.plotting.paper_figures
   ```

## Standalone Scripts

- **`scripts/run_hetero_benchmark.sh`** — Quick comparison of rebalancing strategies
- **`scripts/run_matmul_mps_sweep.sh`** — Generate tflops-based performance profile from matmul benchmark
- **`scripts/benchmark_ring_vs_regular.sh`** — Ring vs regular attention benchmark (SLURM)
- **`scripts/benchmark_hetero_full_model.sh`** — Full model heterogeneous benchmark

## Prerequisites

1. **Python Libraries**:
   ```bash
   pip install pandas matplotlib seaborn wandb
   ```

2. **Weights & Biases Login** (one-time):
   ```bash
   wandb login
   ```

3. **CUDA MPS Daemon** (for simulating heterogeneous GPUs):
   ```bash
   sudo nvidia-cuda-mps-control -d
   ```
