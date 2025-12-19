# Heterogeneity-Aware Context Parallelism in Ring Attention

This repository contains the code, experiments, and analysis for **Heterogeneity-Aware Context Parallelism in Ring Attention**, a study of how intra-node GPU heterogeneity impacts ring attention performance and how heterogeneity-aware KV partitioning can mitigate straggler effects in long-context LLM inference.

---

## Overview

### Background and Motivation
Large Language Model (LLM) inference increasingly relies on **context parallelism** techniques such as Ring Attention to scale to long sequence lengths beyond single-GPU memory limits. Existing implementations implicitly assume **homogeneous hardware**, uniformly partitioning the KV cache across ranks.

In practice, however, datacenter deployments may exhibit **intra-node heterogeneity** due to partial upgrades, throttling, or degraded devices. In synchronous distributed execution, this heterogeneity induces severe **straggler effects**, collapsing system throughput to that of the slowest rank.

### What This Project Does
This project:
- Quantifies the performance degradation caused by GPU heterogeneity in ring attention
- Implements **heterogeneity-aware context partitioning strategies**
- Evaluates multiple allocation schemes under controlled heterogeneity
- Demonstrates substantial recovery of lost performance with simple proportional KV sharding

### Key Contributions
- Empirical characterization of heterogeneity-induced slowdown in ring attention
- Design and implementation of uneven KV partitioning strategies
- Evaluation of lookup-table and regression-based allocation models
- Open-source integration with IBM’s Foundation Model Stack (FMS)

---

## Dependencies and Environment Setup

### System Requirements
- **OS:** Linux (required for distributed training)
- **Python:** 3.11 recommended for best performance
- **CUDA:** >= 12.1
- **PyTorch:** >= 2.1
- **GPUs:** 2+ NVIDIA GPUs with P2P support

### Installation

**1. Clone the repository:**
```bash
git clone https://github.com/AnshSG13/hetero_gpu_context_parallelism.git
cd hetero_gpu_context_parallelism
```

**2. Install PyTorch with CUDA (if not already installed):**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```
> **Note:** This step ensures GPU support. If you skip this, `pip install -e .` may install CPU-only PyTorch.

**3. Install the package:**
```bash
pip install -e .
```

**4. Install additional dependencies for benchmarking:**
```bash
pip install triton numpy pandas matplotlib scikit-learn wandb
```

## Experiment Tracking

Benchmark results and metrics are logged to Weights & Biases:
**[View W&B Dashboard](https://wandb.ai/wb2426-columbia-university/heterogeneous-ring-attention/runs/thopykvx?nw=nwuserwb2426)**

## Demo

To run a sweep demonstration of the ring attention implementation and various partitioning strategies, execute:

```bash
./demo.sh
```

## Ring Attention Implementation

This section describes the code changes and new components introduced to support **heterogeneity-aware ring attention** within IBM’s Foundation Model Stack (FMS). The implementation extends the existing distributed strategy framework to allow uneven context partitioning across ranks and overlaps communication with computation using separate CUDA streams.

---

### Modified Files

- **`fms/distributed/strategy.py`**  
  Added a new `RingAttentionStrategy` class that implements context parallelism using a ring topology.  
  Key changes include:
  - Separate CUDA streams for compute and communication to enable overlap
  - Support for uneven token distribution via a `block_lens` parameter  
  - Initial infrastructure for heterogeneity-aware partitioning (work in progress)

- **`fms/models/__init__.py`**  
  Registered `"ring"` as a valid distributed strategy option.  
  The module now parses the `block_lens` argument from keyword arguments and forwards it to the strategy.

- **`fms/models/llama.py`**  
  Modified `LLaMABlock` and `LLaMAHeadless` to support ring attention execution.  
  Standard attention calls are conditionally replaced with the ring-based attention path when the `"ring"` strategy is enabled.

---

### New Files

- **`fms/distributed/ring_attention.py`**  
  Contains the core ring attention implementation.  
  - `ring_forward()` is invoked from `LLaMABlock` and replaces the standard attention forward pass  
  - `_compute_attention_ring_pass_kv()` implements the main ring loop, where KV blocks are rotated across ranks  
  - Uses two CUDA streams: the default stream for attention compute and a dedicated stream for peer-to-peer communication  
  - Relies on an online softmax formulation to correctly accumulate attention across uneven KV shards

- **`fms/distributed/triton_block.py`**  
  Implements a custom Triton kernel used for block-wise attention computation.  
  - Computes per-block softmax statistics (partial sums and maxima)  
  - Enables correct online softmax merging across local and remote KV blocks  
  - Used by the ring attention path for off-diagonal (remote KV) attention tiles

- **`hpml_testing/`**  
  Contains benchmarking and testing utilities used to evaluate heterogeneous ring attention behavior.

- **`hpml_testing/run_hetero_benchmark.sh`**  
  Shell script for running heterogeneous ring attention benchmarks.  
  - Evaluates four partitioning strategies (even, uneven, LUT, formula)  
  - Configures simulated heterogeneity via MPS throttling

- **`hpml_testing/benchmark_hetero_latency.py`**  
  Python script invoked by the benchmark shell script.  
  - Runs ring attention microbenchmarks under specified MPS configurations  
  - Logs latency, slowdown, and efficiency metrics for analysis

---

### Notes

- The current implementation focuses on **prefill (prompt processing)** rather than decode
- Both query and KV tensors use heterogeneity-aware partitioning based on `block_lens`
- Support for dynamic rebalancing and multi-rank (>2) heterogeneous rings is future work

---

## Results

### Experimental Setup

All experiments were conducted on a single node with **two NVIDIA L40 GPUs**. Hardware heterogeneity was simulated using NVIDIA MPS (Multi-Process Service) to throttle one GPU's compute capacity:
- **Rank 0**: 100% MPS (full speed)
- **Rank 1**: 10%-90% MPS (simulated slower GPU)

We evaluated four partitioning strategies:
- **Even Split**: Standard uniform partitioning (baseline)
- **Uneven Split**: Proportional allocation based on GPU capabilities
- **LUT Split**: Lookup-table based allocation from profiling data
- **Formula Split**: Regression-based allocation model

Sequence lengths ranged from **4,096 to 65,536 tokens**.

---

### Strategy Comparison

![Strategy Evaluation](latex/images/strtategy_eval.png)

**Figure 1**: Slowdown factor for each partitioning strategy across sequence lengths and MPS configurations. Lower values indicate better performance. The dashed red line at 1.0 represents ideal (homogeneous) performance.

**Key Observations**:
- Even split performance degrades rapidly as heterogeneity increases, exhibiting **5-8x slowdowns** at 10% MPS
- Uneven split consistently achieves the best performance, reducing slowdown from 5-8x to approximately **2x** at extreme heterogeneity
- LUT and formula strategies provide middle-ground performance but rarely outperform simple proportional allocation

---

### Speedup Analysis

![Speedup Heatmap](latex/images/speedup_heatmap.png)

**Figure 2**: Speedup of uneven split over even split. Darker green indicates larger improvements.

The speedup increases with both sequence length and heterogeneity, reaching **4.4x at 65K tokens with 10% MPS**.

---

### Extreme Heterogeneity Performance

![Extreme Heterogeneity](latex/images/extreme_heterogeneity.png)

**Figure 3**: Absolute latency at 10% MPS (extreme heterogeneity).

At extreme heterogeneity:
- **Even split**: Over 6 seconds for 65K tokens
- **Uneven split**: Under 1.4 seconds for 65K tokens
- Adaptive strategies reduce latency by up to **4.4x**

---

### Efficiency Comparison

![Efficiency vs Heterogeneity](latex/images/efficiency_vs_heterogeneity.png)

**Figure 4**: Efficiency relative to homogeneous baseline across MPS configurations.

**Key Observations**:
- Even split efficiency drops below **20%** at severe heterogeneity (over 80% of potential performance lost to waiting)
- Adaptive strategies maintain **40-50% efficiency** even at 10% MPS
- This represents a **2-3x improvement** in resource utilization

---

### Summary of Findings

| Finding | Details |
|---------|---------|
| Heterogeneity Impact | Even partitioning causes **>5x slowdown** at 10:1 capability ratios |
| Best Strategy | Simple proportional (uneven) split achieves up to **4.4x speedup** over even split |
| Complex Strategies | LUT and formula provide marginal benefit with added complexity |
| Efficiency Recovery | Adaptive strategies recover **2-3x** more resource utilization |

---

## Repository Structure

```
.
├── fms/                          # Modified IBM FMS source
│   ├── distributed/
│   │   ├── ring_attention.py     # Core ring attention implementation
│   │   ├── strategy.py           # RingAttentionStrategy class
│   │   └── triton_block.py       # Custom Triton kernel
│   └── models/
│       ├── __init__.py           # Strategy registration
│       └── llama.py              # LLaMA model modifications
├── hpml_testing/                 # Benchmarking scripts
│   ├── benchmark_hetero_latency.py
│   ├── run_hetero_benchmark.sh
│   └── plots/                    # Generated figures
├── latex/                        # Paper source and figures
│   ├── report.tex
│   └── images/
└── demo.sh                       # Quick demo script
```
