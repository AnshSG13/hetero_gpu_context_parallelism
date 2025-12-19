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
- OS: Linux (recommended)
- GPUs: NVIDIA GPUs with P2P support (tested on L40)
- CUDA: `>= 12.x`
- Python: `>= 3.11`

### Core Dependencies
```txt
torch
triton
numpy
pandas
matplotlib
scikit-learn
wandb
ibm-fms
```

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
- Query tensors remain evenly distributed; heterogeneity-aware partitioning is applied to KV shards
- Support for dynamic rebalancing and multi-rank (>2) heterogeneous rings is future work
