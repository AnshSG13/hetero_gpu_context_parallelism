"""Shared utilities for hpml_testing benchmarks.

Consolidates duplicated code from benchmark scripts:
- Distributed helpers (print0, init)
- Block length / split strategies
- Performance profiling
- Model setup and benchmarking helpers
"""

import csv
import math
import os
import statistics
import time

import torch
import torch.distributed as dist


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------

def print0(*args, **kwargs):
    """Print only on rank 0."""
    if int(os.getenv("RANK", 0)) == 0:
        print(*args, **kwargs)


def init_distributed(device_type="cuda"):
    """Read env vars, init process group, return (rank, local_rank, world_size, device).

    Skips initialization when WORLD_SIZE <= 1 or device_type != cuda.
    """
    rank = int(os.getenv("RANK", 0))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))

    if world_size > 1 and device_type == "cuda":
        torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device(device_type)

    return rank, local_rank, world_size, device


# ---------------------------------------------------------------------------
# Block length / split strategies
# ---------------------------------------------------------------------------

def compute_block_lens(seq_len, world_size, split_type, slowdown_factor=0.5,
                       perf_profile_path=None, rank_mps_list=None):
    """Compute block lengths for heterogeneous ring attention.

    Parameters
    ----------
    seq_len : int
        Total sequence length.
    world_size : int
        Number of GPUs.
    split_type : str
        One of "even", "uneven", "lut", "formula".
    slowdown_factor : float
        Proportional slowdown of the weak GPU (for "uneven" split).
    perf_profile_path : str, optional
        Path to performance profile CSV (for "lut" split).
    rank_mps_list : list[float], optional
        MPS percentages per rank (for "lut"/"formula" splits).

    Returns
    -------
    list[int]
        Block lengths summing to seq_len.
    """
    if split_type == "even":
        base_len = seq_len // world_size
        block_lens = [base_len] * world_size
        remainder = seq_len % world_size
        for i in range(remainder):
            block_lens[i] += 1

    elif split_type == "uneven":
        weights = [1.0] + [slowdown_factor] * (world_size - 1)
        total_weight = sum(weights)
        block_lens = [int(round(seq_len * (w / total_weight))) for w in weights]
        diff = sum(block_lens) - seq_len
        block_lens[-1] -= diff

    elif split_type == "lut":
        if not perf_profile_path:
            raise ValueError("--use-perf-profile required for 'lut' split")
        if rank_mps_list is None or len(rank_mps_list) != world_size:
            raise ValueError(f"rank_mps_list must have {world_size} values")

        import pandas as pd
        df = pd.read_csv(perf_profile_path)

        if "latency_ms" in df.columns:
            perf_profile = df.set_index("mps_pct")["latency_ms"].to_dict()
            raw_perf = [get_performance_for_mps(perf_profile, mps) for mps in rank_mps_list]
            weights = [1.0 / p for p in raw_perf]
        elif "tflops" in df.columns:
            available_sizes = sorted(df["size"].unique())
            closest_size = min(available_sizes, key=lambda x: abs(x - seq_len))
            df_filtered = df[df["size"] == closest_size]
            if df_filtered.empty:
                raise ValueError(f"No data for size {closest_size}")
            perf_profile = df_filtered.set_index("mps_pct")["tflops"].to_dict()
            raw_perf = [get_performance_for_mps(perf_profile, mps) for mps in rank_mps_list]
            weights = raw_perf
        else:
            raise ValueError("Profile must contain 'latency_ms' or 'tflops' column")

        total_weight = sum(weights)
        block_lens = [int(round(seq_len * (w / total_weight))) for w in weights]
        diff = sum(block_lens) - seq_len
        block_lens[-1] -= diff

    elif split_type == "formula":
        if rank_mps_list is None or len(rank_mps_list) != world_size:
            raise ValueError(f"rank_mps_list must have {world_size} values")
        weights = [empirical_normalized_perf(seq_len, mps) for mps in rank_mps_list]
        total_weight = sum(weights)
        block_lens = [int(round(seq_len * (w / total_weight))) for w in weights]
        diff = sum(block_lens) - seq_len
        block_lens[-1] -= diff

    else:
        raise ValueError(f"Unknown split_type: {split_type}")

    return block_lens


# ---------------------------------------------------------------------------
# Performance profiling
# ---------------------------------------------------------------------------

def load_performance_profile(profile_path, size=None):
    """Load a performance profile from CSV.

    Returns (profile_dict, metric_type) where metric_type is 'latency' or 'tflops'.
    """
    import pandas as pd

    if not os.path.exists(profile_path):
        raise FileNotFoundError(f"Performance profile not found: {profile_path}")

    df = pd.read_csv(profile_path)

    if "latency_ms" in df.columns:
        return df.set_index("mps_pct")["latency_ms"].to_dict(), "latency"
    elif "tflops" in df.columns:
        available_sizes = sorted(df["size"].unique())
        closest_size = min(available_sizes, key=lambda x: abs(x - size))
        df_filtered = df[df["size"] == closest_size]
        if df_filtered.empty:
            raise ValueError(f"No data for size {closest_size}")
        return df_filtered.set_index("mps_pct")["tflops"].to_dict(), "tflops"
    else:
        raise ValueError("Profile must contain 'latency_ms' or 'tflops' column")


def get_performance_for_mps(profile, mps_pct):
    """Get performance for a given MPS percentage, with linear interpolation."""
    sorted_mps = sorted(profile.keys())

    if mps_pct in profile:
        return profile[mps_pct]

    if mps_pct > sorted_mps[-1]:
        return profile[sorted_mps[-1]]
    if mps_pct < sorted_mps[0]:
        return profile[sorted_mps[0]]

    for i, p in enumerate(sorted_mps):
        if p > mps_pct:
            high_p = p
            low_p = sorted_mps[i - 1]
            break

    high_val = profile[high_p]
    low_val = profile[low_p]
    weight = (mps_pct - low_p) / (high_p - low_p)
    return low_val + weight * (high_val - low_val)


def empirical_normalized_perf(seq_len, mps_pct):
    """Compute normalized performance (%) as a function of sequence length and MPS percentage.

    Polynomial fit from matmul MPS sweep data.
    """
    log_seq = math.log10(seq_len)
    return (
        -662.478
        + 365.074 * log_seq
        + 0.195907 * mps_pct
        - 49.6378 * (log_seq ** 2)
        + 0.337949 * log_seq * mps_pct
        - 0.00710625 * (mps_pct ** 2)
    )


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------

def setup_model(architecture, model_path, device_type, strategy, block_lens,
                dtype, variant=None, source=None):
    """Load model via fms.models.get_model(), set eval mode, disable grad."""
    from fms import models

    kwargs = dict(
        model_path=model_path,
        device_type=device_type,
        distributed_strategy=strategy,
        block_lens=block_lens,
        data_type=dtype,
    )
    if architecture == "hf_pretrained":
        model = models.get_model(architecture, **kwargs)
    else:
        model = models.get_model(
            architecture,
            variant or "8b",
            source=source or "hf",
            **kwargs,
        )
    model.eval()
    torch.set_grad_enabled(False)
    return model


def create_random_input(batch_size, num_tokens, device, vocab_size=128256):
    """Create random token ids in a safe vocab range."""
    return torch.randint(
        100, vocab_size - 100,
        (batch_size, num_tokens),
        dtype=torch.long,
        device=device,
    )


def run_generation_benchmark(model, input_ids, num_decode, device, label=""):
    """Run prefill + decode benchmark. Returns dict with timing metrics.

    Returns
    -------
    dict with keys: ttft_ms, avg_decode_ms, total_time_ms, logits
    """
    ids = input_ids.clone().to(device)

    if device.type == "cuda":
        torch.cuda.synchronize()

    # Prefill (TTFT)
    t0 = time.perf_counter()
    out = model.forward(ids, use_cache=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    ttft_ms = (time.perf_counter() - t0) * 1000

    logits, cache = (
        (out[0], out[1]) if isinstance(out, tuple)
        else (out.logits, out.past_key_value_states)
    )
    last_token = ids[:, -1:]

    # Decode
    decode_times = []
    for _ in range(num_decode):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model.forward(last_token, past_key_value_states=cache, use_cache=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        decode_times.append((time.perf_counter() - t0) * 1000)

        logits, cache = (
            (out[0], out[1]) if isinstance(out, tuple)
            else (out.logits, out.past_key_value_states)
        )
        last_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

    avg_decode_ms = statistics.mean(decode_times) if decode_times else 0.0
    total_time_ms = ttft_ms + sum(decode_times)

    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank == 0 and label:
        print0(f"\n{label}:")
        print0(f"  TTFT: {ttft_ms:.2f} ms | Avg Decode: {avg_decode_ms:.2f} ms | Total: {total_time_ms:.2f} ms")

    return {
        "ttft_ms": ttft_ms,
        "avg_decode_ms": avg_decode_ms,
        "total_time_ms": total_time_ms,
        "logits": logits,
    }


def append_csv_row(csv_path, headers, row):
    """Append a row to a CSV file, writing headers if file doesn't exist."""
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(headers)
        writer.writerow(row)
