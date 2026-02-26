"""Benchmark heterogeneous ring attention on FULL Llama models.

This combines:
- Full model loading from benchmark_ring.py
- Heterogeneous split strategies from benchmark_hetero_latency.py
"""

import argparse
import os
import statistics
import time
import csv
import gc
import torch
import torch.distributed as dist
from pathlib import Path
import pandas as pd

from fms import models
from fms.distributed.strategy import NoOpStrategy
from fms.distributed.ring_attention import reset_layer_counter, print_timing_summary

# Import the empirical performance function for formula-based splits
import sys
sys.path.insert(0, str(Path(__file__).parent))
try:
    from empirical_normalized_perf import empirical_normalized_perf
except ImportError:
    empirical_normalized_perf = None


def print0(*args, **kwargs):
    if int(os.getenv("RANK", 0)) == 0:
        print(*args, **kwargs)


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark Heterogeneous Ring Attention on Full Model")
    script_path = Path(__file__).resolve()
    repo_dir = script_path.parents[1]

    # Model args
    parser.add_argument("--architecture", type=str, default="hf_pretrained")
    parser.add_argument("--model_path", type=str, required=True, help="Path to HF model")
    parser.add_argument("--device_type", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"])

    # Benchmark args
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_tokens", type=int, required=True, help="Number of prompt tokens")
    parser.add_argument("--num_decode_tokens", type=int, default=30)
    parser.add_argument("--summary_csv", type=str, default=None)

    # Heterogeneous split args (from benchmark_hetero_latency.py)
    parser.add_argument("--split-type", type=str, choices=["even", "uneven", "lut", "formula"],
                        default="even", help="Workload split type")
    parser.add_argument("--slowdown-factor", type=float, default=0.5,
                        help="Proportional slowdown of the weak GPU (for 'uneven' split)")
    parser.add_argument("--use-perf-profile", type=str, default=None,
                        help="Path to performance profile CSV (for 'lut' split)")
    parser.add_argument("--rank-mps", type=str, default="100,50",
                        help="Comma-separated MPS percentages per rank (for 'lut'/'formula' splits)")

    return parser.parse_args()


def compute_block_lens(args, world_size):
    """Compute block lengths based on split strategy (from benchmark_hetero_latency.py)."""
    seq_len = args.num_tokens

    if args.split_type == "even":
        base_len = seq_len // world_size
        block_lens = [base_len] * world_size
        remainder = seq_len % world_size
        for i in range(remainder):
            block_lens[i] += 1

    elif args.split_type == "uneven":
        # Rank 0 = fast GPU, Rank 1 = slow GPU
        weights = [1.0] + [args.slowdown_factor] * (world_size - 1)
        total_weight = sum(weights)
        block_lens = [int(round(seq_len * (w / total_weight))) for w in weights]
        diff = sum(block_lens) - seq_len
        block_lens[-1] -= diff

    elif args.split_type == "lut":
        if not args.use_perf_profile:
            raise ValueError("--use-perf-profile required for 'lut' split")

        rank_mps_list = [float(p) for p in args.rank_mps.split(',')]
        if len(rank_mps_list) != world_size:
            raise ValueError(f"rank-mps has {len(rank_mps_list)} values but world_size is {world_size}")

        df = pd.read_csv(args.use_perf_profile)
        perf_profile = df.set_index('mps_pct')['latency_ms'].to_dict()

        # Lower latency = higher weight (inverse)
        raw_perf = [perf_profile.get(mps, perf_profile[100]) for mps in rank_mps_list]
        weights = [1.0 / p for p in raw_perf]
        total_weight = sum(weights)
        block_lens = [int(round(seq_len * (w / total_weight))) for w in weights]
        diff = sum(block_lens) - seq_len
        block_lens[-1] -= diff

    elif args.split_type == "formula":
        if empirical_normalized_perf is None:
            raise ValueError("empirical_normalized_perf.py not found")

        rank_mps_list = [float(p) for p in args.rank_mps.split(',')]
        if len(rank_mps_list) != world_size:
            raise ValueError(f"rank-mps has {len(rank_mps_list)} values but world_size is {world_size}")

        weights = [empirical_normalized_perf(seq_len, mps) for mps in rank_mps_list]
        total_weight = sum(weights)
        block_lens = [int(round(seq_len * (w / total_weight))) for w in weights]
        diff = sum(block_lens) - seq_len
        block_lens[-1] -= diff

    return block_lens


def setup_model(args, block_lens, dtype):
    """Load full model with heterogeneous ring attention strategy."""
    model = models.get_model(
        args.architecture,
        model_path=args.model_path,
        device_type=args.device_type,
        distributed_strategy="ring",
        block_lens=block_lens,
        data_type=dtype
    )
    model.eval()
    torch.set_grad_enabled(False)
    return model


def run_benchmark(model, input_ids, num_decode, device, block_lens):
    """Run generation benchmark with heterogeneous ring attention."""
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    ids = input_ids.clone().to(device)

    reset_layer_counter()

    # Warmup
    print0("Warmup pass...")
    with torch.no_grad():
        _ = model.forward(ids, use_cache=False)
    print("forward call works")
    torch.cuda.synchronize()
    print('synchronize works')
    reset_layer_counter()
    print0("Warmup done")

    # Prefill (TTFT)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = model.forward(ids, use_cache=True)
    torch.cuda.synchronize()
    ttft_ms = (time.perf_counter() - t0) * 1000

    logits, cache = (out[0], out[1]) if isinstance(out, tuple) else (out.logits, out.past_key_value_states)
    last_token = ids[:, -1:]

    # Decode
    decode_times = []
    for _ in range(num_decode):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model.forward(last_token, past_key_value_states=cache, use_cache=True)
        torch.cuda.synchronize()
        decode_times.append((time.perf_counter() - t0) * 1000)

        logits, cache = (out[0], out[1]) if isinstance(out, tuple) else (out.logits, out.past_key_value_states)
        last_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

    avg_decode_ms = statistics.mean(decode_times) if decode_times else 0.0
    total_time_ms = ttft_ms + sum(decode_times)

    # Print per-rank timing
    print_timing_summary(rank)

    # Gather per-rank metrics
    local_tokens = block_lens[rank]
    result_obj = {"rank": rank, "tokens": local_tokens, "ttft_ms": ttft_ms}
    gathered = [None] * world_size
    dist.gather_object(result_obj, gathered if rank == 0 else None, dst=0)

    if rank == 0:
        print0(f"\n--- Per-Rank Results ---")
        for r in gathered:
            print0(f"  Rank {r['rank']}: {r['tokens']} tokens, TTFT={r['ttft_ms']:.2f} ms")
        print0(f"\n  Overall TTFT (max): {max(r['ttft_ms'] for r in gathered):.2f} ms")
        print0(f"  Avg Decode: {avg_decode_ms:.2f} ms | Total: {total_time_ms:.2f} ms")

    return {"ttft_ms": ttft_ms, "avg_decode_ms": avg_decode_ms, "total_time_ms": total_time_ms}


def main():
    args = parse_args()
    rank = int(os.getenv("RANK", 0))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))

    if world_size < 2:
        raise ValueError("Ring attention requires at least 2 GPUs")

    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    device = torch.device("cuda", local_rank)

    dtype = getattr(torch, args.dtype)
    torch.set_default_dtype(dtype)

    # Compute heterogeneous block lengths
    block_lens = compute_block_lens(args, world_size)
    print0(f"\nSplit type: {args.split_type}")
    print0(f"Block lengths: {block_lens} (total: {sum(block_lens)})")

    # Create input
    vocab_size = 128256
    ids = torch.randint(100, vocab_size - 100, (args.batch_size, args.num_tokens),
                        dtype=torch.long, device=device)
    dist.broadcast(ids, src=0)

    print0(f"Benchmark: {args.num_tokens} prompt tokens, {args.num_decode_tokens} decode tokens")

    # Load model and run
    model = setup_model(args, block_lens, dtype)
    result = run_benchmark(model, ids, args.num_decode_tokens, device, block_lens)

    # Write CSV
    if rank == 0 and args.summary_csv:
        file_exists = os.path.exists(args.summary_csv)
        with open(args.summary_csv, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["split_type", "block_lens", "prompt_tokens", "ttft_ms", "avg_decode_ms", "total_ms"])
            writer.writerow([args.split_type, str(block_lens), args.num_tokens,
                            f"{result['ttft_ms']:.2f}", f"{result['avg_decode_ms']:.2f}",
                            f"{result['total_time_ms']:.2f}"])

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
