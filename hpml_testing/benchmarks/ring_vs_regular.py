"""Benchmark script for comparing Ring Attention vs Regular Attention."""

import argparse
import gc
import os
from pathlib import Path

import torch
import torch.distributed as dist

from fms.distributed.strategy import NoOpStrategy

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import (
    print0,
    init_distributed,
    setup_model,
    create_random_input,
    run_generation_benchmark,
    append_csv_row,
)

SUMMARY_HEADERS = ["strategy", "prompt_tokens", "ttft_ms", "avg_decode_ms", "total_time_ms"]


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark Ring vs Regular Attention")
    script_path = Path(__file__).resolve()
    repo_dir = script_path.parents[3]
    model_dir = repo_dir.parent / "llama-hf"

    parser.add_argument("--device_type", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--architecture", type=str, default="llama")
    parser.add_argument("--variant", type=str, default="8b")
    parser.add_argument("--model_path", type=str, default=str(model_dir))
    parser.add_argument("--tokenizer", type=str, default=str(model_dir / "tokenizer.model"))
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_tokens", type=int, required=True, help="Number of prompt tokens")
    parser.add_argument("--num_decode_tokens", type=int, default=30, help="Number of tokens to decode")
    parser.add_argument("--run_ring_first", action="store_true", default=True)
    parser.add_argument("--no-run_ring_first", dest="run_ring_first", action="store_false")
    parser.add_argument("--summary_csv", type=str, default=None, help="Summary CSV path (appends)")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--disable_flash", action="store_true", default=False,
                        help="Disable FlashAttention for fair comparison with ring attention")

    return parser.parse_args()


def main():
    args = parse_args()
    rank, local_rank, world_size, device = init_distributed(args.device_type)

    # Disable FlashAttention if requested
    if args.disable_flash:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        print0("FlashAttention DISABLED - using naive math attention for fair comparison")
    else:
        print0(f"SDPA backends: flash={torch.backends.cuda.flash_sdp_enabled()}, "
               f"mem_efficient={torch.backends.cuda.mem_efficient_sdp_enabled()}, "
               f"math={torch.backends.cuda.math_sdp_enabled()}")

    dtype = getattr(torch, args.dtype)
    torch.set_default_dtype(dtype)

    ids = create_random_input(args.batch_size, args.num_tokens, device)

    # Synchronize random tokens across ranks
    if world_size > 1:
        dist.broadcast(ids, src=0)

    print0(f"Benchmark: {args.num_tokens} prompt tokens, {args.num_decode_tokens} decode tokens")

    # Define strategies
    strategies = [("Ring", "ring"), ("Regular", NoOpStrategy)]
    if not args.run_ring_first:
        strategies.reverse()

    results = []
    for label, strategy in strategies:
        # Skip Ring if not distributed
        if strategy == "ring" and not dist.is_initialized():
            print0(f"Skipping {label} (requires distributed)")
            continue

        # Regular only runs on rank 0
        is_regular = strategy is NoOpStrategy
        should_run = not (is_regular and rank != 0)

        model = None
        if should_run:
            if args.device_type == "cuda":
                torch.cuda.empty_cache()

            # Compute block_lens for ring attention
            block_lens = None
            if strategy == "ring" and dist.is_initialized():
                local_len = args.num_tokens // world_size
                block_lens = [local_len] * world_size

            model = setup_model(
                args.architecture, args.model_path, args.device_type,
                strategy, block_lens, dtype,
                variant=args.variant, source="hf",
            )

        if should_run:
            result = run_generation_benchmark(
                model, ids, args.num_decode_tokens, device, label=label,
            )
            result["strategy"] = label
            results.append(result)

            del model
            gc.collect()
            if args.device_type == "cuda":
                torch.cuda.empty_cache()

        # ALL ranks sync here
        if world_size > 1:
            dist.barrier()

    # Write summary CSV
    if rank == 0 and args.summary_csv and results:
        for r in results:
            append_csv_row(
                args.summary_csv,
                SUMMARY_HEADERS,
                [r["strategy"], args.num_tokens, f"{r['ttft_ms']:.2f}",
                 f"{r['avg_decode_ms']:.2f}", f"{r['total_time_ms']:.2f}"],
            )

    # Print summary table
    if rank == 0 and results:
        print0(f"\n{'Strategy':<10} {'Tokens':<8} {'TTFT':<10} {'Avg Decode':<12} {'Total':<10}")
        print0("-" * 50)
        for r in results:
            print0(f"{r['strategy']:<10} {args.num_tokens:<8} {r['ttft_ms']:<10.2f} "
                   f"{r['avg_decode_ms']:<12.2f} {r['total_time_ms']:<10.2f}")

    if world_size > 1 and dist.is_initialized():
        dist.barrier()


if __name__ == "__main__":
    try:
        main()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
