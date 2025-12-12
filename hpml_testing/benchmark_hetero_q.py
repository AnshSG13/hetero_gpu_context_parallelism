"""Benchmark heterogeneous Q distribution in Ring Attention.

Tests how different Q/KV split ratios affect performance and load balancing.
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

from fms import models
from fms.distributed.strategy import NoOpStrategy
from fms.distributed.ring_attention import reset_layer_counter, print_timing_summary

SUMMARY_HEADERS = ["split_ratio", "gpu0_tokens", "gpu1_tokens", "total_tokens", "ttft_ms", "gpu0_time_ms", "gpu1_time_ms"]


def print0(*args, **kwargs):
    if int(os.getenv("RANK", 0)) == 0:
        print(*args, **kwargs)


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark Heterogeneous Q Distribution")
    script_path = Path(__file__).resolve()
    repo_dir = script_path.parents[2]
    model_dir = repo_dir.parent / "llama-hf"

    parser.add_argument("--device_type", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--architecture", type=str, default="hf_pretrained")
    parser.add_argument("--variant", type=str, default="8b")
    parser.add_argument("--model_path", type=str, default=str(model_dir))
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_tokens", type=int, required=True, help="Total number of tokens")
    parser.add_argument("--gpu0_tokens", type=int, required=True, help="Tokens for GPU 0")
    parser.add_argument("--gpu1_tokens", type=int, required=True, help="Tokens for GPU 1")
    parser.add_argument("--num_decode_tokens", type=int, default=0, help="Number of tokens to decode")
    parser.add_argument("--summary_csv", type=str, default=None, help="Summary CSV path (appends)")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float32", "float16", "bfloat16"])

    return parser.parse_args()


def setup_model(args, block_lens, dtype):
    """Load model with heterogeneous Q distribution."""
    if args.architecture == "hf_pretrained":
        model = models.get_model(
            args.architecture,
            model_path=args.model_path,
            device_type=args.device_type,
            distributed_strategy="ring",
            block_lens=block_lens,
            data_type=dtype
        )
    else:
        model = models.get_model(
            args.architecture,
            args.variant,
            model_path=args.model_path,
            device_type=args.device_type,
            source="hf",
            distributed_strategy="ring",
            block_lens=block_lens,
            data_type=dtype
        )
    model.eval()
    torch.set_grad_enabled(False)
    return model


def run_benchmark(model, input_ids, num_decode, device):
    """Run generation benchmark. Returns dict with timing metrics."""
    rank = dist.get_rank() if dist.is_initialized() else 0
    ids = input_ids.clone().to(device)

    # Reset layer counter for ring attention profiling
    reset_layer_counter()

    # Warmup pass
    print0("Warmup pass")
    with torch.no_grad():
        _ = model.forward(ids, use_cache=False)
    if device.type == "cuda":
        torch.cuda.synchronize()
    reset_layer_counter()
    print0("Warmup done, starting timed run")

    if device.type == "cuda":
        torch.cuda.synchronize()

    # Prefill (TTFT)
    t0 = time.perf_counter()
    out = model.forward(ids, use_cache=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    ttft_ms = (time.perf_counter() - t0) * 1000

    # Gather timing from all ranks
    ttft_tensor = torch.tensor([ttft_ms], device=device)
    all_ttft = [torch.zeros_like(ttft_tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(all_ttft, ttft_tensor)

    gpu0_time = all_ttft[0].item()
    gpu1_time = all_ttft[1].item()

    # Print ring attention timing summary
    print_timing_summary(rank)

    if rank == 0:
        print0(f"\nResults:")
        print0(f"  Total TTFT: {max(gpu0_time, gpu1_time):.2f} ms (bottleneck)")
        print0(f"  GPU 0 time: {gpu0_time:.2f} ms")
        print0(f"  GPU 1 time: {gpu1_time:.2f} ms")
        print0(f"  Load balance: {min(gpu0_time, gpu1_time) / max(gpu0_time, gpu1_time) * 100:.1f}%")

    return {
        "ttft_ms": max(gpu0_time, gpu1_time),
        "gpu0_time_ms": gpu0_time,
        "gpu1_time_ms": gpu1_time,
    }


def main():
    args = parse_args()
    rank = int(os.getenv("RANK", 0))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))

    # Initialize distributed
    if world_size > 1 and args.device_type == "cuda":
        torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device(args.device_type)

    # Print SDPA backends
    if args.device_type == "cuda":
        print0(f"SDPA backends: flash={torch.backends.cuda.flash_sdp_enabled()}, "
               f"mem_efficient={torch.backends.cuda.mem_efficient_sdp_enabled()}, "
               f"math={torch.backends.cuda.math_sdp_enabled()}")

    dtype = getattr(torch, args.dtype)
    torch.set_default_dtype(dtype)

    # Validate block lengths
    assert args.gpu0_tokens + args.gpu1_tokens == args.num_tokens, \
        f"gpu0_tokens ({args.gpu0_tokens}) + gpu1_tokens ({args.gpu1_tokens}) != num_tokens ({args.num_tokens})"

    block_lens = [args.gpu0_tokens, args.gpu1_tokens]
    split_ratio = args.gpu0_tokens / args.num_tokens

    print0(f"Benchmark: {args.num_tokens} total tokens")
    print0(f"Split ratio: {split_ratio:.1%} / {1-split_ratio:.1%}")
    print0(f"Block lengths: {block_lens}")

    # Create random input tokens
    vocab_size = 128256
    ids = torch.randint(100, vocab_size - 100, (args.batch_size, args.num_tokens), dtype=torch.long, device=device)

    # Synchronize random tokens across ranks
    if world_size > 1:
        dist.broadcast(ids, src=0)

    # Load model
    print0("Loading model...")
    if args.device_type == "cuda":
        torch.cuda.empty_cache()

    model = setup_model(args, block_lens, dtype)
    print0("Model loaded")

    # Run benchmark
    result = run_benchmark(model, ids, args.num_decode_tokens, device)

    # Write summary CSV
    if rank == 0 and args.summary_csv:
        file_exists = os.path.exists(args.summary_csv)
        with open(args.summary_csv, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(SUMMARY_HEADERS)
            writer.writerow([
                f"{split_ratio:.1%}",
                args.gpu0_tokens,
                args.gpu1_tokens,
                args.num_tokens,
                f"{result['ttft_ms']:.2f}",
                f"{result['gpu0_time_ms']:.2f}",
                f"{result['gpu1_time_ms']:.2f}",
            ])

    # Cleanup
    del model
    gc.collect()
    if args.device_type == "cuda":
        torch.cuda.empty_cache()

    if world_size > 1:
        dist.barrier()


if __name__ == "__main__":
    try:
        main()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
