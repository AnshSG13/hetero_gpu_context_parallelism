"""Benchmark script for comparing Ring Attention vs Regular Attention."""

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
from fms.utils import tokenizers
from fms.distributed.strategy import NoOpStrategy
from fms.distributed.ring_attention import reset_layer_counter, print_timing_summary

SUMMARY_HEADERS = ["strategy", "prompt_tokens", "ttft_ms", "avg_decode_ms", "total_time_ms"]


def print0(*args, **kwargs):
    if int(os.getenv("RANK", 0)) == 0:
        print(*args, **kwargs)


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark Ring vs Regular Attention")
    script_path = Path(__file__).resolve()
    repo_dir = script_path.parents[2]
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


def setup_model(args, strategy, dtype):
    rank = dist.get_rank() if dist.is_initialized() else 0
    print(f"[Rank {rank}] setup_model ENTER, strategy={strategy}", flush=True)

    # Compute block_lens for ring attention
    block_lens = None
    if strategy == "ring" and dist.is_initialized():
        world_size = dist.get_world_size()
        local_len = args.num_tokens // world_size
        block_lens = [local_len] * world_size
        print(f"[Rank {rank}] setup_model: block_lens={block_lens}", flush=True)

    # For hf_pretrained, don't pass variant or source - let it infer from model_path
    print(f"[Rank {rank}] setup_model: BEFORE models.get_model()", flush=True)
    if args.architecture == "hf_pretrained":
        model = models.get_model(
            args.architecture,
            model_path=args.model_path,
            device_type=args.device_type,
            distributed_strategy=strategy,
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
            distributed_strategy=strategy,
            block_lens=block_lens,
            data_type=dtype
        )
    print(f"[Rank {rank}] setup_model: AFTER models.get_model()", flush=True)
    model.eval()
    torch.set_grad_enabled(False)
    print(f"[Rank {rank}] setup_model EXIT", flush=True)
    return model


def run_benchmark(model, input_ids, num_decode, label, device, is_ring=False):
    """Run generation benchmark. Returns dict with timing metrics."""
    rank = dist.get_rank() if dist.is_initialized() else 0
    print(f"[Rank {rank}] run_benchmark ENTER, label={label}, is_ring={is_ring}", flush=True)
    ids = input_ids.clone().to(device)
    print(f"[Rank {rank}] run_benchmark: ids.shape={ids.shape}, device={ids.device}", flush=True)

    # Reset layer counter for ring attention profiling (if using ring)
    if is_ring:
        reset_layer_counter()

    # Warmup pass
    # print0("Warmup pass")
    # with torch.no_grad():
    #     _ = model.forward(ids, use_cache=False)
    # print("passed synchronize1")
    # if device.type == "cuda":
    #     torch.cuda.synchronize()

    # # Barrier to ensure all ranks finish warmup before starting timed run
    # if is_ring and dist.is_initialized():
    #     dist.barrier()

    # if is_ring:
    #     reset_layer_counter()
    # print0("Warmup done, starting timed run")

    if device.type == "cuda":
        torch.cuda.synchronize()

    # Prefill (TTFT)
    print(f"[Rank {rank}] run_benchmark: BEFORE model.forward()", flush=True)
    t0 = time.perf_counter()
    out = model.forward(ids, use_cache=True)
    print(f"[Rank {rank}] run_benchmark: AFTER model.forward()", flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    print(f'[Rank {rank}] passed synchronize2', flush=True)
    ttft_ms = (time.perf_counter() - t0) * 1000

    logits, cache = (out[0], out[1]) if isinstance(out, tuple) else (out.logits, out.past_key_value_states)
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

        logits, cache = (out[0], out[1]) if isinstance(out, tuple) else (out.logits, out.past_key_value_states)
        last_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

    avg_decode_ms = statistics.mean(decode_times) if decode_times else 0.0
    total_time_ms = ttft_ms + sum(decode_times)

    # Print ring attention timing summary
    if is_ring:
        print_timing_summary(rank)

    if rank == 0:
        print0(f"\n{label}:")
        print0(f"  TTFT: {ttft_ms:.2f} ms | Avg Decode: {avg_decode_ms:.2f} ms | Total: {total_time_ms:.2f} ms")

    return {
        "ttft_ms": ttft_ms, "avg_decode_ms": avg_decode_ms, "total_time_ms": total_time_ms, "logits": logits
    }


def main():
    args = parse_args()
    rank = int(os.getenv("RANK", 0))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))

    # Initialize distributed
    if world_size > 1 and args.device_type == "cuda":
        print(f'[Rank {rank}] multiple gpus found', flush=True)
        torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            print(f"[Rank {rank}] BEFORE init_process_group", flush=True)
            dist.init_process_group(backend="nccl")
            print(f"[Rank {rank}] AFTER init_process_group", flush=True)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device(args.device_type)
    print(f'[Rank {rank}] device={device}', flush=True)
    # Disable FlashAttention if requested (for fair comparison with ring attention)
    if args.disable_flash:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)  # Force naive math backend
        print0("FlashAttention DISABLED - using naive math attention for fair comparison")
    else:
        # Print what backends are available/enabled
        print0(f"SDPA backends: flash={torch.backends.cuda.flash_sdp_enabled()}, "
               f"mem_efficient={torch.backends.cuda.mem_efficient_sdp_enabled()}, "
               f"math={torch.backends.cuda.math_sdp_enabled()}")

    dtype = getattr(torch, args.dtype)
    torch.set_default_dtype(dtype)

    # Create random input tokens (use hardcoded vocab range to avoid tokenizer loading issues)
    # LLaMA vocab is typically 32000-128256, use safe range
    vocab_size = 128256
    ids = torch.randint(100, vocab_size - 100, (args.batch_size, args.num_tokens), dtype=torch.long, device=device)

    # Synchronize random tokens across ranks
    if world_size > 1:
        print(f"[Rank {rank}] BEFORE broadcast ids", flush=True)
        dist.broadcast(ids, src=0)
        print(f"[Rank {rank}] AFTER broadcast ids", flush=True)

    print0(f"Benchmark: {args.num_tokens} prompt tokens, {args.num_decode_tokens} decode tokens")

    # Define strategies
    strategies = [("Ring", "ring"), ("Regular", NoOpStrategy)]
    if not args.run_ring_first:
        strategies.reverse()

    results = []
    for label, strategy in strategies:
        print(f"[Rank {rank}] Processing strategy: {label}", flush=True)
        # Skip Ring if not distributed
        if strategy == "ring" and not dist.is_initialized():
            print0(f"Skipping {label} (requires distributed)")
            continue

        # Regular only runs on rank 0
        is_regular = strategy is NoOpStrategy
        should_run = not (is_regular and rank != 0)
        print(f"[Rank {rank}] should_run={should_run}, is_regular={is_regular}", flush=True)

        if should_run:
            if args.device_type == "cuda":
                torch.cuda.empty_cache()

            # Stagger model loading to avoid I/O contention
            # Rank 0 loads first, then Rank 1
            if dist.is_initialized() and rank > 0:
                print(f"[Rank {rank}] Waiting for Rank 0 to load model first...", flush=True)
                dist.barrier()

            print(f"[Rank {rank}] BEFORE setup_model()", flush=True)
            model = setup_model(args, strategy, dtype)
            print(f"[Rank {rank}] AFTER setup_model()", flush=True)

            # Rank 0 signals it's done loading
            if dist.is_initialized() and rank == 0:
                print(f"[Rank {rank}] Model loaded, signaling other ranks...", flush=True)
                dist.barrier()

            # CRITICAL: Barrier after model loading to sync all ranks
            # before any rank starts forward pass
            if dist.is_initialized():
                try:
                    import sys
                    print(f"[Rank {rank}] BARRIER after model load", flush=True)
                    sys.stdout.flush(); sys.stderr.flush()

                    # Force any pending CUDA errors to surface
                    torch.cuda.synchronize()
                    print(f"[Rank {rank}] CUDA synced, calling barrier()...", flush=True)
                    sys.stdout.flush(); sys.stderr.flush()

                    dist.barrier()
                    print(f"[Rank {rank}] barrier() returned!", flush=True)
                    sys.stdout.flush(); sys.stderr.flush()

                    torch.cuda.synchronize()
                    print(f"[Rank {rank}] PASSED barrier after model load", flush=True)
                    sys.stdout.flush(); sys.stderr.flush()

                except Exception as e:
                    print(f"[Rank {rank}] EXCEPTION at barrier: {e}", flush=True)
                    import traceback
                    traceback.print_exc()
                    raise

            is_ring = (strategy == "ring")
            print(f"[Rank {rank}] is_ring={is_ring}, about to call run_benchmark", flush=True)
            import sys; sys.stdout.flush(); sys.stderr.flush()

            try:
                print(f"[Rank {rank}] BEFORE run_benchmark()", flush=True)
                sys.stdout.flush(); sys.stderr.flush()
                result = run_benchmark(model, ids, args.num_decode_tokens, label, device, is_ring=is_ring)
                print(f"[Rank {rank}] AFTER run_benchmark()", flush=True)
            except Exception as e:
                print(f"[Rank {rank}] EXCEPTION in run_benchmark: {e}", flush=True)
                import traceback
                traceback.print_exc()
                raise
            result["strategy"] = label
            results.append(result)

            del model
            gc.collect()
            if args.device_type == "cuda":
                torch.cuda.empty_cache()

        # ALL ranks sync here (same barrier for everyone)
        if world_size > 1:
            dist.barrier()

    # Write summary CSV
    if rank == 0 and args.summary_csv and results:
        file_exists = os.path.exists(args.summary_csv)
        with open(args.summary_csv, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(SUMMARY_HEADERS)
            for r in results:
                writer.writerow([r["strategy"], args.num_tokens, f"{r['ttft_ms']:.2f}",
                                f"{r['avg_decode_ms']:.2f}", f"{r['total_time_ms']:.2f}"])

    # Print summary table
    if rank == 0 and results:
        print0(f"\n{'Strategy':<10} {'Tokens':<8} {'TTFT':<10} {'Avg Decode':<12} {'Total':<10}")
        print0("-" * 50)
        for r in results:
            print0(f"{r['strategy']:<10} {args.num_tokens:<8} {r['ttft_ms']:<10.2f} {r['avg_decode_ms']:<12.2f} {r['total_time_ms']:<10.2f}")
    print("printed results")
    if world_size > 1 and dist.is_initialized():
        print("hanging at dist.barrier()")
        dist.barrier()
if __name__ == "__main__":
    try:
        main()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
