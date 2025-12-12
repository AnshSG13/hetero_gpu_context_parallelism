"""Benchmark script for testing Regular Attention only."""

import argparse
import os
import statistics
import time
import torch
from pathlib import Path

from fms import models
from fms.distributed.strategy import NoOpStrategy


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark Regular Attention")
    script_path = Path(__file__).resolve()
    repo_dir = script_path.parents[2]
    model_dir = repo_dir.parent / "llama-hf"

    parser.add_argument("--device_type", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--architecture", type=str, default="hf_pretrained")
    parser.add_argument("--variant", type=str, default="8b")
    parser.add_argument("--model_path", type=str, default=str(model_dir))
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_tokens", type=int, required=True, help="Number of prompt tokens")
    parser.add_argument("--num_decode_tokens", type=int, default=0, help="Number of tokens to decode")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--disable_flash", action="store_true", default=False,
                        help="Disable FlashAttention")

    return parser.parse_args()


def setup_model(args, dtype):
    """Load model with regular attention (no distributed)."""
    # Use same logic as benchmark_ring.py
    if args.architecture == "hf_pretrained":
        model = models.get_model(
            args.architecture,
            model_path=args.model_path,
            device_type=args.device_type,
            distributed_strategy=NoOpStrategy,
            data_type=dtype
        )
    else:
        model = models.get_model(
            args.architecture,
            args.variant,
            model_path=args.model_path,
            device_type=args.device_type,
            source="hf",
            distributed_strategy=NoOpStrategy,
            data_type=dtype
        )
    model.eval()
    torch.set_grad_enabled(False)
    return model


def run_benchmark(model, input_ids, num_decode, device):
    """Run generation benchmark. Returns dict with timing metrics."""
    ids = input_ids.clone().to(device)

    # Warmup pass
    print("Warmup pass")
    with torch.no_grad():
        _ = model.forward(ids, use_cache=False)
    if device.type == "cuda":
        torch.cuda.synchronize()
    print("Warmup done, starting timed run")

    if device.type == "cuda":
        torch.cuda.synchronize()

    # Prefill (TTFT)
    t0 = time.perf_counter()
    out = model.forward(ids, use_cache=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    ttft_ms = (time.perf_counter() - t0) * 1000

    logits, cache = (out[0], out[1]) if isinstance(out, tuple) else (out.logits, out.past_key_value_states)
    last_token = ids[:, -1:]

    # Decode
    decode_times = []
    for i in range(num_decode):
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

    return {
        "ttft_ms": ttft_ms,
        "avg_decode_ms": avg_decode_ms,
        "total_time_ms": total_time_ms
    }


def main():
    args = parse_args()

    # Single GPU setup (no distributed)
    if args.device_type == "cuda":
        device = torch.device("cuda:0")
        torch.cuda.set_device(0)
    else:
        device = torch.device(args.device_type)

    # Configure attention backend
    if args.disable_flash:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        print("FlashAttention DISABLED - using naive math attention")
    else:
        print(f"SDPA backends: flash={torch.backends.cuda.flash_sdp_enabled()}, "
              f"mem_efficient={torch.backends.cuda.mem_efficient_sdp_enabled()}, "
              f"math={torch.backends.cuda.math_sdp_enabled()}")

    dtype = getattr(torch, args.dtype)
    torch.set_default_dtype(dtype)

    # Create random input tokens
    vocab_size = 128256
    ids = torch.randint(100, vocab_size - 100, (args.batch_size, args.num_tokens),
                        dtype=torch.long, device=device)

    print(f"Benchmark: {args.num_tokens} prompt tokens, {args.num_decode_tokens} decode tokens")

    # Load model
    print("Loading model...")
    model = setup_model(args, dtype)
    print("Model loaded")

    # Run benchmark
    result = run_benchmark(model, ids, args.num_decode_tokens, device)

    # Print results
    print(f"\n{'='*60}")
    print(f"RESULTS - Regular Attention")
    print(f"{'='*60}")
    print(f"Prompt tokens:     {args.num_tokens}")
    print(f"TTFT:              {result['ttft_ms']:.2f} ms")
    print(f"Avg decode time:   {result['avg_decode_ms']:.2f} ms")
    print(f"Total time:        {result['total_time_ms']:.2f} ms")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
