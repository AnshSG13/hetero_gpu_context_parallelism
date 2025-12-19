import argparse
import math
from typing import List, Tuple

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-GPU matmul benchmark")
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        required=True,
        help="Square matrix sizes to benchmark (e.g., 4096 8192).",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Datatype to use for matmul inputs.",
    )
    parser.add_argument(
        "--num-iters",
        type=int,
        default=20,
        help="Number of timed iterations per size.",
    )
    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=5,
        help="Warmup iterations before timing.",
    )
    parser.add_argument(
        "--background-streams",
        type=int,
        default=0,
        help="Number of additional CUDA streams running the same matmul "
        "to create contention.",
    )
    return parser.parse_args()


def _maybe_init_background(
    n: int, dtype: torch.dtype, device: torch.device, num_streams: int
) -> Tuple[List[torch.cuda.Stream], List[Tuple[torch.Tensor, torch.Tensor]]]:
    if num_streams == 0:
        return [], []
    streams = [torch.cuda.Stream() for _ in range(num_streams)]
    mats = []
    for _ in range(num_streams):
        a_bg = torch.randn(n, n, device=device, dtype=dtype)
        b_bg = torch.randn(n, n, device=device, dtype=dtype)
        mats.append((a_bg, b_bg))
    return streams, mats


def run_case(
    n: int,
    dtype: torch.dtype,
    device: torch.device,
    num_iters: int,
    warmup_iters: int,
    background_streams: int,
) -> Tuple[float, float]:
    a = torch.randn(n, n, device=device, dtype=dtype)
    b = torch.randn(n, n, device=device, dtype=dtype)

    bg_streams, bg_mats = _maybe_init_background(
        n, dtype, device, background_streams
    )

    # Warmup
    for _ in range(warmup_iters):
        _ = a @ b
        for stream, (a_bg, b_bg) in zip(bg_streams, bg_mats):
            with torch.cuda.stream(stream):
                _ = a_bg @ b_bg
    torch.cuda.synchronize()

    # Timed runs
    times_ms: List[float] = []
    for _ in range(num_iters):
        for stream, (a_bg, b_bg) in zip(bg_streams, bg_mats):
            with torch.cuda.stream(stream):
                _ = a_bg @ b_bg

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _ = a @ b
        end.record()
        torch.cuda.synchronize()
        times_ms.append(start.elapsed_time(end))

    avg_ms = float(sum(times_ms) / len(times_ms))
    flops = 2 * (n**3)  # GEMM: 2*N^3 operations
    tflops = flops / (avg_ms / 1000.0) / 1e12
    return avg_ms, tflops


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark.")

    device = torch.device("cuda")
    dtype = getattr(torch, args.dtype)

    header = (
        f"{'size':>8} | {'dtype':>8} | {'bg_streams':>11} | "
        f"{'avg_ms':>10} | {'TFLOPs':>10}"
    )
    print(header)
    print("-" * len(header))

    for n in args.sizes:
        avg_ms, tflops = run_case(
            n=n,
            dtype=dtype,
            device=device,
            num_iters=args.num_iters,
            warmup_iters=args.warmup_iters,
            background_streams=args.background_streams,
        )
        print(
            f"{n:8d} | {args.dtype:>8} | {args.background_streams:11d} | "
            f"{avg_ms:10.2f} | {tflops:10.2f}"
        )


if __name__ == "__main__":
    main()
