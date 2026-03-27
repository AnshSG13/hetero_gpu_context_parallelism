"""
Test for ring attention prefill (pass-KV) with async overlap.
"""
import os
import torch
import torch.distributed as dist
import argparse
import time
import math # Import math for math.sqrt

def main():
    parser = argparse.ArgumentParser(description="Ring Attention Benchmark")
    parser.add_argument("--total_seq_len", type=int, default=16384, help="Total sequence length to distribute across GPUs.")
    args = parser.parse_args()

    # Initialize distributed - environment variables are set by the launcher script
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    
    if "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE" in os.environ:
        print(f"Rank {rank}: Detected CUDA_MPS_ACTIVE_THREAD_PERCENTAGE={os.environ['CUDA_MPS_ACTIVE_THREAD_PERCENTAGE']}%\n")

    # Import after dist init
    from fms.distributed.strategy import RingAttentionStrategy
    from fms.distributed.ring_attention import _compute_attention_ring_pass_kv

    # Test config
    batch_size = 1
    total_seq_len = args.total_seq_len
    nheads = 8
    head_dim = 64
    
    scale = head_dim ** 0.5
    accum_dtype = torch.float32
    causal = True

    # Setup strategy - check for BLOCK_LENS env var for heterogeneous sharding
    block_lens_str = os.environ.get("BLOCK_LENS", None)
    if block_lens_str:
        # Parse comma-separated block lengths (e.g., "12288,4096")
        block_lens = [int(x) for x in block_lens_str.split(",")]
        assert len(block_lens) == world_size, f"BLOCK_LENS has {len(block_lens)} entries but world_size is {world_size}"
        if rank == 0:
            print(f"Heterogeneous sharding: block_lens={block_lens}")
    else:
        # Even split
        local_seq_len = total_seq_len // world_size
        block_lens = [local_seq_len] * world_size
        if rank == 0:
            print(f"Even sharding: block_lens={block_lens}")

    strategy = RingAttentionStrategy(block_lens=block_lens, group=None)
    local_seq_len = block_lens[rank]  # This rank's sequence length
    q_start = strategy.block_starts[rank]

    # Create local Q, K, V tensors for this rank
    torch.manual_seed(42 + rank)
    q = torch.randn(batch_size, nheads, local_seq_len, head_dim, device=device, dtype=torch.float16)
    k = torch.randn(batch_size, nheads, local_seq_len, head_dim, device=device, dtype=torch.float16)
    v = torch.randn(batch_size, nheads, local_seq_len, head_dim, device=device, dtype=torch.float16)

    if rank == 0:
        print(f"Rank {rank}: Q shape={q.shape}, K shape={k.shape}, V shape={v.shape}")
        print(f"Rank {rank}: q_start={q_start}, local_seq_len={local_seq_len}")

    # Run ring attention once for correctness check if needed
    out = _compute_attention_ring_pass_kv(
        q, k, v, None, strategy, q_start, local_seq_len, scale, accum_dtype, causal
    )
    if rank == 0:
        print(f"Rank {rank}: Initial output sum={out.sum().item():.4f}")

    # Simple timing test
    torch.cuda.synchronize()

    # Warmup
    for _ in range(3):
        _ = _compute_attention_ring_pass_kv(
            q, k, v, None, strategy, q_start, local_seq_len, scale, accum_dtype, causal
        )
    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    num_iters = 10
    for _ in range(num_iters):
        out = _compute_attention_ring_pass_kv(
            q, k, v, None, strategy, q_start, local_seq_len, scale, accum_dtype, causal
        )
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / num_iters * 1000

    if rank == 0:
        print(f"\n--- Benchmark Summary ---")
        print(f"GPUs: {world_size}")
        print(f"Total seq_len: {total_seq_len}")
        print(f"Block lens: {block_lens}")
        print(f"Avg time per call: {elapsed:.2f} ms")
        print(f"--- End Summary ---\n")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()