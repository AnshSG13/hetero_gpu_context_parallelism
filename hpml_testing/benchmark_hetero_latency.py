import torch
import torch.distributed as dist
import argparse
import time
import math
import pandas as pd
import os

from fms.distributed.strategy import RingAttentionStrategy
from fms.modules.attention import MultiHeadAttention
from fms.modules.positions import RotaryEmbedding
from fms.distributed.ring_attention import _ring_attention_pass_kv, reset_layer_counter

def setup_distributed(rank, world_size):
    """Initializes torch.distributed."""
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        init_method="tcp://127.0.0.1:29500"
    )
    torch.cuda.set_device(rank)

def get_model_and_input(rank, world_size, seq_len, n_heads, emb_dim, block_lens):
    """Creates a dummy attention module and input tensor."""
    
    head_dim = emb_dim // n_heads
    
    # Correctly instantiate the position encoder
    rope = RotaryEmbedding(dim=head_dim)
    
    # This would normally be part of a larger model
    attn_module = MultiHeadAttention(
        emb_dim,
        emb_kq=head_dim,
        emb_v=head_dim,
        nheads=n_heads,
        kvheads=n_heads,
        position_encoder=rope
    ).cuda().to(torch.bfloat16)

    # Create dummy input data for the entire sequence
    full_input = torch.randn(
        1, seq_len, emb_dim, device="cuda", dtype=torch.bfloat16
    )
    
    # The strategy will shard the input for us
    strategy = RingAttentionStrategy(block_lens=block_lens)
    
    local_input = strategy.shard_input(full_input)
    
    return attn_module, local_input, strategy

def run_benchmark(rank, world_size, n_steps, attn_module, local_input, strategy):
    """Runs the benchmark and returns the average latency."""
    # Warmup runs
    for _ in range(5):
        _ring_attention_pass_kv(
            x_norm=local_input,
            attn_module=attn_module,
            strategy=strategy,
            valid_len=strategy.local_q_len,
            causal=True
        )
        dist.barrier()

    torch.cuda.synchronize()
    
    # Timed runs
    start_time = time.time()
    for _ in range(n_steps):
        _ring_attention_pass_kv(
            x_norm=local_input,
            attn_module=attn_module,
            strategy=strategy,
            valid_len=strategy.local_q_len,
            causal=True
        )
    
    # We need to sync/barrier here to make sure all ranks have finished
    # their work before stopping the timer.
    dist.barrier()
    torch.cuda.synchronize()
    end_time = time.time()

    avg_latency_ms = (end_time - start_time) / n_steps * 1000
    return avg_latency_ms

def load_performance_profile(profile_path, size):
    """Loads a performance profile from a CSV file."""
    if not os.path.exists(profile_path):
        raise FileNotFoundError(f"Performance profile not found: {profile_path}")
    
    df = pd.read_csv(profile_path)
    
    # Get the closest available size in the dataframe
    available_sizes = sorted(df['size'].unique())
    closest_size = min(available_sizes, key=lambda x: abs(x - size))
    
    print(f"Using size {closest_size} from performance profile for target size {size}")
    
    df_filtered = df[df['size'] == closest_size]
    
    if df_filtered.empty:
        raise ValueError(f"No data found for size {closest_size} in the performance profile.")
        
    return df_filtered.set_index('mps_pct')['tflops'].to_dict()

def main():
    parser = argparse.ArgumentParser(description="Heterogeneous Ring Attention Benchmark")
    parser.add_argument("--rank", type=int, required=True, help="Rank of the process")
    parser.add_argument("--world-size", type=int, required=True, help="Total number of processes")
    parser.add_argument("--seq-len", type=int, default=4096, help="Total sequence length")
    parser.add_argument("--n-heads", type=int, default=32, help="Number of attention heads")
    parser.add_argument("--emb-dim", type=int, default=4096, help="Embedding dimension")
    parser.add_argument("--n-steps", type=int, default=20, help="Number of benchmark iterations")
    parser.add_argument("--split-type", type=str, choices=["even", "uneven", "lut"], default="even", help="Workload split type")
    parser.add_argument("--slowdown-factor", type=float, default=0.5, help="Proportional slowdown of the weak GPU (e.g., 0.5 for 50%)")
    parser.add_argument("--use-perf-profile", type=str, default=None, help="Path to the performance profile CSV file.")
    parser.add_argument("--rank-mps", type=str, default="100,50", help="Comma-separated list of MPS percentages for each rank.")

    args = parser.parse_args()

    setup_distributed(args.rank, args.world_size)

    # Determine block lengths based on split type
    if args.split_type == "even":
        base_len = args.seq_len // args.world_size
        block_lens = [base_len] * args.world_size
        # Adjust for remainder
        remainder = args.seq_len % args.world_size
        for i in range(remainder):
            block_lens[i] += 1
    elif args.split_type == "uneven":
        # We assume rank 1 is the slower GPU
        weights = [1.0] * args.world_size
        if args.rank == 1:
             weights[1] = args.slowdown_factor
        
        # This logic must be consistent across ranks
        weights = [1.0, args.slowdown_factor]
        total_weight = sum(weights)
        
        # Calculate tokens for each rank based on weight
        block_lens = []
        for i in range(args.world_size):
            block_lens.append(int(round(args.seq_len * (weights[i] / total_weight))))
        
        # Ensure total length is correct
        diff = sum(block_lens) - args.seq_len
        block_lens[-1] -= diff
    elif args.split_type == "lut":
        if not args.use_perf_profile:
            raise ValueError("Performance profile must be specified for 'lut' split type.")
        
        perf_profile = load_performance_profile(args.use_perf_profile, args.seq_len)
        
        rank_mps_list = [int(p) for p in args.rank_mps.split(',')]
        
        if len(rank_mps_list) != args.world_size:
            raise ValueError("Number of MPS percentages must match world size.")
            
        weights = [perf_profile[mps] for mps in rank_mps_list]
        total_weight = sum(weights)

        block_lens = []
        for i in range(args.world_size):
            block_lens.append(int(round(args.seq_len * (weights[i] / total_weight))))

        diff = sum(block_lens) - args.seq_len
        block_lens[-1] -= diff

    attn_module, local_input, strategy = get_model_and_input(
        args.rank, args.world_size, args.seq_len, args.n_heads, args.emb_dim, block_lens
    )

    if args.rank == 0:
        print(f"Running benchmark with '{args.split_type}' split.")
        print(f"Sequence Length: {args.seq_len}, Block lengths: {block_lens}")

    # Everyone resets their counters before the benchmark
    reset_layer_counter()
    dist.barrier()

    latency = run_benchmark(args.rank, args.world_size, args.n_steps, attn_module, local_input, strategy)

    # Gather results to rank 0
    output = [None] * args.world_size
    dist.gather_object(
        {"rank": args.rank, "latency": latency, "tokens": strategy.local_q_len},
        output if args.rank == 0 else None,
        dst=0
    )

    if args.rank == 0:
        total_latency = 0
        print("\n--- Results ---")
        for res in output:
            print(f"Rank {res['rank']} ({res['tokens']} tokens): {res['latency']:.2f} ms")
            # The overall latency is the max latency of any rank
            total_latency = max(total_latency, res['latency'])
        print(f"Overall Latency (max of ranks): {total_latency:.2f} ms")
        print("-----------------\n")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()

