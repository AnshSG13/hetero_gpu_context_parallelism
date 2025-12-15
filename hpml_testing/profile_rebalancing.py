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

# --- Profiling Configuration ---
SEQ_LEN = 8192
EMB_DIM = 4096
N_HEADS = 32
N_STEPS = 5 # A few steps are enough for profiling
# -----------------------------

# --- Copied Helper Functions ---
# These are the same helpers from benchmark_hetero_latency.py
def setup_distributed(rank, world_size):
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        init_method="tcp://127.0.0.1:29500"
    )
    torch.cuda.set_device(rank)

def get_model_and_input(rank, world_size, block_lens):
    head_dim = EMB_DIM // N_HEADS
    rope = RotaryEmbedding(dim=head_dim)
    attn_module = MultiHeadAttention(
        EMB_DIM, emb_kq=head_dim, emb_v=head_dim, nheads=N_HEADS,
        kvheads=N_HEADS, position_encoder=rope
    ).cuda().to(torch.bfloat16)

    full_input = torch.randn(1, SEQ_LEN, EMB_DIM, device="cuda", dtype=torch.bfloat16)
    strategy = RingAttentionStrategy(block_lens=block_lens)
    local_input = strategy.shard_input(full_input)
    
    return attn_module, local_input, strategy

def load_performance_profile(profile_path, size):
    if not os.path.exists(profile_path):
        raise FileNotFoundError(f"Performance profile not found: {profile_path}")
    df = pd.read_csv(profile_path)
    available_sizes = sorted(df['size'].unique())
    closest_size = min(available_sizes, key=lambda x: abs(x - size))
    print(f"Using size {closest_size} from performance profile for target size {size}")
    df_filtered = df[df['size'] == closest_size]
    return df_filtered.set_index('mps_pct')['tflops'].to_dict()

def get_performance_for_mps(profile, mps_pct):
    sorted_mps = sorted(profile.keys())
    if mps_pct in profile: return profile[mps_pct]
    if mps_pct > sorted_mps[-1]: return profile[sorted_mps[-1]]
    if mps_pct < sorted_mps[0]: return profile[sorted_mps[0]]
    for i, p in enumerate(sorted_mps):
        if p > mps_pct:
            high_p = p
            low_p = sorted_mps[i-1]
            break
    high_val, low_val = profile[high_p], profile[low_p]
    weight = (mps_pct - low_p) / (high_p - low_p)
    return low_val + weight * (high_val - low_val)
# --- End Helper Functions ---


def run_profiling(rank, world_size, n_steps, attn_module, local_input, strategy, prof):
    """
    Runs the attention pass for a few steps under the profiler.
    """
    for step in range(n_steps):
        _ring_attention_pass_kv(
            x_norm=local_input,
            attn_module=attn_module,
            strategy=strategy,
            valid_len=strategy.local_q_len,
            causal=True
        )
        dist.barrier()
        # The profiler needs to be stepped at the end of each iteration
        prof.step()


def main():
    parser = argparse.ArgumentParser(description="Heterogeneous Ring Attention Profiling")
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--split-type", type=str, choices=["even", "uneven", "lut"], required=True)
    parser.add_argument("--slowdown-factor", type=float, default=0.5)
    parser.add_argument("--use-perf-profile", type=str, default="hpml_testing/results/matmul_mps_sweep.csv")
    parser.add_argument("--rank-mps", type=str, default="100,50")
    parser.add_argument("--output-file", type=str, required=True, help="Output file for the profiler trace (.json)")
    
    args = parser.parse_args()

    setup_distributed(args.rank, args.world_size)

    # Determine block lengths based on split type
    block_lens = []
    if args.split_type == "even":
        base_len = SEQ_LEN // args.world_size
        block_lens = [base_len] * args.world_size
        remainder = SEQ_LEN % args.world_size
        for i in range(remainder):
            block_lens[i] += 1
    else: # 'uneven' or 'lut'
        weights = []
        if args.split_type == "uneven":
            weights = [1.0, args.slowdown_factor]
        else: # 'lut'
            perf_profile = load_performance_profile(args.use_perf_profile, SEQ_LEN)
            rank_mps_list = [int(p) for p in args.rank_mps.split(',')]
            weights = [get_performance_for_mps(perf_profile, mps) for mps in rank_mps_list]
        
        total_weight = sum(weights)
        for i in range(args.world_size):
            block_lens.append(int(round(SEQ_LEN * (weights[i] / total_weight))))
        diff = sum(block_lens) - SEQ_LEN
        block_lens[-1] -= diff

    attn_module, local_input, strategy = get_model_and_input(args.rank, args.world_size, block_lens)

    if args.rank == 0:
        print(f"Profiling with '{args.split_type}' split.")
        print(f"Sequence Length: {SEQ_LEN}, Block lengths: {block_lens}")

    reset_layer_counter()
    dist.barrier()
    
    # Setup the profiler
    # We run 5 steps: 1 wait, 1 warmup, 3 active recording steps
    schedule = torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1)
    
    def trace_handler(p):
        # Suffix the filename with the rank to save traces for both ranks.
        base_name, ext = os.path.splitext(args.output_file)
        output_path = f"{base_name}_rank{args.rank}{ext}"
        print(f"Rank {args.rank} saving profiler trace to {output_path}...")
        p.export_chrome_trace(output_path)
        print(f"Rank {args.rank} trace saved.")

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=schedule,
        on_trace_ready=trace_handler,
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        run_profiling(args.rank, args.world_size, N_STEPS, attn_module, local_input, strategy, prof)


if __name__ == "__main__":
    main()
