import os
import subprocess
import re
import pandas as pd
import math
import time

# --- Configuration ---
SEQLEN_SWEEP = [4096, 8192, 16384, 32768]
SLOWDOWN_SWEEP = [70, 50, 30, 20, 10] # MPS percentage for the slower GPU
PERF_PROFILE_PATH = "hpml_testing/results/matmul_mps_sweep.csv"
OUTPUT_CSV_PATH = "hpml_testing/results/sweep_results.csv"
BENCHMARK_SCRIPT = "hpml_testing/benchmark_hetero_latency.py"

# --- Helper Functions ---
def parse_single_benchmark_output(output_string):
    """
    Parses the output of a single benchmark_hetero_latency.py execution (rank 0's output).
    Extracts overall latency, individual rank latencies, and token counts.
    """
    results = {}

    # Example output:
    # Running benchmark with 'even' split.
    # Sequence Length: 8192, Block lengths: [4096, 4096]
    # ...
    # --- Results ---
    # Rank 0 (4096 tokens): 87.45 ms
    # Rank 1 (4096 tokens): 87.45 ms
    # Overall Latency (max of ranks): 87.45 ms
    # -----------------

    split_type_match = re.search(r"Running benchmark with '(\w+)' split.", output_string)
    if split_type_match:
        results['split_type'] = split_type_match.group(1)
    else:
        results['split_type'] = "unknown" # Should not happen

    seq_len_match = re.search(r"Sequence Length: (\d+), Block lengths: [^]]*]", output_string)
    if seq_len_match:
        results['seq_len'] = int(seq_len_match.group(1))

    overall_latency_match = re.search(r"Overall Latency \(max of ranks\): (\d+\.\d+) ms", output_string)
    if overall_latency_match:
        results['overall_latency_ms'] = float(overall_latency_match.group(1))

    rank_latency_token_matches = re.findall(r"Rank (\d+) \((\d+) tokens\): (\d+\.\d+) ms", output_string)
    for match in rank_latency_token_matches:
        rank = int(match[0])
        tokens = int(match[1])
        latency = float(match[2])
        results[f'rank{rank}_tokens'] = tokens
        results[f'rank{rank}_latency_ms'] = latency
        
    return results

# --- Main Sweep Logic ---
def main():
    print("Starting benchmark sweep...")
    
    # Set CUDA_VISIBLE_DEVICES to ensure consistency
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

    all_sweep_results = []

    # Ensure python path includes current directory for fms imports
    os.environ['PYTHONPATH'] = os.getcwd() + ':' + os.environ.get('PYTHONPATH', '')

    try:
        for seq_len in SEQLEN_SWEEP:
            for slowdown_pct in SLOWDOWN_SWEEP:
                print(f"\n--- Running: SEQ_LEN={seq_len}, SLOWDOWN_PCT={slowdown_pct} ---")
                current_slowdown_factor = slowdown_pct / 100.0

                # --- Run Even Split ---
                print("Running Even Split...")
                p0_even = subprocess.Popen(
                    ["python3", BENCHMARK_SCRIPT, 
                     "--rank", "0", "--world-size", "2", 
                     "--seq-len", str(seq_len), "--emb-dim", "4096", "--n-heads", "32", 
                     "--split-type", "even"],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
                )
                p1_even = subprocess.Popen(
                    ["python3", BENCHMARK_SCRIPT, 
                     "--rank", "1", "--world-size", "2", 
                     "--seq-len", str(seq_len), "--emb-dim", "4096", "--n-heads", "32", 
                     "--split-type", "even"],
                    env={**os.environ, "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE": str(slowdown_pct)},
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
                )
                stdout_even_rank0, stderr_even_rank0 = p0_even.communicate()
                _, stderr_even_rank1 = p1_even.communicate() # Only need rank0 output for summary
                
                if p0_even.returncode != 0:
                    print(f"Error in even split (rank 0): {stderr_even_rank0}")
                if p1_even.returncode != 0:
                    print(f"Error in even split (rank 1): {stderr_even_rank1}")

                even_results = parse_single_benchmark_output(stdout_even_rank0)
                even_results.update({
                    "seq_len": seq_len,
                    "slowdown_pct": slowdown_pct,
                    "split_type": "even", # Override parsed to ensure consistency
                })
                all_sweep_results.append(even_results)


                # --- Run Uneven Split ---
                print("Running Uneven Split (factor-based)...")
                p0_uneven = subprocess.Popen(
                    ["python3", BENCHMARK_SCRIPT, 
                     "--rank", "0", "--world-size", "2", 
                     "--seq-len", str(seq_len), "--emb-dim", "4096", "--n-heads", "32", 
                     "--split-type", "uneven", "--slowdown-factor", str(current_slowdown_factor)],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
                )
                p1_uneven = subprocess.Popen(
                    ["python3", BENCHMARK_SCRIPT, 
                     "--rank", "1", "--world-size", "2", 
                     "--seq-len", str(seq_len), "--emb-dim", "4096", "--n-heads", "32", 
                     "--split-type", "uneven", "--slowdown-factor", str(current_slowdown_factor)],
                    env={**os.environ, "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE": str(slowdown_pct)},
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
                )
                stdout_uneven_rank0, stderr_uneven_rank0 = p0_uneven.communicate()
                _, stderr_uneven_rank1 = p1_uneven.communicate()

                if p0_uneven.returncode != 0:
                    print(f"Error in uneven split (rank 0): {stderr_uneven_rank0}")
                if p1_uneven.returncode != 0:
                    print(f"Error in uneven split (rank 1): {stderr_uneven_rank1}")
                
                uneven_results = parse_single_benchmark_output(stdout_uneven_rank0)
                uneven_results.update({
                    "seq_len": seq_len,
                    "slowdown_pct": slowdown_pct,
                    "split_type": "uneven",
                })
                all_sweep_results.append(uneven_results)


                # --- Run LUT Split ---
                print("Running LUT Split...")
                p0_lut = subprocess.Popen(
                    ["python3", BENCHMARK_SCRIPT, 
                     "--rank", "0", "--world-size", "2", 
                     "--seq-len", str(seq_len), "--emb-dim", "4096", "--n-heads", "32", 
                     "--split-type", "lut", 
                     "--use-perf-profile", PERF_PROFILE_PATH, 
                     "--rank-mps", f"100,{slowdown_pct}"],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
                )
                p1_lut = subprocess.Popen(
                    ["python3", BENCHMARK_SCRIPT, 
                     "--rank", "1", "--world-size", "2", 
                     "--seq-len", str(seq_len), "--emb-dim", "4096", "--n-heads", "32", 
                     "--split-type", "lut", 
                     "--use-perf-profile", PERF_PROFILE_PATH, 
                     "--rank-mps", f"100,{slowdown_pct}"],
                    env={**os.environ, "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE": str(slowdown_pct)},
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
                )
                stdout_lut_rank0, stderr_lut_rank0 = p0_lut.communicate()
                _, stderr_lut_rank1 = p1_lut.communicate()

                if p0_lut.returncode != 0:
                    print(f"Error in lut split (rank 0): {stderr_lut_rank0}")
                if p1_lut.returncode != 0:
                    print(f"Error in lut split (rank 1): {stderr_lut_rank1}")
                
                lut_results = parse_single_benchmark_output(stdout_lut_rank0)
                lut_results.update({
                    "seq_len": seq_len,
                    "slowdown_pct": slowdown_pct,
                    "split_type": "lut",
                })
                all_sweep_results.append(lut_results)
                
                time.sleep(1) # Small pause between configurations
                
    except Exception as e:
        print(f"An error occurred during the sweep: {e}")

    print("\nSweep finished. Saving results to CSV...")
    df_results = pd.DataFrame(all_sweep_results)
    df_results.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"Results saved to {OUTPUT_CSV_PATH}")

if __name__ == "__main__":
    main()
