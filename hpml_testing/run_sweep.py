import os
import subprocess
import re
import pandas as pd
import math
import time
import argparse
import wandb
from plot_sweep_results import generate_and_log_plots

# --- Configuration ---
SEQLEN_SWEEP = [4096]#, 8192, 16384, 32768, 65536]
SLOWDOWN_SWEEP = [90, 80]#, 70, 60, 50, 40, 30, 20, 10] # MPS percentage for the slower GPU 
DEFAULT_PROFILE_PATH = "hpml_testing/results/ring_attention_profile.csv"
OUTPUT_CSV_PATH = "hpml_testing/results/sweep_results.csv"
BENCHMARK_SCRIPT = "hpml_testing/benchmark_hetero_latency.py"

def parse_single_benchmark_output(output_string):
    """
    Parses the output of a single benchmark_hetero_latency.py execution (rank 0's output).
    Extracts overall latency, individual rank latencies, and token counts.
    """
    results = {}

    split_type_match = re.search(r"Running benchmark with '(\w+)' split.", output_string)
    if split_type_match:
        results['split_type'] = split_type_match.group(1)
    else:
        results['split_type'] = "unknown" 

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

def main():
    parser = argparse.ArgumentParser(description="Run a sweep of benchmarks for heterogeneous ring attention.")
    parser.add_argument(
        "--profile-path",
        type=str,
        default=DEFAULT_PROFILE_PATH,
        help="Path to the performance profile (LUT) to use for the 'lut' strategy."
    )
    args = parser.parse_args()

    # --- WANDB Integration: Initialize Run ---
    wandb.init(
        project="heterogeneous-ring-attention",
        config={
            "seq_len_sweep": SEQLEN_SWEEP,
            "slowdown_sweep": SLOWDOWN_SWEEP,
            "profile_path": args.profile_path
        }
    )

    print("Starting benchmark sweep...")
    print(f"Using performance profile for LUT strategy: {args.profile_path}")
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

    all_sweep_results = []

    os.environ['PYTHONPATH'] = os.getcwd() + ':' + os.environ.get('PYTHONPATH', '')

    try:
        for seq_len in SEQLEN_SWEEP:
            for slowdown_pct in SLOWDOWN_SWEEP:
                print(f"\n--- Running: SEQ_LEN={seq_len}, SLOWDOWN_PCT={slowdown_pct} ---")
                current_slowdown_factor = slowdown_pct / 100.0

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
                _, stderr_even_rank1 = p1_even.communicate() 
                
                if p0_even.returncode != 0:
                    print(f"Error in even split (rank 0): {stderr_even_rank0}")
                if p1_even.returncode != 0:
                    print(f"Error in even split (rank 1): {stderr_even_rank1}")

                even_results = parse_single_benchmark_output(stdout_even_rank0)
                even_results.update({
                    "seq_len": seq_len,
                    "slowdown_pct": slowdown_pct,
                    "split_type": "even", 
                })
                all_sweep_results.append(even_results)


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


                print("Running LUT Split...")
                p0_lut = subprocess.Popen(
                    ["python3", BENCHMARK_SCRIPT, 
                     "--rank", "0", "--world-size", "2", 
                     "--seq-len", str(seq_len), "--emb-dim", "4096", "--n-heads", "32", 
                     "--split-type", "lut", 
                     "--use-perf-profile", args.profile_path, 
                     "--rank-mps", f"100,{slowdown_pct}"],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
                )
                p1_lut = subprocess.Popen(
                    ["python3", BENCHMARK_SCRIPT, 
                     "--rank", "1", "--world-size", "2", 
                     "--seq-len", str(seq_len), "--emb-dim", "4096", "--n-heads", "32", 
                     "--split-type", "lut", 
                     "--use-perf-profile", args.profile_path, 
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


                # --- Run Formula Split ---
                print("Running Formula Split...")
                p0_formula = subprocess.Popen(
                    ["python3", BENCHMARK_SCRIPT,
                     "--rank", "0", "--world-size", "2",
                     "--seq-len", str(seq_len), "--emb-dim", "4096", "--n-heads", "32",
                     "--split-type", "formula",
                     "--rank-mps", f"100,{slowdown_pct}"],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
                )
                p1_formula = subprocess.Popen(
                    ["python3", BENCHMARK_SCRIPT,
                     "--rank", "1", "--world-size", "2",
                     "--seq-len", str(seq_len), "--emb-dim", "4096", "--n-heads", "32",
                     "--split-type", "formula",
                     "--rank-mps", f"100,{slowdown_pct}"],
                    env={**os.environ, "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE": str(slowdown_pct)},
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
                )
                stdout_formula_rank0, stderr_formula_rank0 = p0_formula.communicate()
                _, stderr_formula_rank1 = p1_formula.communicate()

                if p0_formula.returncode != 0:
                    print(f"Error in formula split (rank 0): {stderr_formula_rank0}")
                if p1_formula.returncode != 0:
                    print(f"Error in formula split (rank 1): {stderr_formula_rank1}")

                formula_results = parse_single_benchmark_output(stdout_formula_rank0)
                formula_results.update({
                    "seq_len": seq_len,
                    "slowdown_pct": slowdown_pct,
                    "split_type": "formula",
                })
                all_sweep_results.append(formula_results)
                

                # --- Run Homogeneous Reference Split ---
                print("Running Homogeneous Reference Split...")
                p0_ref = subprocess.Popen(
                    ["python3", BENCHMARK_SCRIPT, 
                     "--rank", "0", "--world-size", "2", 
                     "--seq-len", str(seq_len), "--emb-dim", "4096", "--n-heads", "32", 
                     "--split-type", "even"],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
                )
                p1_ref = subprocess.Popen(
                    ["python3", BENCHMARK_SCRIPT, 
                     "--rank", "1", "--world-size", "2", 
                     "--seq-len", str(seq_len), "--emb-dim", "4096", "--n-heads", "32", 
                     "--split-type", "even"],
                    # NOTE: No MPS slowdown applied to either rank for the reference case
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
                )
                stdout_ref_rank0, stderr_ref_rank0 = p0_ref.communicate()
                _, stderr_ref_rank1 = p1_ref.communicate()

                if p0_ref.returncode != 0:
                    print(f"Error in reference split (rank 0): {stderr_ref_rank0}")
                if p1_ref.returncode != 0:
                    print(f"Error in reference split (rank 1): {stderr_ref_rank1}")

                ref_results = parse_single_benchmark_output(stdout_ref_rank0)
                ref_results.update({
                    "seq_len": seq_len,
                    "slowdown_pct": slowdown_pct, # Keep for grouping, though it's not applied
                    "split_type": "reference_homogeneous",
                })
                all_sweep_results.append(ref_results)

                time.sleep(1) # Small pause between configurations
                
    except Exception as e:
        print(f"An error occurred during the sweep: {e}")
    finally:
        # --- WANDB Integration: Generate and Log Plot, then Finish Run ---
        print("\nSweep finished. Generating plot, saving results, and closing wandb run...")
        if all_sweep_results: # Only plot if there is data
            df_results = pd.DataFrame(all_sweep_results)
            df_results.to_csv(OUTPUT_CSV_PATH, index=False)
            print(f"Results saved to {OUTPUT_CSV_PATH}")
            wandb.save(OUTPUT_CSV_PATH) # Save the CSV as a wandb artifact
            generate_and_log_plots(df_results) # New call to generate and log plot
        else:
            print("No results were generated, skipping CSV save and plotting.")

        wandb.finish()

if __name__ == "__main__":
    main()
