import os
import subprocess
import re
import pandas as pd
import time

# --- Configuration ---
SEQ_LEN = 8192 # Use a representative sequence length for the profile
MPS_SWEEP = [100, 90, 80, 70, 60, 50, 40, 30, 20, 10]
OUTPUT_CSV_PATH = "hpml_testing/results/ring_attention_profile.csv"
BENCHMARK_SCRIPT = "hpml_testing/benchmark_hetero_latency.py"

def parse_latency_from_output(output_string):
    """
    Parses the 'Overall Latency' from the benchmark output string.
    """
    match = re.search(r"Overall Latency \(max of ranks\): (\d+\.\d+) ms", output_string)
    if match:
        return float(match.group(1))
    return None

def main():
    """
    Runs the ring attention benchmark under various MPS settings to generate
    a latency-based performance profile (LUT).
    """
    print("Generating new latency-based performance profile (LUT)...")
    print(f"Using fixed sequence length: {SEQ_LEN}")
    
    # Set CUDA_VISIBLE_DEVICES to ensure consistency
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

    profile_data = []

    # Ensure python path includes current directory for fms imports
    os.environ['PYTHONPATH'] = os.getcwd() + ':' + os.environ.get('PYTHONPATH', '')
    
    # Assumes MPS daemon is already running
    print("Assuming CUDA MPS daemon is running.")

    for mps_pct in MPS_SWEEP:
        print(f"\n--- Profiling with {mps_pct}% capacity on Rank 1 ---")
        
        env_rank1 = os.environ.copy()
        if mps_pct < 100:
            env_rank1["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(mps_pct)

        p0 = subprocess.Popen(
            ["python3", BENCHMARK_SCRIPT, 
             "--rank", "0", "--world-size", "2", 
             "--seq-len", str(SEQ_LEN), "--emb-dim", "4096", "--n-heads", "32", 
             "--split-type", "even"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        p1 = subprocess.Popen(
            ["python3", BENCHMARK_SCRIPT, 
             "--rank", "1", "--world-size", "2", 
             "--seq-len", str(SEQ_LEN), "--emb-dim", "4096", "--n-heads", "32", 
             "--split-type", "even"],
            env=env_rank1,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        stdout_rank0, stderr_rank0 = p0.communicate()
        _, stderr_rank1 = p1.communicate()
        
        if p0.returncode != 0:
            print(f"Error in benchmark run (rank 0): {stderr_rank0}")
            continue
        if p1.returncode != 0:
            print(f"Error in benchmark run (rank 1): {stderr_rank1}")
            continue

        latency = parse_latency_from_output(stdout_rank0)
        
        if latency is not None:
            print(f"  -> Measured Latency: {latency:.2f} ms")
            profile_data.append({"mps_pct": mps_pct, "latency_ms": latency})
        else:
            print("  -> Failed to parse latency from output.")
        
        time.sleep(1)

    if not profile_data:
        print("\nNo profiling data was generated.")
        return
        
    print(f"\nProfile generation complete. Saving results to {OUTPUT_CSV_PATH}")
    df_results = pd.DataFrame(profile_data)
    df_results.to_csv(OUTPUT_CSV_PATH, index=False)
    print("New LUT saved.")

if __name__ == "__main__":
    main()
