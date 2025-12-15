import json
import argparse
import os
from collections import defaultdict
from tabulate import tabulate

def analyze_top_kernels(file_path, top_n=10):
    """
    Parses a PyTorch profiler trace file and finds the top N most time-consuming
    CUDA kernels.
    """
    if not os.path.exists(file_path):
        print(f"Error: Trace file not found: {file_path}")
        return

    print(f"Analyzing top {top_n} kernels from {file_path}...")

    kernel_durations = defaultdict(int)

    with open(file_path, 'r') as f:
        trace_data = json.load(f)

    # Filter for CUDA kernel events and aggregate their durations
    for event in trace_data.get('traceEvents', []):
        # 'cat' == 'kernel' identifies CUDA kernel execution events
        if event.get('cat') == 'kernel' and event.get('ph') == 'X' and 'dur' in event:
            kernel_name = event['name']
            kernel_durations[kernel_name] += event['dur']

    if not kernel_durations:
        print("No CUDA kernel events found in the trace file.")
        return

    # Sort kernels by total duration in descending order
    sorted_kernels = sorted(kernel_durations.items(), key=lambda item: item[1], reverse=True)

    # Prepare data for tabulation (convert us to ms)
    table_data = [
        {
            "Rank": i + 1,
            "Kernel Name": name,
            "Total Time (ms)": duration / 1000.0,
            "Percentage": (duration / sum(kernel_durations.values())) * 100
        }
        for i, (name, duration) in enumerate(sorted_kernels[:top_n])
    ]

    print(f"\n--- Top {top_n} CUDA Kernels by Total Time ---")
    print(tabulate(table_data, headers="keys", tablefmt="grid", floatfmt=".4f"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze CUDA kernels from a PyTorch Profiler trace.")
    parser.add_argument(
        "--trace-file",
        type=str,
        default="hpml_testing/results/profile_lut_rank1.json",
        help="Path to the profiler trace JSON file to analyze."
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of top kernels to display."
    )
    
    args = parser.parse_args()
    analyze_top_kernels(args.trace_file, args.top_n)
