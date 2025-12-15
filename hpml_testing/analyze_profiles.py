import os
import json
import argparse
from tabulate import tabulate # For pretty printing tables

# Add tabulate to the prerequisites for the user
# pip install tabulate

def parse_trace_file(file_path):
    """
    Parses a PyTorch profiler trace file and extracts aggregated durations
    for custom labels 'comm_wait' and 'compute_block'.
    Durations in trace are in microseconds (us). Convert to milliseconds (ms).
    """
    total_comm_wait_us = 0
    total_compute_block_us = 0

    if not os.path.exists(file_path):
        print(f"Warning: Trace file not found: {file_path}")
        return {'comm_wait': 0, 'compute_block': 0}

    with open(file_path, 'r') as f:
        trace_data = json.load(f)

    for event in trace_data.get('traceEvents', []):
        if event.get('ph') == 'X' and 'dur' in event: # 'X' denotes a complete event
            if event.get('name') == 'comm_wait':
                total_comm_wait_us += event['dur']
            elif event.get('name') == 'compute_block':
                total_compute_block_us += event['dur']

    return {
        'comm_wait': total_comm_wait_us / 1000.0,    # convert to ms
        'compute_block': total_compute_block_us / 1000.0 # convert to ms
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze PyTorch Profiler traces for Ring Attention benchmark.")
    parser.add_argument("--results-dir", type=str, default="hpml_testing/results",
                        help="Directory containing the profiler trace JSON files.")
    parser.add_argument("--strategies", type=str, nargs='+',
                        default=["even", "uneven", "lut"],
                        help="List of strategies to analyze (e.g., even uneven lut).")
    parser.add_argument("--ranks", type=int, nargs='+',
                        default=[0, 1],
                        help="List of ranks to analyze (e.g., 0 1).")
    
    args = parser.parse_args()

    analysis_results = []

    print(f"Analyzing profiler traces from: {args.results_dir}")

    for strategy in args.strategies:
        for rank in args.ranks:
            trace_filename = f"profile_{strategy}_rank{rank}.json"
            file_path = os.path.join(args.results_dir, trace_filename)
            
            print(f"Processing {trace_filename}...")
            parsed_data = parse_trace_file(file_path)
            
            analysis_results.append({
                'Strategy': strategy,
                'Rank': rank,
                'Total comm_wait (ms)': parsed_data['comm_wait'],
                'Total compute_block (ms)': parsed_data['compute_block']
            })

    if not analysis_results:
        print("No analysis results found. Check your trace files and paths.")
        return

    print("\n--- Profiling Analysis Summary ---")
    print(tabulate(analysis_results, headers="keys", tablefmt="grid"))
    print("\nNotes:")
    print("- 'comm_wait' includes the time spent waiting for communication, plus data transfers.")
    print("- 'compute_block' includes the time spent on attention computation for the local block.")
    print("- Durations are aggregated across all steps of the profiling run.")


if __name__ == "__main__":
    main()
