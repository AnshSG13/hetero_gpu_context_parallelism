# Benchmark Scripts

### `run_sweep.py`
-   Runs the full benchmark sweep over a matrix of sequence lengths and GPU slowdowns, testing all three rebalancing strategies.
-   **Command**: `python3 hpml_testing/run_sweep.py`

### `plot_sweep_results.py`
-  	Generates speedup plots from the `sweep_results.csv` file created by the main sweep.
-   **Command**: `python3 hpml_testing/plot_sweep_results.py`

### `run_matmul_mps_sweep.sh`
-   Generates the performance lookup table (`matmul_mps_sweep.csv`) required for the `lut` strategy. Run this once on your target hardware.
-   **Command**: `bash hpml_testing/run_matmul_mps_sweep.sh`

### `run_hetero_benchmark.sh`

-   **What it does**: Runs a single, quick comparison of the three rebalancing strategies for the one configuration set at the top of the script.

-   **Command**: `bash hpml_testing/run_hetero_benchmark.sh`



### `profile_rebalancing.py`

-   **What it does**: Runs a single benchmark configuration under the PyTorch Profiler, saving a detailed trace file for each GPU.

-   **Command**: This script is typically called by `run_profiling_comparison.sh` but can be run directly.



### `run_profiling_comparison.sh`

-   **What it does**: Orchestrates the profiling of the `even`, `uneven`, and `lut` strategies, generating trace files for each rank.

-   **Command**: `bash hpml_testing/run_profiling_comparison.sh`



### `analyze_profiles.py`

-   **What it does**: Parses the generated PyTorch profiler trace files and prints a summary table of `comm_wait` and `compute_block` durations for each strategy and rank.

-   **Command**: `python3 hpml_testing/analyze_profiles.py`



---



## Prerequisites



1.  **Python Libraries**:

    ```bash

    pip install pandas matplotlib seaborn tabulate

    ```

2.  **CUDA MPS Daemon**: The benchmark requires the NVIDIA MPS daemon to be active to simulate a heterogeneous setup.

    ```bash

    # Start the daemon before running any benchmarks

    sudo nvidia-cuda-mps-control -d

    ```


