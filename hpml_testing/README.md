# Benchmark Scripts

This file provides a guide to the scripts used for running benchmarks and analyzing results.

---

### Primary Workflow

The primary workflow involves generating a performance profile, running a full benchmark sweep, and then plotting the results.

1.  **`generate_ring_profile.py`**
    -   **What it does**: Generates the recommended, latency-based performance profile (`ring_attention_profile.csv`) by running the actual ring attention workload.
    -   **Command**: `python3 hpml_testing/generate_ring_profile.py`

2.  **`run_sweep.py`**
    -   **What it does**: Runs the full benchmark sweep over a matrix of sequence lengths and GPU slowdowns. All results and the final summary plot are logged to Weights & Biases.
    -   **Command**: `python3 hpml_testing/run_sweep.py --profile-path hpml_testing/results/ring_attention_profile.csv`

3.  **`plot_sweep_results.py`**
    -   **What it does**: Generates plots from the `sweep_results.csv` file. Can be run standalone for testing, but is automatically called by `run_sweep.py` to log plots to `wandb`.
    -   **Command**: `python3 hpml_testing/plot_sweep_results.py`

---

### Standalone & Legacy Scripts

-   **`run_hetero_benchmark.sh`**
    -   **What it does**: Runs a single, quick comparison of the rebalancing strategies for the configuration set at the top of the script.

-   **`run_matmul_mps_sweep.sh`**
    -   **What it does**: Generates the legacy, `tflops`-based performance profile from a simple matrix multiplication benchmark.

---

## Prerequisites

1.  **Python Libraries**:
    ```bash
    pip install pandas matplotlib seaborn wandb
    ```

2.  **Weights & Biases Login**:
    You will need to log in to a Weights & Biases account to record your experiments. This is a one-time setup.
    ```bash
    wandb login
    ```

3.  **CUDA MPS Daemon**: Required to simulate a heterogeneous setup.
    ```bash
    # Start the daemon before running any benchmarks
    sudo nvidia-cuda-mps-control -d
    ```