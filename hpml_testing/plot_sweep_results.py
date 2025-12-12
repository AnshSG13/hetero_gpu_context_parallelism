import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# --- Configuration ---
DEFAULT_INPUT_CSV = "hpml_testing/results/sweep_results.csv"
OUTPUT_DIR = "hpml_testing/plots"

def create_plots(csv_path):
    """
    Reads the sweep results from a CSV file and generates plots
    to show the speedup of different split strategies relative to the 'even' baseline.
    """
    if not os.path.exists(csv_path):
        print(f"Error: Input CSV file not found at {csv_path}")
        return

    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(csv_path)

    # --- Data Transformation for Speedup Calculation ---
    # Pivot the table to get latency for each split_type in columns
    pivot_df = df.pivot_table(
        index=['seq_len', 'slowdown_pct'],
        columns='split_type',
        values='overall_latency_ms'
    ).reset_index()

    # Calculate speedup relative to the 'even' baseline
    pivot_df['uneven_speedup'] = pivot_df['even'] / pivot_df['uneven']
    pivot_df['lut_speedup'] = pivot_df['even'] / pivot_df['lut']

    # Melt the DataFrame back into a long format for plotting
    speedup_df = pivot_df.melt(
        id_vars=['seq_len', 'slowdown_pct'],
        value_vars=['uneven_speedup', 'lut_speedup'],
        var_name='split_type',
        value_name='speedup'
    )
    # Clean up the 'split_type' names
    speedup_df['split_type'] = speedup_df['split_type'].str.replace('_speedup', '')


    # --- Grouped Bar Charts for Speedup ---
    print("Generating speedup bar charts...")
    g_bar = sns.catplot(
        data=speedup_df,
        x="slowdown_pct",
        y="speedup",
        hue="split_type",
        col="seq_len",
        kind="bar",
        height=5,
        aspect=1.2,
        sharey=True,  # Y-axis (speedup) should be shared for better comparison
        legend_out=True
    )
    g_bar.fig.suptitle("Speedup vs Even Split Baseline", y=1.03)
    g_bar.set_axis_labels("Slowdown Percentage for Rank 1", "Speedup (Higher is Better)")
    g_bar.set_titles("Sequence Length: {col_name}")
    g_bar.despine(left=True)

    # Add a horizontal line at y=1.0 to represent the baseline
    for ax in g_bar.axes.flat:
        ax.axhline(y=1.0, color='r', linestyle='--', linewidth=2, label='Baseline (Even Split)')
        ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Handle legend
    plt.legend()
    
    bar_chart_path = os.path.join(OUTPUT_DIR, "sweep_speedup_bar_charts.png")
    g_bar.savefig(bar_chart_path)
    print(f"Speedup bar charts saved to {bar_chart_path}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        input_csv = sys.argv[1]
    else:
        input_csv = DEFAULT_INPUT_CSV
    
    create_plots(input_csv)
