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
    Reads sweep results and generates a slowdown comparison plot, saving it locally.
    """
    if not os.path.exists(csv_path):
        print(f"Error: Input CSV file not found at {csv_path}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = pd.read_csv(csv_path)

    # --- Data Transformation for Slowdown Calculation ---
    pivot_df = df.pivot_table(
        index=['seq_len', 'slowdown_pct'],
        columns='split_type',
        values='overall_latency_ms'
    ).reset_index()

    pivot_df['even_slowdown'] = pivot_df['even'] / pivot_df['reference_homogeneous']
    pivot_df['uneven_slowdown'] = pivot_df['uneven'] / pivot_df['reference_homogeneous']
    pivot_df['lut_slowdown'] = pivot_df['lut'] / pivot_df['reference_homogeneous']
    pivot_df['formula_slowdown'] = pivot_df['formula'] / pivot_df['reference_homogeneous']

    slowdown_df = pivot_df.melt(
        id_vars=['seq_len', 'slowdown_pct'],
        value_vars=['even_slowdown', 'uneven_slowdown', 'lut_slowdown', 'formula_slowdown'],
        var_name='split_type',
        value_name='slowdown'
    )
    # Clean up the 'split_type' names
    slowdown_df['split_type'] = slowdown_df['split_type'].str.replace('_slowdown', '')

    # Sort the dataframe to ensure the plot facets are in ascending order of seq_len
    slowdown_df = slowdown_df.sort_values("seq_len")


    # --- Grouped Bar Charts for Slowdown ---
    print("Generating slowdown comparison bar charts...")
    g_bar = sns.catplot(
        data=slowdown_df,
        x="slowdown_pct",
        y="slowdown",
        hue="split_type",
        row="seq_len",  # Changed from col to row for vertical stacking
        kind="bar",
        height=2.5,     # Reduced height for a tighter layout
        aspect=4,       # Increased aspect ratio (width/height)
        sharex=True,   # All subplots will share the same x-axis
        legend_out=True
    )
    g_bar.fig.suptitle("Slowdown of Rebalancing Strategies vs. Ideal Homogeneous Baseline", y=1.03)
    g_bar.set_axis_labels("Slowdown Percentage for Rank 1", "Slowdown Factor (Lower is Better)")
    g_bar.set_titles("Sequence Length: {row_name}")
    g_bar.despine(left=True)

    # Use a log scale for the y-axis and add a baseline at y=1.0
    for ax in g_bar.axes.flat:
        ax.axhline(y=1.0, color='r', linestyle='--', linewidth=2, label='Ideal Performance')
        ax.set_yscale('log')
        ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Handle legend
    g_bar.add_legend(title="Strategy")
    
    # Save the plot to a local file
    bar_chart_path = os.path.join(OUTPUT_DIR, "sweep_slowdown_comparison.png")
    g_bar.savefig(bar_chart_path)
    plt.close(g_bar.fig)
    print(f"Plot saved locally to {bar_chart_path}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        input_csv = sys.argv[1]
    else:
        input_csv = DEFAULT_INPUT_CSV
    
    create_plots(input_csv)
