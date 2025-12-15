import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import wandb

# --- Configuration ---
DEFAULT_INPUT_CSV = "hpml_testing/results/sweep_results.csv"

def generate_and_log_plots(df: pd.DataFrame):
    """
    Takes a DataFrame of sweep results, generates a slowdown comparison plot,
    and logs it directly to Weights & Biases.
    """
    # --- Data Transformation for Slowdown Calculation ---
    # Pivot the table to get latency for each split_type in columns
    pivot_df = df.pivot_table(
        index=['seq_len', 'slowdown_pct'],
        columns='split_type',
        values='overall_latency_ms'
    ).reset_index()

    # Calculate slowdown relative to the 'reference_homogeneous' baseline
    # Slowdown = Latency_measured / Latency_ideal
    pivot_df['even_slowdown'] = pivot_df['even'] / pivot_df['reference_homogeneous']
    pivot_df['uneven_slowdown'] = pivot_df['uneven'] / pivot_df['reference_homogeneous']
    pivot_df['lut_slowdown'] = pivot_df['lut'] / pivot_df['reference_homogeneous']
    pivot_df['formula_slowdown'] = pivot_df['formula'] / pivot_df['reference_homogeneous']

    # Melt the DataFrame back into a long format for plotting
    slowdown_df = pivot_df.melt(
        id_vars=['seq_len', 'slowdown_pct'],
        value_vars=['even_slowdown', 'uneven_slowdown', 'lut_slowdown', 'formula_slowdown'],
        var_name='split_type',
        value_name='slowdown'
    )
    # Clean up the 'split_type' names
    slowdown_df['split_type'] = slowdown_df['split_type'].str.replace('_slowdown', '')


    # --- Grouped Bar Charts for Slowdown ---
    print("Generating and logging slowdown comparison bar charts...")
    g_bar = sns.catplot(
        data=slowdown_df,
        x="slowdown_pct",
        y="slowdown",
        hue="split_type",
        col="seq_len",
        kind="bar",
        height=6,
        aspect=1.3,
        sharey=False,
        legend_out=True
    )
    g_bar.fig.suptitle("Slowdown of Rebalancing Strategies vs. Ideal Homogeneous Baseline", y=1.03)
    g_bar.set_axis_labels("Slowdown Percentage for Rank 1", "Slowdown Factor (Lower is Better)")
    g_bar.set_titles("Sequence Length: {col_name}")
    g_bar.despine(left=True)

    # Use a log scale for the y-axis and add a baseline at y=1.0
    for ax in g_bar.axes.flat:
        ax.axhline(y=1.0, color='r', linestyle='--', linewidth=2, label='Ideal Performance')
        ax.set_yscale('log')
        ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Handle legend
    g_bar.add_legend(title="Strategy")
    
    # --- WANDB Integration: Log the plot ---
    wandb.log({"Slowdown Comparison Plot": g_bar})
    plt.close(g_bar.fig) # Close the figure to free up memory
    print("Plot logged to wandb.")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        input_csv = sys.argv[1]
    else:
        input_csv = DEFAULT_INPUT_CSV

    if not os.path.exists(input_csv):
        print(f"Error: Input CSV file not found at {input_csv}")
        sys.exit(1)
        
    print("Running plot script as a standalone test. Initializing a dummy wandb run.")
    # This allows running the script standalone for testing/debugging plots
    # Note: This will create a new, separate run in wandb.
    wandb.init(project="heterogeneous-ring-attention", name="standalone_plot_test")
    
    df = pd.read_csv(input_csv)
    generate_and_log_plots(df)
    
    wandb.finish()
    print("Standalone test finished.")
