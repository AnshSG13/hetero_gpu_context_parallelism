import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python plot_matmul_mps.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    df = pd.read_csv(csv_path)
    csv_path = Path(csv_path)

    # ---------------------------------------------------------
    # Pre-processing: Calculate Performance Metrics
    # ---------------------------------------------------------
    # 1. Identify Baseline (MPS=100) for each matrix size
    baseline = df[df["mps_pct"] == 100][["size", "avg_ms"]].rename(columns={"avg_ms": "base_ms"})
    
    # 2. Merge baseline back into main dataframe
    merged = df.merge(baseline, on="size", how="inner")
    
    # 3. Calculate Normalized Performance
    #    Example: If Base=10ms, Actual=100ms -> Norm Perf = 0.1 (10% of speed)
    merged["norm_perf"] = merged["base_ms"] / merged["avg_ms"]
    
    # 4. Calculate Slowdown Factor (Reciprocal of Norm Perf)
    #    Example: If Base=10ms, Actual=100ms -> Slowdown = 10x
    merged["slowdown"] = merged["avg_ms"] / merged["base_ms"]

    # Define representative sizes to keep plots readable (avoiding too many lines)
    # Adjust this list if you want to see all sizes
    rep_sizes = [1024, 2048, 4096, 8192, 16384, 32768]

    # ---------------------------------------------------------
    # Graph 1: Latency vs Size (Log-Log)
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))
    mps_values = sorted(df["mps_pct"].unique(), reverse=True)
    
    for mps in mps_values:
        group = df[df["mps_pct"] == mps].sort_values("size")
        plt.plot(
            group["size"], 
            group["avg_ms"], 
            marker="o", 
            label=f"MPS {mps}%"
        )

    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.xlabel("Matrix Size (N)")
    plt.ylabel("Latency (ms)")
    plt.title("Matmul Latency vs Size (Log-Log)")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()
    
    out_file = csv_path.with_suffix(".graph1_latency.png")
    plt.savefig(out_file, dpi=200)
    print(f"Saved {out_file}")

    # ---------------------------------------------------------
    # Graph 2: Normalized Performance vs MPS Percentage
    # (Visualizing the Mapping vs Ideal)
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))
    
    for size in rep_sizes:
        if size in merged["size"].unique():
            group = merged[merged["size"] == size].sort_values("mps_pct")
            plt.plot(
                group["mps_pct"], 
                group["norm_perf"], 
                marker="o", 
                label=f"N={size}"
            )
            
    # Add Ideal Line: y = x / 100 (Linear Scaling)
    # If MPS=10%, Ideal Perf=0.1
    x_ideal = np.array(sorted(merged["mps_pct"].unique()))
    y_ideal = x_ideal / 100.0
    plt.plot(x_ideal, y_ideal, "k--", linewidth=2, label="Ideal Linear Scaling")

    plt.xscale("log")
    plt.yscale("log")
    
    plt.xlabel("MPS Percentage (Log Scale)")
    plt.ylabel("Normalized Performance")
    plt.title("Performance Scaling vs MPS Percentage scaling")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    
    # Custom ticks for readability on log axes
    ticks = [1, 2, 5, 10, 20, 50, 100]
    tick_labels = [f"{t}%" for t in ticks]
    plt.xticks(ticks, tick_labels)
    plt.yticks([t/100.0 for t in ticks], tick_labels)
    
    plt.tight_layout()
    
    out_file = csv_path.with_suffix(".graph2_scaling.png")
    plt.savefig(out_file, dpi=200)
    print(f"Saved {out_file}")

    # ---------------------------------------------------------
    # Graph 3: MPS Efficiency (Actual / Theoretical)
    # (Quantifying the deviation from ideal)
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))
    
    for size in rep_sizes:
        if size in merged["size"].unique():
            group = merged[merged["size"] == size].sort_values("mps_pct")
            
            # Theoretical Performance = MPS% / 100
            # Efficiency = Actual_Norm_Perf / Theoretical_Perf
            # If Result < 1.0, we are underperforming relative to the resource share
            ratio = group["norm_perf"] / (group["mps_pct"] / 100.0)
            
            plt.plot(group["mps_pct"], ratio, marker="o", label=f"N={size}")

    plt.axhline(1.0, color="k", linestyle="--", label="Ideal Efficiency (1.0)")
    
    plt.xscale("log")
    plt.xlabel("MPS Percentage")
    plt.ylabel("Efficiency (Actual Speed / Theoretical Speed)")
    plt.title("MPS Efficiency vs Percentage")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.xticks(ticks, tick_labels)
    plt.ylim(bottom=0) # Anchor Y-axis at 0
    plt.tight_layout()
    
    out_file = csv_path.with_suffix(".graph3_efficiency.png")
    plt.savefig(out_file, dpi=200)
    print(f"Saved {out_file}")

    # ---------------------------------------------------------
    # Graph 4: Slowdown Bar Chart (Bonus)
    # ---------------------------------------------------------
    plt.figure()
    
    # Subset for cleaner bars
    sizes_for_bar = [4096, 16384, 32768] # Adjust as needed
    subset = merged[merged["size"].isin(sizes_for_bar)]
    
    pivot_df = subset.pivot(index="mps_pct", columns="size", values="slowdown")
    pivot_df = pivot_df.sort_index(ascending=False)
    
    pivot_df.plot(kind="bar", figsize=(12, 6), width=0.8)
    
    plt.xlabel("MPS Percentage")
    plt.ylabel("Slowdown Factor (vs MPS 100%)")
    plt.title("Slowdown Factor by MPS Setting")
    plt.legend(title="Matrix Size")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    
    out_file = csv_path.with_suffix(".graph4_bars.png")
    plt.savefig(out_file, dpi=200)
    print(f"Saved {out_file}")

if __name__ == "__main__":
    main()
