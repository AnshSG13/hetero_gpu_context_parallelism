"""
Plot matmul MPS sweep results (log2 x-axis, linear y-axis).

Expects a CSV with columns:
  mps_pct,size,dtype,bg_streams,avg_ms,tflops

Usage:
  python hpml_testing/plot_matmul_mps.py hpml_testing/results/matmul_mps_sweep_*.csv
"""
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python plot_matmul_mps.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    df = pd.read_csv(csv_path)
    csv_path = Path(csv_path)

    # Latency vs size (log2 x-axis)
    plt.figure(figsize=(10, 6))
    for mps_pct, group in df.groupby("mps_pct"):
        group = group.sort_values("size")
        plt.plot(
            group["size"],
            group["avg_ms"],
            marker="o",
            label=f"MPS {mps_pct}%",
        )

    plt.xlabel("Matrix size (N x N)")
    plt.ylabel("Average latency (ms)")
    plt.title("Matmul latency vs size (log2 x-axis)")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.xscale("log", base=2)
    plt.legend()
    plt.tight_layout()
    latency_png = csv_path.with_suffix(".latency.png")
    plt.savefig(latency_png, dpi=200)
    plt.show()

    # Slowdown bar plot vs 100% baseline
    if 100 not in df["mps_pct"].unique():
        print("Baseline mps_pct=100 not found; skipping slowdown plot.")
        return

    baseline = df[df["mps_pct"] == 100][["size", "avg_ms"]].rename(columns={"avg_ms": "base_ms"})
    merged = df[df["mps_pct"] != 100].merge(baseline, on="size", how="inner")
    merged["slowdown"] = merged["avg_ms"] / merged["base_ms"]

    slowdown_by_pct = (
        merged.groupby("mps_pct")["slowdown"]
        .mean()
        .reset_index()
        .sort_values("mps_pct", ascending=False)
    )

    plt.figure(figsize=(8, 5))
    plt.bar(slowdown_by_pct["mps_pct"].astype(str), slowdown_by_pct["slowdown"])
    plt.xlabel("MPS throttle (%)")
    plt.ylabel("Slowdown vs 100% (avg across sizes)")
    plt.title("Matmul slowdown by throttle (baseline = 100%)")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    slowdown_png = csv_path.with_suffix(".slowdown.png")
    plt.savefig(slowdown_png, dpi=200)
    plt.show()

    print(f"Saved plots to {latency_png} and {slowdown_png}")


if __name__ == "__main__":
    main()
