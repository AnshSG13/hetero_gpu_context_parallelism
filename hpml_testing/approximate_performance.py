import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator

# -----------------------------
# CONFIG
# -----------------------------
CSV_PATH = "./results/matmul_mps_sweep.csv"

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv(CSV_PATH)

required_cols = {
    "mps_pct", "size", "dtype",
    "bg_streams", "avg_ms", "tflops"
}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError("Missing columns: {}".format(missing))

# Rename for clarity
df = df.rename(columns={"size": "seq_len"})

# -----------------------------
# NORMALIZE TFLOPs (per MPS)
# -----------------------------
df["normalized_perf_pct"] = (
    df.groupby("mps_pct")["tflops"]
      .transform(lambda x: 100.0 * x / x.max())
)

# -----------------------------
# ONE FIGURE PER MPS %
# -----------------------------
for mps_pct, group in df.groupby("mps_pct"):
    group = group.sort_values("seq_len")

    x = group["seq_len"].values
    y = group["normalized_perf_pct"].values

    # PCHIP interpolation (monotonic, shape-preserving)
    pchip = PchipInterpolator(x, y)

    # Smooth evaluation grid (log-spaced for seq_len)
    x_fit = np.logspace(np.log10(x.min()), np.log10(x.max()), 500)
    y_fit = pchip(x_fit)

    # -----------------------------
    # PLOT
    # -----------------------------
    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, s=45, alpha=0.7, label="Measured")
    plt.plot(x_fit, y_fit, linewidth=2, label="PCHIP best fit")

    plt.xlabel("Sequence Length")
    plt.ylabel("Normalized Performance (%)")
    plt.title(f"MPS {mps_pct}% — Normalized Performance Curve")
    plt.grid(True)
    plt.xscale("log")
    plt.legend()

    plt.tight_layout()
    plt.show()
