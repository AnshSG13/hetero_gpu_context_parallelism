import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go

# ============================================================
# CONFIG
# ============================================================
CSV_PATH = "./results/matmul_mps_sweep.csv"
POLY_DEGREE = 2   # recommended; higher degrees overfit badly

# ============================================================
# LOAD DATA
# ============================================================
df = pd.read_csv(CSV_PATH)

required_cols = {
    "mps_pct", "size", "dtype",
    "bg_streams", "avg_ms", "tflops"
}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError("Missing columns: {}".format(missing))

df = df.rename(columns={"size": "seq_len"})

# ============================================================
# NORMALIZE PERFORMANCE
# ============================================================
df["normalized_perf_pct"] = 100.0 * df["tflops"] / df["tflops"].max()

# Use log10(seq_len) because seq_len is plotted on log scale
df["log_seq_len"] = np.log10(df["seq_len"])

# ============================================================
# POLYNOMIAL REGRESSION (2D SURFACE)
# ============================================================
X = df[["log_seq_len", "mps_pct"]].to_numpy()
y = df["normalized_perf_pct"].to_numpy()

poly = PolynomialFeatures(degree=POLY_DEGREE, include_bias=True)
X_poly = poly.fit_transform(X)

reg = LinearRegression()
reg.fit(X_poly, y)

r2 = reg.score(X_poly, y)

# ============================================================
# PRINT POLYNOMIAL EQUATION
# ============================================================
feature_names = poly.get_feature_names_out(
    ["log10(seq_len)", "mps_pct"]
)

terms = []
for name, coef in zip(feature_names, reg.coef_):
    if name == "1":
        continue
    terms.append(f"({coef:+.6g})·{name}")

equation = (
    "NormalizedPerf(%) = "
    f"{reg.intercept_:.6g} "
    + " ".join(terms)
)

print("\n================ POLYNOMIAL REGRESSION =================")
print("Degree:", POLY_DEGREE)
print("R^2:", r2)
print("\nEquation:")
print(equation)
print("\nWhere: log10(seq_len) = log10(size)")
print("========================================================\n")

# ============================================================
# CREATE GRID FOR 3D SURFACE
# ============================================================
log_seq_grid = np.linspace(
    df["log_seq_len"].min(),
    df["log_seq_len"].max(),
    80
)
mps_grid = np.linspace(
    df["mps_pct"].min(),
    df["mps_pct"].max(),
    50
)

LOG_SEQ, MPS = np.meshgrid(log_seq_grid, mps_grid)

X_pred = np.column_stack([
    LOG_SEQ.ravel(),
    MPS.ravel()
])

Z = reg.predict(poly.transform(X_pred)).reshape(LOG_SEQ.shape)

SEQ = 10 ** LOG_SEQ  # back to linear scale for display

# ============================================================
# INTERACTIVE 3D PLOT (PLOTLY)
# ============================================================
fig = go.Figure()

fig.add_trace(go.Surface(
    x=SEQ,
    y=MPS,
    z=Z,
    colorscale="Viridis",
    opacity=0.85,
    name="Polynomial Surface"
))

fig.add_trace(go.Scatter3d(
    x=df["seq_len"],
    y=df["mps_pct"],
    z=df["normalized_perf_pct"],
    mode="markers",
    marker=dict(size=4, color="black"),
    name="Measured Data"
))

fig.update_layout(
    title=f"Polynomial Regression Surface (degree={POLY_DEGREE}, R²={r2:.3f})",
    scene=dict(
        xaxis=dict(
            title="Sequence Length",
            type="log"
        ),
        yaxis=dict(
            title="MPS (%)"
        ),
        zaxis=dict(
            title="Normalized Performance (%)"
        )
    ),
    width=900,
    height=650
)

fig.show()
