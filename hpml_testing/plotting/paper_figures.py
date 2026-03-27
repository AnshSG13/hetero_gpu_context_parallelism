#!/usr/bin/env python3
"""
Plotting script for heterogeneous ring attention sweep results.
Generates figures for the paper from sweep_results.csv
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os
import sys

INPUT_CSV = "hpml_testing/results/sweep_results.csv"
OUTPUT_DIR = "latex/images"

COLORS = {
    'even': '#E74C3C',
    'uneven': '#27AE60',
    'lut': '#3498DB',
    'formula': '#9B59B6',
    'reference_homogeneous': '#95A5A6'
}

LABELS = {
    'even': 'Even Split',
    'uneven': 'Uneven Split',
    'lut': 'LUT Split',
    'formula': 'Formula Split',
    'reference_homogeneous': 'Homogeneous'
}

plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
})


def load_data(csv_path):
    df = pd.read_csv(csv_path)

    pivot = df.pivot_table(
        index=['seq_len', 'slowdown_pct'],
        columns='split_type',
        values='overall_latency_ms'
    ).reset_index()

    for strategy in ['even', 'uneven', 'lut', 'formula']:
        pivot[f'{strategy}_efficiency'] = (pivot['reference_homogeneous'] / pivot[strategy]) * 100
        pivot[f'{strategy}_speedup'] = pivot['even'] / pivot[strategy]

    return df, pivot


def plot_latency_vs_heterogeneity(df, pivot, output_dir):
    """Latency vs MPS% for each sequence length."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    seq_lens = sorted(df['seq_len'].unique())
    strategies = ['even', 'uneven', 'lut', 'formula']

    for idx, seq_len in enumerate(seq_lens):
        ax = axes[idx]
        subset = pivot[pivot['seq_len'] == seq_len].sort_values('slowdown_pct')

        for strategy in strategies:
            ax.plot(subset['slowdown_pct'], subset[strategy],
                   marker='o', linewidth=2, markersize=6,
                   color=COLORS[strategy], label=LABELS[strategy])

        homo_val = subset['reference_homogeneous'].iloc[0]
        ax.axhline(y=homo_val, color=COLORS['reference_homogeneous'],
                   linestyle='--', linewidth=2, label='Homogeneous')

        ax.set_xlabel('MPS %')
        ax.set_ylabel('Latency (ms)')
        ax.set_title(f'Seq Len = {seq_len:,}')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.invert_xaxis()

    if len(seq_lens) < 6:
        axes[-1].axis('off')

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=5, bbox_to_anchor=(0.5, 0.02))
    fig.suptitle('Latency vs GPU Heterogeneity', fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    plt.savefig(os.path.join(output_dir, 'latency_vs_heterogeneity.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'latency_vs_heterogeneity.pdf'), bbox_inches='tight')
    plt.close()


def plot_efficiency(df, pivot, output_dir):
    """Efficiency relative to homogeneous baseline."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    seq_lens = sorted(df['seq_len'].unique())
    strategies = ['even', 'uneven', 'lut', 'formula']

    for idx, seq_len in enumerate(seq_lens):
        ax = axes[idx]
        subset = pivot[pivot['seq_len'] == seq_len].sort_values('slowdown_pct')

        for strategy in strategies:
            ax.plot(subset['slowdown_pct'], subset[f'{strategy}_efficiency'],
                   marker='o', linewidth=2, markersize=6,
                   color=COLORS[strategy], label=LABELS[strategy])

        ax.axhline(y=100, color='gray', linestyle='--', linewidth=1, alpha=0.7)
        ax.set_xlabel('MPS %')
        ax.set_ylabel('Efficiency (%)')
        ax.set_title(f'Seq Len = {seq_len:,}')
        ax.set_ylim(0, 110)
        ax.grid(True, alpha=0.3)
        ax.invert_xaxis()

    if len(seq_lens) < 6:
        axes[-1].axis('off')

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 0.02))
    fig.suptitle('Efficiency vs Homogeneous Baseline', fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    plt.savefig(os.path.join(output_dir, 'efficiency_vs_heterogeneity.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'efficiency_vs_heterogeneity.pdf'), bbox_inches='tight')
    plt.close()


def plot_speedup_bars(df, pivot, output_dir):
    """Speedup over even split baseline."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    seq_lens = sorted(df['seq_len'].unique())
    strategies = ['uneven', 'lut', 'formula']

    for idx, seq_len in enumerate(seq_lens):
        ax = axes[idx]
        subset = pivot[pivot['seq_len'] == seq_len].sort_values('slowdown_pct')

        x = np.arange(len(subset))
        width = 0.25

        for i, strategy in enumerate(strategies):
            offset = (i - 1) * width
            ax.bar(x + offset, subset[f'{strategy}_speedup'], width,
                  label=LABELS[strategy], color=COLORS[strategy], alpha=0.85)

        ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1.5)
        ax.set_xlabel('MPS %')
        ax.set_ylabel('Speedup vs Even')
        ax.set_title(f'Seq Len = {seq_len:,}')
        ax.set_xticks(x)
        ax.set_xticklabels(subset['slowdown_pct'].astype(int))
        ax.grid(True, alpha=0.3, axis='y')

    if len(seq_lens) < 6:
        axes[-1].axis('off')

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 0.02))
    fig.suptitle('Speedup Over Even Split', fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    plt.savefig(os.path.join(output_dir, 'speedup_over_even.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'speedup_over_even.pdf'), bbox_inches='tight')
    plt.close()


def plot_token_distribution(df, output_dir):
    """Token distribution across GPUs."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    slowdowns = [90, 50, 10]
    seq_len = 16384
    strategies = ['even', 'uneven', 'lut', 'formula']

    for idx, slowdown in enumerate(slowdowns):
        ax = axes[idx]
        subset = df[(df['seq_len'] == seq_len) & (df['slowdown_pct'] == slowdown)]
        subset = subset[subset['split_type'].isin(strategies)]

        x = np.arange(len(strategies))

        rank0 = [subset[subset['split_type'] == s]['rank0_tokens'].values[0] for s in strategies]
        rank1 = [subset[subset['split_type'] == s]['rank1_tokens'].values[0] for s in strategies]

        ax.bar(x, rank0, label='GPU 0 (100%)', color='#3498DB', alpha=0.85)
        ax.bar(x, rank1, bottom=rank0, label=f'GPU 1 ({slowdown}%)', color='#E74C3C', alpha=0.85)

        ax.set_xlabel('Strategy')
        ax.set_ylabel('Tokens')
        ax.set_title(f'MPS = {slowdown}%')
        ax.set_xticks(x)
        ax.set_xticklabels(['Even', 'Uneven', 'LUT', 'Formula'], rotation=15)
        ax.axhline(y=seq_len/2, color='gray', linestyle='--', linewidth=1, alpha=0.7)
        ax.grid(True, alpha=0.3, axis='y')

        for i, (r0, r1) in enumerate(zip(rank0, rank1)):
            ax.annotate(f'{r0:,}', xy=(i, r0/2), ha='center', va='center', fontsize=8, color='white', fontweight='bold')
            ax.annotate(f'{r1:,}', xy=(i, r0 + r1/2), ha='center', va='center', fontsize=8, color='white', fontweight='bold')

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 0.02))
    fig.suptitle(f'Token Distribution (Seq Len = {seq_len:,})', fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0.08, 1, 0.93])

    plt.savefig(os.path.join(output_dir, 'token_distribution.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'token_distribution.pdf'), bbox_inches='tight')
    plt.close()


def plot_scaling(df, pivot, output_dir):
    """Latency scaling with sequence length."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    slowdowns = [90, 50, 10]
    strategies = ['even', 'uneven', 'lut', 'formula', 'reference_homogeneous']

    for idx, slowdown in enumerate(slowdowns):
        ax = axes[idx]
        subset = pivot[pivot['slowdown_pct'] == slowdown].sort_values('seq_len')

        for strategy in strategies:
            linestyle = '--' if strategy == 'reference_homogeneous' else '-'
            ax.plot(subset['seq_len'], subset[strategy],
                   marker='o', linewidth=2, markersize=6,
                   color=COLORS[strategy], label=LABELS[strategy],
                   linestyle=linestyle)

        ax.set_xlabel('Sequence Length')
        ax.set_ylabel('Latency (ms)')
        ax.set_title(f'MPS = {slowdown}%')
        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{int(x/1024)}K'))
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=5, bbox_to_anchor=(0.5, 0.02))
    fig.suptitle('Latency Scaling with Sequence Length', fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0.1, 1, 0.93])

    plt.savefig(os.path.join(output_dir, 'latency_scaling.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'latency_scaling.pdf'), bbox_inches='tight')
    plt.close()


def plot_heatmap(df, pivot, output_dir):
    """Heatmap of uneven vs even speedup."""
    heatmap_data = pivot.pivot(index='seq_len', columns='slowdown_pct', values='uneven_speedup')
    heatmap_data = heatmap_data.sort_index(ascending=True)
    heatmap_data = heatmap_data[sorted(heatmap_data.columns, reverse=True)]

    fig, ax = plt.subplots(figsize=(10, 6))

    im = ax.imshow(heatmap_data.values, cmap='RdYlGn', aspect='auto', vmin=0.8, vmax=5)

    ax.set_xticks(np.arange(len(heatmap_data.columns)))
    ax.set_yticks(np.arange(len(heatmap_data.index)))
    ax.set_xticklabels([f'{int(x)}%' for x in heatmap_data.columns])
    ax.set_yticklabels([f'{int(x/1024)}K' for x in heatmap_data.index])

    ax.set_xlabel('MPS %')
    ax.set_ylabel('Sequence Length')
    ax.set_title('Speedup: Uneven vs Even Split')

    for i in range(len(heatmap_data.index)):
        for j in range(len(heatmap_data.columns)):
            val = heatmap_data.values[i, j]
            color = 'white' if val > 2.5 or val < 1.2 else 'black'
            ax.text(j, i, f'{val:.1f}x', ha='center', va='center', color=color, fontsize=9)

    plt.colorbar(im, ax=ax, label='Speedup')
    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, 'speedup_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'speedup_heatmap.pdf'), bbox_inches='tight')
    plt.close()


def plot_extreme_heterogeneity(df, pivot, output_dir):
    """Bar chart at 10% MPS (extreme case)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    subset = pivot[pivot['slowdown_pct'] == 10].sort_values('seq_len')

    x = np.arange(len(subset))
    width = 0.18
    strategies = ['even', 'uneven', 'lut', 'formula', 'reference_homogeneous']

    for i, strategy in enumerate(strategies):
        offset = (i - 2) * width
        bars = ax.bar(x + offset, subset[strategy], width,
                     label=LABELS[strategy], color=COLORS[strategy], alpha=0.85)

        for bar in bars:
            height = bar.get_height()
            label = f'{height:.0f}' if height < 1000 else f'{height/1000:.1f}K'
            ax.annotate(label, xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords='offset points',
                       ha='center', va='bottom', fontsize=7, rotation=90)

    ax.set_xlabel('Sequence Length')
    ax.set_ylabel('Latency (ms)')
    ax.set_title('Performance at 10% MPS (Extreme Heterogeneity)')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{int(s/1024)}K' for s in subset['seq_len']])
    ax.set_yscale('log')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, 'extreme_heterogeneity.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'extreme_heterogeneity.pdf'), bbox_inches='tight')
    plt.close()


def print_summary(pivot):
    """Print summary table for 10% MPS."""
    summary = pivot[pivot['slowdown_pct'] == 10][['seq_len', 'even', 'uneven', 'lut', 'formula', 'reference_homogeneous']].copy()
    summary['uneven_speedup'] = (summary['even'] / summary['uneven']).round(2)
    summary['lut_speedup'] = (summary['even'] / summary['lut']).round(2)
    summary['formula_speedup'] = (summary['even'] / summary['formula']).round(2)

    for col in ['even', 'uneven', 'lut', 'formula', 'reference_homogeneous']:
        summary[col] = summary[col].round(1)

    print("\n" + "="*80)
    print("Results at 10% MPS (extreme heterogeneity)")
    print("="*80)
    print(summary.to_string(index=False))
    print("="*80)


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else INPUT_CSV

    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading {csv_path}...")
    df, pivot = load_data(csv_path)

    print(f"Found {len(df)} rows")
    print(f"Seq lengths: {sorted(df['seq_len'].unique())}")
    print(f"MPS values: {sorted(df['slowdown_pct'].unique())}")

    print(f"\nGenerating plots to {OUTPUT_DIR}/...")

    plot_latency_vs_heterogeneity(df, pivot, OUTPUT_DIR)
    print("  - latency_vs_heterogeneity.png")

    plot_efficiency(df, pivot, OUTPUT_DIR)
    print("  - efficiency_vs_heterogeneity.png")

    plot_speedup_bars(df, pivot, OUTPUT_DIR)
    print("  - speedup_over_even.png")

    plot_token_distribution(df, OUTPUT_DIR)
    print("  - token_distribution.png")

    plot_scaling(df, pivot, OUTPUT_DIR)
    print("  - latency_scaling.png")

    plot_heatmap(df, pivot, OUTPUT_DIR)
    print("  - speedup_heatmap.png")

    plot_extreme_heterogeneity(df, pivot, OUTPUT_DIR)
    print("  - extreme_heterogeneity.png")

    print_summary(pivot)

    print("\nDone.")


if __name__ == "__main__":
    main()
