"""Trend analysis for training metrics (port of the original ``2.py``).

Computes the linear-regression slope of every numeric metric vs episode and
plots each with seaborn regplots.

Usage:
    uv sync --extra dashboards
    uv run ttcg-trends --csv checkpoints/training_metrics.csv
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd


def analyze_trends(df: pd.DataFrame, x_col: str) -> pd.DataFrame:
    """
    Calculate the slope of the trend for every numeric column and plot the
    results (seaborn regplots arranged in a grid).
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols.remove(x_col)

    trend_results = []

    n_cols = 2
    n_rows = (len(numeric_cols) + 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows))
    axes = axes.flatten()

    for i, col in enumerate(numeric_cols):
        clean = df[[x_col, col]].dropna()
        slope = float(np.polyfit(clean[x_col], clean[col], 1)[0]) if len(clean) > 1 else 0.0
        direction = "Increasing" if slope > 0 else "Decreasing"
        trend_results.append({"Metric": col, "Slope": slope, "Trend": direction})

        sns.regplot(x=x_col, y=col, data=clean, ax=axes[i],
                    line_kws={"color": "red", "alpha": 0.5})
        axes[i].set_title(f"{col}\nSlope: {slope:.2e} ({direction})")
        axes[i].grid(True, linestyle="--", alpha=0.6)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()
    return pd.DataFrame(trend_results)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Trend analysis of training metrics")
    parser.add_argument("--csv", type=str, default="checkpoints/training_metrics.csv",
                        help="Path to training_metrics.csv")
    args = parser.parse_args(argv)

    import os

    if not os.path.exists(args.csv):
        for candidate in ["checkpoints17/training_metrics.csv", "checkpoints/training_metrics.csv"]:
            if os.path.exists(candidate):
                args.csv = candidate
                break

    df = pd.read_csv(args.csv)
    summary = analyze_trends(df, "episode")

    print("\n--- Trend Summary ---")
    print(summary)


if __name__ == "__main__":
    main()