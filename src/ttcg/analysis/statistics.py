"""Statistical primitives for the paired-deal win-rate analysis.

- Clopper-Pearson exact binomial confidence intervals for a win rate.
- Nonparametric bootstrap CI for the Hybrid - Pure win-rate difference.
- McNemar's test for paired binary outcomes (the primary test: does Hybrid
  win more often than Pure on the SAME deals?).
- Data loaders for the game-log CSVs.
"""

from __future__ import annotations

import os
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy import stats


def winrate_ci(wins: int, total: int, ci: float = 0.95) -> Tuple[float, float, float]:
    """Clopper-Pearson exact CI for a proportion. Returns (point, lo, hi)."""
    if total == 0:
        return 0.0, 0.0, 0.0
    alpha = 1 - ci
    lo = stats.beta.ppf(alpha / 2, wins, total - wins + 1) if wins > 0 else 0.0
    hi = stats.beta.ppf(1 - alpha / 2, wins + 1, total - wins) if wins < total else 1.0
    return wins / total, lo, hi


def bootstrap_winrate_diff_ci(
    h_wins_arr: np.ndarray,
    p_wins_arr: np.ndarray,
    n_boot: int = 10000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, np.ndarray]:
    """
    Bootstrap CI on the difference in win rates (Hybrid - Pure).

    ``h_wins_arr`` / ``p_wins_arr``: paired 0/1 win indicators (same deals).
    Returns (lo, hi, all bootstrap diffs).
    """
    rng = np.random.RandomState(seed)
    n = len(h_wins_arr)
    diffs = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        diffs.append(h_wins_arr[idx].mean() - p_wins_arr[idx].mean())
    diffs = np.array(diffs)
    lo = np.percentile(diffs, (1 - ci) / 2 * 100)
    hi = np.percentile(diffs, (1 + ci) / 2 * 100)
    return lo, hi, diffs


def mcnemar_test(h_wins: np.ndarray, p_wins: np.ndarray) -> dict:
    """
    McNemar's test for paired binary outcomes.

    Tests whether Hybrid wins more often than Pure on the same deals, using
    only the discordant pairs. Returns the contingency counts, chi-squared
    statistic, and p-value (with a continuity correction).
    """
    both_win = int(((h_wins == 1) & (p_wins == 1)).sum())
    h_only = int(((h_wins == 1) & (p_wins == 0)).sum())  # H wins, P loses
    p_only = int(((h_wins == 0) & (p_wins == 1)).sum())  # H loses, P wins
    both_lose = int(((h_wins == 0) & (p_wins == 0)).sum())

    n_disc = h_only + p_only
    if n_disc == 0:
        return {
            "both_win": both_win, "h_only": h_only, "p_only": p_only,
            "both_lose": both_lose, "n_discordant": 0, "chi2": 0.0, "p_value": 1.0,
        }

    chi2 = (abs(h_only - p_only) - 1) ** 2 / n_disc
    p_value = 1 - stats.chi2.cdf(chi2, df=1)

    return {
        "both_win": both_win,
        "h_only": h_only,
        "p_only": p_only,
        "both_lose": both_lose,
        "n_discordant": n_disc,
        "chi2": chi2,
        "p_value": p_value,
    }


def significance_label(p_value: float) -> str:
    """Map a p-value to a significance star label."""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "n.s."


# ------------------------------------------------------------------ data loading

def load_hvp_data(log_dir: str):
    """Load hybrid_vs_pure_log.csv, or None if absent/empty."""
    path = os.path.join(log_dir, "hybrid_vs_pure_log.csv")
    if not os.path.exists(path):
        print(f"  [SKIP] {path} not found — no Hybrid vs Pure data.")
        return None
    df = pd.read_csv(path)
    if df.empty:
        print(f"  [SKIP] {path} is empty.")
        return None
    return df


def load_eval_data(log_dir: str):
    """Load eval_games_log.csv, or None if absent/empty."""
    path = os.path.join(log_dir, "eval_games_log.csv")
    if not os.path.exists(path):
        print(f"  [SKIP] {path} not found — no eval data.")
        return None
    df = pd.read_csv(path)
    if df.empty:
        print(f"  [SKIP] {path} is empty.")
        return None
    return df