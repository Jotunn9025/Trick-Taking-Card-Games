"""Unit tests for the statistical primitives."""

import numpy as np
import pytest

from ttcg.analysis.statistics import (
    bootstrap_winrate_diff_ci,
    mcnemar_test,
    significance_label,
    winrate_ci,
)


class TestWinrateCI:
    def test_zero_total(self):
        wr, lo, hi = winrate_ci(0, 0)
        assert wr == lo == hi == 0.0

    def test_zero_wins(self):
        wr, lo, hi = winrate_ci(0, 20)
        assert wr == 0.0
        assert lo == 0.0
        assert hi > 0.0

    def test_all_wins(self):
        wr, lo, hi = winrate_ci(20, 20)
        assert wr == 1.0
        assert hi == 1.0
        assert lo < 1.0

    def test_half(self):
        wr, lo, hi = winrate_ci(10, 20)
        assert wr == pytest.approx(0.5)
        assert lo < wr < hi


class TestMcNemar:
    def test_known_table(self):
        h = np.array([1, 1, 1, 0, 0, 0, 1, 1, 1, 1])
        p = np.array([1, 1, 0, 0, 0, 1, 1, 0, 0, 0])
        mc = mcnemar_test(h, p)

        assert mc["both_win"] == 3
        assert mc["h_only"] == 4
        assert mc["p_only"] == 1
        assert mc["both_lose"] == 2
        assert mc["n_discordant"] == 5
        # chi2 with continuity correction: (|4-1|-1)^2 / 5 = 0.8
        assert mc["chi2"] == pytest.approx(0.8)
        from scipy import stats

        assert mc["p_value"] == pytest.approx(1 - stats.chi2.cdf(0.8, df=1))

    def test_no_discordant_pairs(self):
        h = np.array([1, 0, 1, 0])
        p = h.copy()
        mc = mcnemar_test(h, p)
        assert mc["n_discordant"] == 0
        assert mc["chi2"] == 0.0
        assert mc["p_value"] == 1.0

    def test_significance_labels(self):
        assert significance_label(0.0009) == "***"
        assert significance_label(0.005) == "**"
        assert significance_label(0.03) == "*"
        assert significance_label(0.2) == "n.s."


class TestBootstrap:
    def test_deterministic_seed(self):
        rng = np.random.RandomState(0)
        h = rng.randint(0, 2, 200)
        p = rng.randint(0, 2, 200)
        lo1, hi1, _ = bootstrap_winrate_diff_ci(h, p, n_boot=500, seed=7)
        lo2, hi2, _ = bootstrap_winrate_diff_ci(h, p, n_boot=500, seed=7)
        assert lo1 == lo2 and hi1 == hi2

    def test_bounds_in_range(self):
        rng = np.random.RandomState(1)
        h = rng.randint(0, 2, 100)
        p = rng.randint(0, 2, 100)
        lo, hi, diffs = bootstrap_winrate_diff_ci(h, p, n_boot=500)
        assert lo <= hi
        assert -1.0 <= lo <= hi <= 1.0
        assert diffs.shape == (500,)

    def test_identical_arrays_contain_zero(self):
        rng = np.random.RandomState(2)
        wins = rng.randint(0, 2, 200)
        lo, hi, mean_diff = bootstrap_winrate_diff_ci(wins, wins.copy(), n_boot=500)
        assert lo <= 0.0 <= hi
        assert abs(mean_diff.mean()) < 0.05