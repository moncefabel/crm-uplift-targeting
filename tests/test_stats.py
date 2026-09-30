import numpy as np

from src import stats as st
from src.uplift import incremental_capture


def test_srm_detects_imbalance():
    assert st.srm_pvalue([10000, 10050, 9980]) > 0.05
    assert st.srm_pvalue([10000, 11000, 9000]) < 0.001


def test_diff_in_means_ci_contains_true_effect():
    rng = np.random.default_rng(0)
    c, t = rng.binomial(1, 0.10, 50000), rng.binomial(1, 0.12, 50000)
    r = st.diff_in_means(t, c)
    assert r["ci_low"] < 0.02 < r["ci_high"]


def test_cuped_reduces_variance_with_correlated_covariate():
    rng = np.random.default_rng(1)
    x = rng.normal(size=10000)
    y = 2 * x + rng.normal(size=10000)
    y_adj, _ = st.cuped(y, x)
    assert y_adj.var() < 0.3 * y.var()


def test_perfect_ranking_captures_everything_in_top_group():
    y = np.array([1, 0, 1, 0, 0, 0, 0, 0])
    t = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    u = np.array([9, 8, 7, 6, 1, 1, 1, 1], dtype=float)
    assert incremental_capture(y, u, t, k=0.5) == 1.0
