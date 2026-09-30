"""Outils statistiques pour l'analyse d'A/B test : SRM, lift, intervalles de confiance, CUPED, MDE."""
import numpy as np
from scipy import stats


def srm_pvalue(counts, expected_share=None) -> float:
    """Test du chi-deux de Sample Ratio Mismatch : la répartition observée correspond-elle au plan ?"""
    counts = np.asarray(counts, dtype=float)
    if expected_share is None:
        expected_share = np.full(len(counts), 1 / len(counts))
    expected = counts.sum() * np.asarray(expected_share)
    return float(stats.chisquare(counts, expected).pvalue)


def diff_in_means(y_t, y_c, alpha: float = 0.05) -> dict:
    """Écart de moyennes (Welch) avec IC, p-value et lift relatif."""
    y_t, y_c = np.asarray(y_t, float), np.asarray(y_c, float)
    diff = y_t.mean() - y_c.mean()
    se = np.sqrt(y_t.var(ddof=1) / len(y_t) + y_c.var(ddof=1) / len(y_c))
    z = stats.norm.ppf(1 - alpha / 2)
    p = 2 * (1 - stats.norm.cdf(abs(diff / se)))
    return {
        "control_mean": float(y_c.mean()),
        "treatment_mean": float(y_t.mean()),
        "abs_diff": float(diff),
        "ci_low": float(diff - z * se),
        "ci_high": float(diff + z * se),
        "rel_lift": float(diff / y_c.mean()) if y_c.mean() else float("nan"),
        "p_value": float(p),
        "se": float(se),
    }


def cuped(y, x):
    """Ajustement CUPED : y - theta (x - moyenne(x)), avec x une covariable mesurée avant l'expérience."""
    y, x = np.asarray(y, float), np.asarray(x, float)
    theta = np.cov(y, x, ddof=1)[0, 1] / x.var(ddof=1)
    return y - theta * (x - x.mean()), float(theta)


def mde_proportion(p_base: float, n_per_arm: int, alpha: float = 0.05, power: float = 0.8) -> float:
    """Effet minimal détectable (absolu) pour une proportion, test bilatéral."""
    z_a, z_b = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    return float((z_a + z_b) * np.sqrt(2 * p_base * (1 - p_base) / n_per_arm))
