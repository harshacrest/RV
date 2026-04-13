"""
significance.py — Statistical significance testing for Sharpe ratios.

Provides standard error estimation, confidence intervals, and hypothesis
testing for comparing strategy Sharpe ratios. Prevents the meta-harness
from treating noise as signal.

Usage:
    from significance import sharpe_se, sharpe_diff_significant, bootstrap_sharpe_ci

    se = sharpe_se(pnl_series)
    is_sig, pval = sharpe_diff_significant(pnl_a, pnl_b)
    ci_lo, ci_hi = bootstrap_sharpe_ci(pnl_series)
"""

import numpy as np
import pandas as pd

ANNUALIZATION = 252
RISK_FREE_PCT = 5.5


def _annualized_sharpe(pnl: pd.Series) -> float:
    """Compute annualized Sharpe from daily % returns."""
    pnl = pnl.dropna()
    if len(pnl) < 10:
        return 0.0
    m = float(pnl.mean())
    s = float(pnl.std())
    if s == 0:
        return 0.0
    return (m * ANNUALIZATION - RISK_FREE_PCT) / (s * np.sqrt(ANNUALIZATION))


def sharpe_se(pnl: pd.Series) -> float:
    """Standard error of annualized Sharpe estimate.

    Uses the Lo (2002) formula:
        SE(SR) = sqrt((1 + 0.5 * SR^2) / (n - 1)) * sqrt(252)

    This accounts for non-normality in the Sharpe estimate.
    """
    pnl = pnl.dropna()
    n = len(pnl)
    if n < 10:
        return float("inf")

    sr = _annualized_sharpe(pnl) / np.sqrt(ANNUALIZATION)  # daily Sharpe
    se_daily = np.sqrt((1 + 0.5 * sr ** 2) / (n - 1))
    return float(se_daily * np.sqrt(ANNUALIZATION))


def sharpe_diff_significant(
    pnl_a: pd.Series,
    pnl_b: pd.Series,
    alpha: float = 0.05,
) -> tuple[bool, float]:
    """Test whether two Sharpe ratios are significantly different.

    Uses the Jobson-Korkie (1981) test with Memmel (2003) correction.

    Returns:
        (is_significant, z_statistic)
    """
    a = pnl_a.dropna().values
    b = pnl_b.dropna().values

    # Align to common dates if both are indexed
    if isinstance(pnl_a, pd.Series) and isinstance(pnl_b, pd.Series):
        common = pnl_a.dropna().index.intersection(pnl_b.dropna().index)
        if len(common) > 10:
            a = pnl_a.loc[common].values
            b = pnl_b.loc[common].values

    n = min(len(a), len(b))
    if n < 20:
        return False, 0.0

    # Truncate to same length
    a = a[:n]
    b = b[:n]

    mu_a, mu_b = a.mean(), b.mean()
    sig_a, sig_b = a.std(ddof=1), b.std(ddof=1)

    if sig_a == 0 or sig_b == 0:
        return False, 0.0

    sr_a = mu_a / sig_a
    sr_b = mu_b / sig_b

    # Correlation between the two return series
    rho = np.corrcoef(a, b)[0, 1]

    # Memmel (2003) corrected variance of the difference
    theta = (1 / n) * (
        2 * (1 - rho)
        + 0.5 * (sr_a ** 2 + sr_b ** 2)
        - (sr_a * sr_b * (1 + rho ** 2)) / (1 + 1e-10)
    )

    if theta <= 0:
        return False, 0.0

    z = (sr_a - sr_b) / np.sqrt(theta)

    # Two-tailed test
    from scipy.stats import norm
    p_value = 2 * (1 - norm.cdf(abs(z)))

    return p_value < alpha, float(z)


def bootstrap_sharpe_ci(
    pnl: pd.Series,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    """Bootstrap confidence interval for annualized Sharpe ratio.

    Returns:
        (ci_lower, ci_upper)
    """
    pnl = pnl.dropna().values
    n = len(pnl)
    if n < 20:
        return (-float("inf"), float("inf"))

    rng = np.random.RandomState(seed)
    boot_sharpes = []

    for _ in range(n_boot):
        sample = rng.choice(pnl, size=n, replace=True)
        m = sample.mean()
        s = sample.std(ddof=1)
        if s > 0:
            sr = (m * ANNUALIZATION - RISK_FREE_PCT) / (s * np.sqrt(ANNUALIZATION))
            boot_sharpes.append(sr)

    if not boot_sharpes:
        return (-float("inf"), float("inf"))

    boot_sharpes = np.array(boot_sharpes)
    lo = float(np.percentile(boot_sharpes, 100 * alpha / 2))
    hi = float(np.percentile(boot_sharpes, 100 * (1 - alpha / 2)))

    return (lo, hi)


def is_improvement_significant(
    pnl_baseline: pd.Series,
    pnl_candidate: pd.Series,
    min_sharpe_delta: float = 0.3,
) -> dict:
    """Comprehensive significance check for a candidate vs baseline.

    Returns a dict with:
        significant: bool — is the improvement real?
        baseline_sharpe, candidate_sharpe: the two Sharpe ratios
        delta: candidate - baseline
        baseline_se, candidate_se: standard errors
        jk_significant: Jobson-Korkie test result
        delta_exceeds_se: whether delta > max(SE_baseline, SE_candidate)
    """
    sr_base = _annualized_sharpe(pnl_baseline)
    sr_cand = _annualized_sharpe(pnl_candidate)
    delta = sr_cand - sr_base

    se_base = sharpe_se(pnl_baseline)
    se_cand = sharpe_se(pnl_candidate)
    max_se = max(se_base, se_cand)

    jk_sig, jk_z = sharpe_diff_significant(pnl_baseline, pnl_candidate)
    delta_exceeds_se = abs(delta) > max_se

    # Conservative: require BOTH JK test AND delta > SE AND delta > min threshold
    significant = jk_sig and delta_exceeds_se and delta > min_sharpe_delta

    return {
        "significant": significant,
        "baseline_sharpe": round(sr_base, 4),
        "candidate_sharpe": round(sr_cand, 4),
        "delta": round(delta, 4),
        "baseline_se": round(se_base, 4),
        "candidate_se": round(se_cand, 4),
        "jk_significant": jk_sig,
        "jk_z": round(jk_z, 4),
        "delta_exceeds_se": delta_exceeds_se,
        "verdict": "SIGNIFICANT" if significant else "NOT SIGNIFICANT",
    }


if __name__ == "__main__":
    # Quick demo with synthetic data
    np.random.seed(42)
    n = 200
    pnl_a = pd.Series(np.random.normal(0.05, 1.0, n))
    pnl_b = pd.Series(np.random.normal(0.10, 1.0, n))

    print("Significance Testing Demo")
    print(f"  Sharpe A: {_annualized_sharpe(pnl_a):.4f}, SE: {sharpe_se(pnl_a):.4f}")
    print(f"  Sharpe B: {_annualized_sharpe(pnl_b):.4f}, SE: {sharpe_se(pnl_b):.4f}")

    ci = bootstrap_sharpe_ci(pnl_a)
    print(f"  Bootstrap CI for A: [{ci[0]:.2f}, {ci[1]:.2f}]")

    sig, z = sharpe_diff_significant(pnl_a, pnl_b)
    print(f"  JK test: significant={sig}, z={z:.4f}")

    result = is_improvement_significant(pnl_a, pnl_b)
    print(f"  Improvement check: {result['verdict']}")
