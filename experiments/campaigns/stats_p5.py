"""Statistical testing and re-analysis tools for P5 revision (P5-K11).

Provides:
- paired_bootstrap_ci: nonparametric percentile bootstrap confidence intervals
- wilcoxon_p: two-sided Wilcoxon signed-rank test
- sign_test_p: exact two-sided binomial sign test (zeros dropped)
- holm: Holm-Bonferroni step-down family-wise error rate control
- noninferiority: one-sided 95% non-inferiority check for AUC (margin delta)
- factorial_2x2: main and interaction effects for 2x2 factorial designs with bootstrap CIs
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple, Union

import numpy as np
from scipy.stats import binomtest, wilcoxon


def paired_bootstrap_ci(
    d: Sequence[float],
    n: int = 10000,
    seed: int = 0,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    """Compute (1 - alpha) two-sided percentile bootstrap confidence interval of mean(d)."""
    arr = np.asarray(d, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(arr), size=(n, len(arr)))
    boot_means = arr[indices].mean(axis=1)
    lower = float(np.percentile(boot_means, 100.0 * (alpha / 2.0)))
    upper = float(np.percentile(boot_means, 100.0 * (1.0 - alpha / 2.0)))
    return lower, upper


def wilcoxon_p(d: Sequence[float]) -> float:
    """Compute two-sided Wilcoxon signed-rank test p-value against zero median."""
    arr = np.asarray(d, dtype=float)
    arr = arr[np.isfinite(arr)]
    nonzero = arr[arr != 0]
    if len(nonzero) == 0:
        return 1.0
    try:
        res = wilcoxon(nonzero, alternative="two-sided")
        return float(res.pvalue)
    except ValueError:
        return 1.0


def sign_test_p(d: Sequence[float]) -> float:
    """Compute exact two-sided sign test p-value (zeros dropped)."""
    arr = np.asarray(d, dtype=float)
    arr = arr[np.isfinite(arr)]
    pos = int((arr > 0).sum())
    neg = int((arr < 0).sum())
    n = pos + neg
    if n == 0:
        return 1.0
    res = binomtest(min(pos, neg), n, p=0.5, alternative="two-sided")
    return float(res.pvalue)


def holm(pvals: Sequence[float]) -> List[float]:
    """Compute Holm-Bonferroni step-down adjusted p-values."""
    if not pvals:
        return []
    m = len(pvals)
    indexed = sorted(enumerate(pvals), key=lambda x: x[1])
    adj_sorted = []
    running_max = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        # (m - rank) multiplier, capped at 1.0
        adj_p = min(1.0, (m - rank) * float(p))
        running_max = max(running_max, adj_p)
        adj_sorted.append((orig_idx, running_max))

    # Restore original input order
    adj_sorted.sort(key=lambda x: x[0])
    return [float(p) for _, p in adj_sorted]


def noninferiority(
    d_auc: Sequence[float],
    delta: float = 0.01,
    n: int = 10000,
    seed: int = 0,
) -> Dict[str, Union[float, bool]]:
    """Non-inferiority test: returns 95% one-sided lower confidence bound and pass flag."""
    arr = np.asarray(d_auc, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {"ci_lower": float("nan"), "delta": delta, "pass": False}
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(arr), size=(n, len(arr)))
    boot_means = arr[indices].mean(axis=1)
    # 95% one-sided lower bound corresponds to 5th percentile
    ci_lower = float(np.percentile(boot_means, 5.0))
    passed = bool(ci_lower > -delta)
    return {
        "ci_lower": ci_lower,
        "delta": delta,
        "pass": passed,
    }


def factorial_2x2(
    y: Dict[str, Sequence[float]],
    n_boot: int = 10000,
    seed: int = 0,
) -> Dict[str, Any]:
    """Compute per-seed main and interaction effects in a 2x2 factorial design with CIs.

    Expected keys in y:
    - RS_R: Rectify-first = True, Magnitude = Rescale
    - RS_C: Rectify-first = True, Magnitude = Clip
    - SR_R: Rectify-first = False, Magnitude = Rescale
    - SR_C: Rectify-first = False, Magnitude = Clip

    Definitions per seed:
    - order = mean(RS - SR) across two magnitude levels
    - magnitude = mean(R - C) across two order levels
    - interaction = (RS_R - RS_C) - (SR_R - SR_C)
    """
    rs_r = np.asarray(y["RS_R"], dtype=float)
    rs_c = np.asarray(y["RS_C"], dtype=float)
    sr_r = np.asarray(y["SR_R"], dtype=float)
    sr_c = np.asarray(y["SR_C"], dtype=float)

    n_seeds = len(rs_r)
    assert len(rs_c) == n_seeds and len(sr_r) == n_seeds and len(sr_c) == n_seeds, (
        f"Mismatched seed counts: {n_seeds}, {len(rs_c)}, {len(sr_r)}, {len(sr_c)}"
    )

    # Per-seed effects
    order_per_seed = 0.5 * ((rs_r - sr_r) + (rs_c - sr_c))
    mag_per_seed = 0.5 * ((rs_r - rs_c) + (sr_r - sr_c))
    inter_per_seed = (rs_r - rs_c) - (sr_r - sr_c)

    order_ci = paired_bootstrap_ci(order_per_seed, n=n_boot, seed=seed)
    mag_ci = paired_bootstrap_ci(mag_per_seed, n=n_boot, seed=seed + 1)
    inter_ci = paired_bootstrap_ci(inter_per_seed, n=n_boot, seed=seed + 2)

    return {
        "order": {
            "mean": float(np.mean(order_per_seed)),
            "ci95": list(order_ci),
            "per_seed": order_per_seed.tolist(),
        },
        "magnitude": {
            "mean": float(np.mean(mag_per_seed)),
            "ci95": list(mag_ci),
            "per_seed": mag_per_seed.tolist(),
        },
        "interaction": {
            "mean": float(np.mean(inter_per_seed)),
            "ci95": list(inter_ci),
            "per_seed": inter_per_seed.tolist(),
        },
    }
