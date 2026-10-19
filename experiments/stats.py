"""Statistical testing and reporting module.

Implements paired Wilcoxon signed-rank test, Cohen's d_z effect size,
bootstrap 95% confidence intervals, and family-wise Holm-Bonferroni correction.
"""
from typing import Dict, List, Tuple
import numpy as np
import math

from scipy.stats import wilcoxon


def paired_report(ours: List[float], base: List[float], lower_is_better: bool = False) -> Dict:
    """Seed-paired comparison. Returns the four quantities every results table reports."""
    o, b = np.asarray(ours, float), np.asarray(base, float)
    assert o.shape == b.shape, "comparison must be paired on the same seeds"
    d = o - b
    
    # Wilcoxon signed-rank test
    if np.all(d == 0):
        stat, p = 0.0, 1.0
    else:
        try:
            stat, p = wilcoxon(o, b)
        except Exception:
            stat, p = 0.0, 1.0

    wins = int(np.sum(d < 0) if lower_is_better else np.sum(d > 0))
    
    # paired Cohen's dz
    std_d = float(d.std(ddof=1)) if len(d) > 1 else 0.0
    dz = float(d.mean() / std_d) if std_d > 0 else 0.0
    
    # 95% bootstrap CI of the mean difference
    rng = np.random.default_rng(0)
    boot = [rng.choice(d, size=len(d), replace=True).mean() for _ in range(10_000)]
    ci95 = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
    
    return {
        "mean_diff": float(d.mean()),
        "ci95": ci95,
        "cohens_dz": dz,
        "wins": f"{wins}/{len(d)}",
        "p_wilcoxon": float(p),
    }


def holm_bonferroni(pvals: Dict[str, float], alpha: float = 0.05) -> Dict[str, bool]:
    """Multiple-comparison correction when comparing Ours against N baselines, per family.

    A non-finite p-value (nan/inf) corrupts the WHOLE family, not just its own entry:
    comparisons with nan are always False, so sorted() returns an arbitrary order, and
    because `reject` is cumulative one misordered entry flips every entry after it.
    Observed on real data: a nan turned p=0.0020 into "not significant" while
    p=0.0039 became "significant". Refuse at the gate instead.
    """
    bad = {k: v for k, v in pvals.items() if not math.isfinite(v)}
    if bad:
        raise ValueError(
            f"holm_bonferroni received non-finite p-values: {bad}. "
            "Drop (and report) contrasts that could not be computed BEFORE correcting; "
            "do not let them enter the test family."
        )
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out = {}
    reject = True
    for i, (k, p) in enumerate(items):
        thresh = alpha / (m - i)
        reject = reject and (p <= thresh)
        out[k] = reject
    return out
