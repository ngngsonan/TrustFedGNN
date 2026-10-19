"""Performance and fairness metrics.

Performance
    AUC-ROC, Average Precision (AP), F1, accuracy, FPR@k%TPR, plus the per-group
    (S=0 / S=1) AUC and AP that make "levelling down" -- closing a fairness gap
    by degrading the worse-off group -- visible rather than hidden inside a
    global average.
Group fairness (binary sensitive attribute S in {0,1})
    - Demographic Parity Difference (DPD)
    - Equal Opportunity Difference (EOD)
    - Equalized Odds (EO)
All fairness metrics accept *soft* probabilities and an optional decision
threshold; DPD is reported on soft scores (mean predicted probability gap),
matching the differentiable training objective, while EOD/EO use hard
predictions at the threshold, matching their textbook definitions.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)


def _np(x):
    import torch

    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _scores(x):
    """Sanitised probability scores so individual metric functions do not raise
    on a diverged model. NOTE: this substitution is *lossy* -- it maps a dead
    model onto the constant 0.5, which reads as ``auc=0.5, dpd=0.0`` i.e.
    "chance accuracy, perfect fairness". Callers that report fairness MUST gate
    on :func:`diverged` first; ``all_metrics`` does (SPEC 4.0(c))."""
    return np.nan_to_num(_np(x).ravel().astype(float), nan=0.5, posinf=1.0, neginf=0.0)


def diverged(y_score) -> bool:
    """True if the model produced any non-finite score.

    SPEC 4.0(c) / proposal 1.3.1(D1): a diverged model has no defined utility or
    fairness. Reporting ``dpd=0.0`` for it -- which is what ``_scores`` would
    silently yield -- scores a dead model as perfectly fair, a false positive in
    the method's own favour. Such runs must surface as NaN and be excluded from
    aggregates, not averaged in as zeros."""
    return not bool(np.isfinite(_np(y_score).ravel().astype(float)).all())


# --------------------------------------------------------------------------- #
# Performance
# --------------------------------------------------------------------------- #
def auc_roc(y_true, y_score) -> float:
    y_true, y_score = _np(y_true).ravel(), _scores(y_score)
    try:
        return float(roc_auc_score(y_true, y_score))
    except ValueError:
        return 0.5


def average_precision(y_true, y_score) -> float:
    y_true, y_score = _np(y_true).ravel(), _scores(y_score)
    try:
        return float(average_precision_score(y_true, y_score))
    except ValueError:
        return float(y_true.mean())


def f1(y_true, y_score, threshold: float = 0.5) -> float:
    y_true, y_score = _np(y_true).ravel(), _scores(y_score)
    y_pred = (y_score >= threshold).astype(int)
    try:
        return float(f1_score(y_true, y_pred))
    except ValueError:
        return 0.0


def accuracy(y_true, y_score, threshold: float = 0.5) -> float:
    y_true, y_score = _np(y_true).ravel(), _scores(y_score)
    return float(((y_score >= threshold).astype(int) == y_true).mean())


def fpr_at_tpr(y_true, y_score, target_tpr: float = 0.8) -> float:
    """False-positive rate at a target true-positive rate (operational metric)."""
    y_true, y_score = _np(y_true).ravel(), _scores(y_score)
    if len(np.unique(y_true)) < 2:
        return 0.0
    fpr, tpr, _ = roc_curve(y_true, y_score)
    idx = np.searchsorted(tpr, target_tpr)
    idx = min(idx, len(fpr) - 1)
    return float(fpr[idx])


# --------------------------------------------------------------------------- #
# Fairness
# --------------------------------------------------------------------------- #
def demographic_parity_difference(y_score, sensitive, threshold=None) -> float:
    """|E[score | S=0] - E[score | S=1]| (soft) or rate gap (if threshold set)."""
    y_score, s = _scores(y_score), _np(sensitive).ravel()
    v = (y_score >= threshold).astype(float) if threshold is not None else y_score
    m0, m1 = s == 0, s == 1
    if m0.sum() == 0 or m1.sum() == 0:
        return 0.0
    return float(abs(v[m0].mean() - v[m1].mean()))


def equal_opportunity_difference(y_true, y_score, sensitive, threshold=0.5) -> float:
    """|TPR(S=0) - TPR(S=1)|  (fairness on the positive class only)."""
    y_true, y_score, s = _np(y_true).ravel(), _scores(y_score), _np(sensitive).ravel()
    y_pred = (y_score >= threshold).astype(int)

    def tpr(group):
        pos = (y_true == 1) & group
        return y_pred[pos].mean() if pos.sum() > 0 else 0.0

    return float(abs(tpr(s == 0) - tpr(s == 1)))


def equalized_odds(y_true, y_score, sensitive, threshold=0.5) -> float:
    """max(|TPR gap|, |FPR gap|) across groups."""
    y_true, y_score, s = _np(y_true).ravel(), _scores(y_score), _np(sensitive).ravel()
    y_pred = (y_score >= threshold).astype(int)

    def rate(group, label, pred):
        m = (y_true == label) & group
        return (y_pred[m] == pred).mean() if m.sum() > 0 else 0.0

    tpr_gap = abs(rate(s == 0, 1, 1) - rate(s == 1, 1, 1))
    fpr_gap = abs(rate(s == 0, 0, 1) - rate(s == 1, 0, 1))
    return float(max(tpr_gap, fpr_gap))


def _group_subset(y_true, y_score, sensitive, group):
    """(y, score) restricted to nodes with ``sensitive == group``; None if the
    subset cannot support a ranking metric (empty, or single-class labels)."""
    s = _np(sensitive).ravel()
    m = (s == group)
    if m.sum() == 0:
        return None
    yt = _np(y_true).ravel()[m]
    if len(np.unique(yt)) < 2:
        return None
    return yt, _scores(y_score)[m]


def auc_roc_group(y_true, y_score, sensitive, group) -> float:
    """AUC-ROC computed on one sensitive group only.

    Returns NaN -- not 0.5 -- when the group is empty or carries a single label
    class, because AUC is undefined there. A 0.5 would read as "this group is
    predicted at chance", which is a measurement the data cannot support and
    which would silently dilute any per-group aggregate."""
    sub = _group_subset(y_true, y_score, sensitive, group)
    if sub is None:
        return float("nan")
    try:
        return float(roc_auc_score(sub[0], sub[1]))
    except ValueError:
        return float("nan")


def average_precision_group(y_true, y_score, sensitive, group) -> float:
    """Average precision computed on one sensitive group only.

    NaN on an empty or single-class group: with all-negative labels sklearn
    returns 0.0 and with all-positive labels 1.0, and neither is a comparable
    utility number next to the other group's AP."""
    sub = _group_subset(y_true, y_score, sensitive, group)
    if sub is None:
        return float("nan")
    try:
        return float(average_precision_score(sub[0], sub[1]))
    except ValueError:
        return float("nan")


def sensitive_homophily(edge_index, sensitive) -> float:
    """h_s = P[s_v == s_u | (v,u) in E]. Sensitive attribute homophily of the graph."""
    s = _np(sensitive).ravel()
    if isinstance(edge_index, torch.Tensor):
        src, dst = edge_index[0].cpu().numpy(), edge_index[1].cpu().numpy()
    else:
        src, dst = np.asarray(edge_index[0]), np.asarray(edge_index[1])
    if len(src) == 0:
        return 0.0
    return float((s[src] == s[dst]).astype(float).mean())


def all_metrics(y_true, y_score, sensitive, threshold: float = 0.5) -> Dict[str, float]:
    """Convenience: compute the full metric suite in one call.

    A diverged model (any non-finite score) yields NaN for every metric plus
    ``diverged=1.0``, rather than the pre-4.0(c) behaviour of silently reporting
    ``auc=0.5, dpd=0.0`` -- see :func:`diverged`."""
    if diverged(y_score):
        keys = ("auc", "ap", "f1", "acc", "fpr@80tpr", "dpd", "dpd_soft", "dpd_hard",
                "eod", "eo", "pred_std",
                "auc_group0", "auc_group1", "ap_group0", "ap_group1")
        return {**{k: float("nan") for k in keys}, "diverged": 1.0}
    
    scores = _scores(y_score)
    dpd_soft = demographic_parity_difference(y_score, sensitive)
    dpd_hard = demographic_parity_difference(y_score, sensitive, threshold=threshold)
    
    return {
        "diverged": 0.0,
        "auc": auc_roc(y_true, y_score),
        "ap": average_precision(y_true, y_score),
        "f1": f1(y_true, y_score, threshold),
        "acc": accuracy(y_true, y_score, threshold),
        "fpr@80tpr": fpr_at_tpr(y_true, y_score, 0.8),
        "dpd": dpd_soft,
        "dpd_soft": dpd_soft,
        "dpd_hard": dpd_hard,
        "eod": equal_opportunity_difference(y_true, y_score, sensitive, threshold),
        "eo": equalized_odds(y_true, y_score, sensitive, threshold),
        "pred_std": float(np.std(scores)),
        # Per-group utility. A DPD improvement bought by degrading the minority
        # group ("levelling down") is invisible in the global auc/ap; it shows up
        # here as auc_group{g} falling while the gap closes. NaN when a group is
        # empty or single-class -- see auc_roc_group.
        "auc_group0": auc_roc_group(y_true, y_score, sensitive, 0),
        "auc_group1": auc_roc_group(y_true, y_score, sensitive, 1),
        "ap_group0": average_precision_group(y_true, y_score, sensitive, 0),
        "ap_group1": average_precision_group(y_true, y_score, sensitive, 1),
    }


# Backwards-compatible aliases (older code / notebooks referenced these names)
calculate_auc = auc_roc
calculate_f1 = f1


def calculate_dpd(y_score, sensitive) -> float:
    return demographic_parity_difference(y_score, sensitive)


def weight_oscillation(weights_history: list) -> float:
    """Mean L1 variation of aggregation weight vectors across communication rounds.

    Omega_w = 1 / (T - 1) * sum_{t=2}^T ||w^(t) - w^(t-1)||_1

    Measures aggregation stability. Unsmoothed Frank-Wolfe/Shapley jumps on simplex
    vertices (Omega_w >= 0.30); temporal EMA smoothing bounds it (Omega_w <= 0.05).

    Returns NaN -- NOT 0.0 -- when fewer than two weight vectors are available.
    Aggregators that are not expressible as client weights (coordinate median,
    trimmed_mean) report no weight vector at all, so a 0.0 here reads as "perfectly
    stable" for a rule whose stability was never measured. NaN keeps the
    unmeasured case visibly distinct from a measured zero; render it as "--".
    """
    valid = [np.asarray(w, dtype=float) for w in weights_history if w is not None and len(w) > 0]
    if len(valid) < 2:
        return float("nan")
    diffs = [np.sum(np.abs(valid[t] - valid[t - 1])) for t in range(1, len(valid))]
    return float(np.mean(diffs))

