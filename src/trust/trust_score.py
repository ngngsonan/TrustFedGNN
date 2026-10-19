"""Composite Trust Score for federated models.

Trustworthy ML requires balancing several axes at once; a model that is
accurate but unfair, or fair but privacy-leaking or attack-fragile, is not
trustworthy. We summarise a model with a single interpretable score in [0, 1]
aggregating five normalised sub-scores:

    utility      AUC-ROC (already in [0,1]).
    fairness     1 - clip((DPD + EOD)/2 / F_REF, 0, 1).
    privacy      clip(1 - eps / EPS_MAX, 0, 1)  (0 if no DP applied).
    robustness   AUC retained under attack / clean AUC (1 if not evaluated).
    calibration  1 - clip(ECE / ECE_REF, 0, 1).

Reference constants are made explicit (and configurable) so the mapping is
transparent -- the score is a communication tool, not a hidden benchmark. The
aggregate is a weighted power mean; p=1 is the arithmetic mean, p->0 the
geometric mean (which harshly penalises any single weak axis -- the stricter,
"must be good at everything" reading of trustworthiness).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

F_REF = 0.20        # (DPD+EOD)/2 at/above which fairness score hits 0
EPS_MAX = 16.0      # epsilon at/above which privacy score hits 0
ECE_REF = 0.20      # ECE at/above which calibration score hits 0


def _clip01(x):
    return float(min(1.0, max(0.0, x)))


@dataclass
class TrustWeights:
    utility: float = 1.0
    fairness: float = 1.0
    privacy: float = 1.0
    robustness: float = 1.0
    calibration: float = 1.0


def sub_scores(metrics: Dict[str, float], *, epsilon: Optional[float] = None,
               clean_auc: Optional[float] = None, ece: Optional[float] = None
               ) -> Dict[str, float]:
    auc = metrics.get("auc", 0.5)
    dpd = metrics.get("dpd", 0.0)
    eod = metrics.get("eod", 0.0)
    s = {"utility": _clip01(auc),
         "fairness": _clip01(1.0 - ((dpd + eod) / 2.0) / F_REF)}
    # Privacy is ALWAYS scored: a method that applies no differential privacy
    # provides zero privacy (epsilon = infinity), so it must not be rewarded by
    # simply omitting the axis. This is what makes the composite favour methods
    # that are trustworthy on *every* dimension, not just the ones they target.
    eps = epsilon if epsilon is not None else metrics.get("epsilon")
    s["privacy"] = _clip01(1.0 - eps / EPS_MAX) if (eps is not None and np.isfinite(eps)) else 0.0
    if clean_auc:
        s["robustness"] = _clip01(auc / clean_auc)
    e = ece if ece is not None else metrics.get("ece")
    if e is not None:
        s["calibration"] = _clip01(1.0 - e / ECE_REF)
    return s


def trust_score(metrics: Dict[str, float], weights: Optional[TrustWeights] = None,
                p: float = 0.0, **kw) -> float:
    """Weighted power-mean of the available sub-scores. p=0 -> geometric mean."""
    s = sub_scores(metrics, **kw)
    w = weights or TrustWeights()
    wd = {k: getattr(w, k) for k in s}
    vals = np.array([s[k] for k in s])
    wts = np.array([wd[k] for k in s])
    wts = wts / wts.sum()
    if p == 0:  # geometric mean
        return float(np.exp(np.sum(wts * np.log(np.clip(vals, 1e-6, 1.0)))))
    return float((np.sum(wts * vals ** p)) ** (1.0 / p))


def trust_report(metrics: Dict[str, float], **kw) -> Dict[str, float]:
    s = sub_scores(metrics, **kw)
    return {**{f"trust_{k}": v for k, v in s.items()},
            "trust_geomean": trust_score(metrics, p=0.0, **kw),
            "trust_mean": trust_score(metrics, p=1.0, **kw)}
