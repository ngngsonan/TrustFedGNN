"""Predictive uncertainty and calibration.

MC-dropout (Gal & Ghahramani, 2016) gives per-node predictive uncertainty by
running T stochastic forward passes with dropout kept active. We report the
mean probability, predictive entropy, and the std across passes, plus two
calibration metrics -- Expected Calibration Error (ECE) and the Brier score --
which quantify whether the model's confidences are trustworthy (a core
requirement for uncertainty estimation in trustworthy ML).
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import torch


@torch.no_grad()
def mc_dropout_predict(model, x, edge_index, sensitive_attr, T: int = 20):
    """Return (mean_prob [N], epistemic_std [N], pred_entropy [N])."""
    model.eval()
    preds = []
    for _ in range(T):
        p = model(x, edge_index, sensitive_attr, mc=True)
        preds.append(p.detach())
    P = torch.stack(preds)                      # [T, N]
    mean = P.mean(0)
    std = P.std(0)
    eps = 1e-8
    entropy = -(mean * (mean + eps).log() + (1 - mean) * (1 - mean + eps).log())
    return mean, std, entropy


def expected_calibration_error(y_true, y_prob, n_bins: int = 10) -> float:
    y_true = np.asarray(y_true).ravel()
    y_prob = np.nan_to_num(np.asarray(y_prob).ravel(), nan=0.5)
    conf = np.where(y_prob >= 0.5, y_prob, 1 - y_prob)
    pred = (y_prob >= 0.5).astype(int)
    correct = (pred == y_true).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() > 0:
            ece += (m.sum() / n) * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def brier_score(y_true, y_prob) -> float:
    y_true = np.asarray(y_true).ravel()
    y_prob = np.nan_to_num(np.asarray(y_prob).ravel(), nan=0.5)
    return float(np.mean((y_prob - y_true) ** 2))


def uncertainty_report(model, data, split="test", T: int = 20) -> Dict[str, float]:
    mask = getattr(data, f"{split}_mask")
    mean, std, ent = mc_dropout_predict(model, data.x, data.edge_index,
                                        data.sensitive_attr, T)
    y = data.y[mask].cpu().numpy()
    m = mean[mask].cpu().numpy()
    s = data.sensitive_attr[mask].cpu().numpy()
    ustd = std[mask].cpu().numpy()
    # group-conditional uncertainty gap: is the model *less certain* for one
    # group?  (an under-studied but important fairness-of-uncertainty signal)
    g0 = ustd[s == 0].mean() if (s == 0).any() else 0.0
    g1 = ustd[s == 1].mean() if (s == 1).any() else 0.0
    return {
        "ece": expected_calibration_error(y, m),
        "brier": brier_score(y, m),
        "mean_epistemic_std": float(ustd.mean()),
        "mean_entropy": float(ent[mask].mean()),
        "uncertainty_gap": float(abs(g0 - g1)),
    }
