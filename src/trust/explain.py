"""Explainability for federated fair GNNs.

Two complementary explanations:

1. Edge-attribution via FSER attention. TrustFedGNN already exposes per-edge
   attention (averaged over heads/layers). We surface the edges the model
   attends to most, and -- crucially for a *fairness* audit -- report how much
   attention mass sits on cross-group vs same-group edges, showing that FSER
   shifts attention away from the biased cross-group edges it is designed to
   suppress.

2. Gradient x input feature attribution (a federated-friendly, model-agnostic
   saliency) with a fairness-attribution twist: we attribute the *demographic
   parity gap* to input features, identifying which features drive disparity.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch


@torch.no_grad()
def fser_edge_attention(model, data) -> Dict:
    """Attention mass on cross-group vs same-group edges (fairness lens)."""
    model.eval()
    _ = model(data.x, data.edge_index, data.sensitive_attr)
    att = model.edge_attention() if hasattr(model, "edge_attention") else None
    if att is None:
        return {"available": False}
    src, dst = data.edge_index
    cross = (data.sensitive_attr[src] != data.sensitive_attr[dst])
    att = att.cpu()
    total = att.sum().item() + 1e-12
    cross_mass = att[cross].sum().item() / total
    cross_frac = float(cross.float().mean())
    return {
        "available": True,
        "cross_group_edge_fraction": cross_frac,
        "cross_group_attention_mass": cross_mass,
        # <1 means FSER attends to cross-group edges *less* than their prevalence
        "attention_bias_ratio": cross_mass / (cross_frac + 1e-12),
        "mean_beta": float(torch.stack([l.beta.detach() for l in model.layers]).mean())
                     if hasattr(model, "layers") else None,
    }


def feature_fairness_attribution(model, data, split="test", top_k: int = 10) -> Dict:
    """Attribute the demographic-parity gap to input features via the gradient
    of (mu0 - mu1) w.r.t. the input features (saliency of disparity)."""
    model.eval()
    mask = getattr(data, f"{split}_mask")
    x = data.x.clone().requires_grad_(True)
    pred = model(x, data.edge_index, data.sensitive_attr)[mask]
    s = data.sensitive_attr[mask]
    if (s == 0).sum() == 0 or (s == 1).sum() == 0:
        return {"available": False}
    gap = pred[s == 0].mean() - pred[s == 1].mean()
    grad = torch.autograd.grad(gap, x, retain_graph=False)[0]
    # importance = mean |grad| over nodes, per feature
    imp = grad[mask].abs().mean(0).detach().cpu().numpy()
    order = np.argsort(imp)[::-1][:top_k]
    return {
        "available": True,
        "top_features": order.tolist(),
        "top_importances": imp[order].tolist(),
        "total_disparity_sensitivity": float(imp.sum()),
    }


def explanation_report(model, data, split="test") -> Dict:
    return {
        "edge_attention": fser_edge_attention(model, data),
        "feature_attribution": feature_fairness_attribution(model, data, split),
    }
