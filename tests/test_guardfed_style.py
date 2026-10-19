"""Unit tests for GuardFed-style aggregator (ADR-27, K7).
"""
import sys
from pathlib import Path
import torch
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.federated.aggregation import aggregate, compute_guardfed_holdout
from src.data.datasets import load_dataset
from src.models import build_model
from src.config import ExperimentConfig
from src.federated.client import flatten_state, load_flat_state
from src.data.partition import carve_server_holdout


def test_guardfed_selection_and_uniform_average():
    """Verify survivor set S selection and uniform averaging."""
    K = 4
    dim = 8
    g_s = torch.ones(dim)
    # u0 and u1: aligned with g_s, low EO gap -> high trust
    u0 = torch.ones(dim)
    u1 = torch.ones(dim) * 2.0
    # u2: anti-aligned (cos < 0 -> dev=0) -> trust = 0
    u2 = -torch.ones(dim)
    # u3: aligned, but high EO gap with tau=10 -> low trust
    u3 = torch.ones(dim)

    updates = [u0, u1, u2, u3]
    meta = [{"n": 100, "perf": 0.8, "dpd": 0.05, "loss": 0.2} for _ in range(K)]
    gs_fair = torch.tensor([0.0, 0.0, 0.0, 5.0])

    agg, info = aggregate(
        "guardfed_style", updates, meta,
        g_s=g_s, gs_fair=gs_fair, gs_tau=2.0, gs_gamma=0.01
    )

    # u0 (trust=1.0) and u1 (trust=1.0) > 0.01; u2 (trust=0) <= 0.01; u3 (trust=e^(-10) ~= 4.5e-5) <= 0.01
    assert info["selected"] == [0, 1], f"Expected [0, 1], got {info['selected']}"
    w = torch.tensor(info["weights"])
    assert torch.isclose(w[0], torch.tensor(0.5))
    assert torch.isclose(w[1], torch.tensor(0.5))
    assert w[2] == 0.0 and w[3] == 0.0
    expected_agg = 0.5 * u0 + 0.5 * u1
    assert torch.allclose(agg, expected_agg)


def test_guardfed_empty_s():
    """When no client passes Gamma, aggregate leaves theta unchanged (returns zero update)."""
    K = 2
    dim = 8
    g_s = torch.ones(dim)
    updates = [-torch.ones(dim), -torch.ones(dim)]
    meta = [{"n": 100, "perf": 0.8, "dpd": 0.05, "loss": 0.2} for _ in range(K)]
    gs_fair = torch.tensor([0.0, 0.0])

    agg, info = aggregate(
        "guardfed_style", updates, meta,
        g_s=g_s, gs_fair=gs_fair, gs_tau=1.0, gs_gamma=0.0
    )

    assert info["selected"] == []
    assert torch.all(agg == 0.0)
    w = torch.tensor(info["weights"])
    assert torch.all(w == 0.0)


def test_guardfed_gs_sign():
    """Verify sign convention: updates aligned with descent direction g_s get dev > 0."""
    dim = 4
    g_s = torch.tensor([1.0, 0.0, 0.0, 0.0])
    u_aligned = torch.tensor([1.0, 0.0, 0.0, 0.0])
    u_opposed = torch.tensor([-1.0, 0.0, 0.0, 0.0])

    updates = [u_aligned, u_opposed]
    meta = [{"n": 100, "perf": 0.8, "dpd": 0.05, "loss": 0.2} for _ in range(2)]

    agg, info = aggregate(
        "guardfed_style", updates, meta,
        g_s=g_s, gs_fair=torch.tensor([0.0, 0.0]), gs_gamma=0.0
    )

    assert info["selected"] == [0]
    assert info["dev_scores"][0] > 0.0
    assert info["dev_scores"][1] == 0.0


def test_guardfed_holdout_e2e():
    """End-to-end check of compute_guardfed_holdout with GNN model on German holdout."""
    data = load_dataset("german", root="data", seed=42)
    holdout, _ = carve_server_holdout(data, size=100, seed=42)
    cfg = ExperimentConfig.canonical(dataset="german", model="trustfedgnn", seed=42)
    model = build_model("trustfedgnn", data.x.shape[1], cfg).cpu()
    global_flat = flatten_state(model.state_dict()).cpu()

    K = 3
    updates = [torch.randn_like(global_flat) * 0.01 for _ in range(K)]
    g_s, gs_fair = compute_guardfed_holdout(model, global_flat, updates, holdout, "cpu")

    assert g_s.shape == global_flat.shape
    assert not torch.isnan(g_s).any()
    assert gs_fair.shape == (K,)
    assert not torch.isnan(gs_fair).any()
    assert (gs_fair >= 0.0).all() and (gs_fair <= 1.0).all()
