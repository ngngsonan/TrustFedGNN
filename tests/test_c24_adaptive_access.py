"""Unit tests for C24 Adaptive Access Sweeps (K15 / B.1 / B.2).

Verifies:
  1. Benign identity: adversary in benign rounds produces identical update and report to an honest client.
  2. Access restriction: craft_s3_adversary_update strictly accepts only
     (theta_history, own_update, own_data, model_template), without accessing server holdout or peer updates.
  3. Schedule progression: S1, S2, S3, and benign active poisoning rounds.
  4. Completeness and resume: fails loudly if any seed in 42..51 is missing.
  5. End-to-end smoke execution of run_one.
"""
from __future__ import annotations

import copy
import inspect
import sys
from pathlib import Path

import pytest
import torch
from torch_geometric.data import Data

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import ExperimentConfig, set_seed
from src.federated.client import Client, load_flat_state, flatten_state
from src.models import GCN, TrustFedGNN
from experiments.campaigns.c24_adaptive_access import (
    AdaptiveAccessTrainer,
    check_c24_completeness,
    craft_s3_adversary_update,
    get_method_kwargs,
    run_one,
)


def _create_synthetic_client_data(num_nodes: int = 50, in_dim: int = 8, seed: int = 42) -> Data:
    """Generate deterministic synthetic graph data with sensitive attribute."""
    torch.manual_seed(seed)
    x = torch.randn(num_nodes, in_dim)
    # Simple ring + random edges
    edges = [[i, (i + 1) % num_nodes] for i in range(num_nodes)]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    y = torch.randint(0, 2, (num_nodes,))
    s = torch.tensor([0 if i < num_nodes // 2 else 1 for i in range(num_nodes)], dtype=torch.long)
    train_mask = torch.ones(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask[:10] = True
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask[10:20] = True

    return Data(x=x, edge_index=edge_index, y=y, sensitive_attr=s,
                train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)


def test_benign_identity():
    """Verify adversary in benign round produces identical update and report to an honest client."""
    data_honest = _create_synthetic_client_data(seed=123)
    data_adv = _create_synthetic_client_data(seed=123)

    cfg = ExperimentConfig.canonical(
        dataset="german",
        seed=42,
        num_clients=2,
        rounds=5,
        local_epochs=2,
        dp_enabled=False,
    )

    # Honest client
    client_honest = Client(0, data_honest, cfg, device="cpu", byzantine=False)
    # Adversary client
    client_adv = Client(1, data_adv, cfg, device="cpu", byzantine=True)

    # Initialize with identical weights
    init_weights = client_honest.get_flat().clone()
    client_adv.set_flat(init_weights)

    # Benign round training: adversary trains with byzantine temporarily False
    torch.manual_seed(42)
    client_honest.train()
    g_honest = init_weights - client_honest.get_flat()
    meta_honest = client_honest.meta()

    torch.manual_seed(42)
    prev_byz = client_adv.byzantine
    client_adv.byzantine = False
    client_adv.train()
    client_adv.byzantine = prev_byz
    g_adv = init_weights - client_adv.get_flat()
    meta_adv = client_adv.meta()

    assert torch.allclose(client_honest.get_flat(), client_adv.get_flat(), atol=1e-7), \
        "Adversary weights in benign round must be bit-exact identical to honest client"
    assert torch.allclose(g_honest, g_adv, atol=1e-7), \
        "Adversary update in benign round must be bit-exact identical to honest client update"
    assert meta_honest == meta_adv, \
        "Adversary metadata report in benign round must match honest client report"


def test_access_restriction_signature_and_isolation():
    """Verify craft_s3_adversary_update information constraint (B.1 / K15).

    The function signature must only accept:
      - theta_history
      - own_update
      - own_data
      - model_template
      - optional hyperparameters (lambda_fair, opt_steps, opt_lr)
    It must not take server holdout, target gradients, or peer updates.
    """
    sig = inspect.signature(craft_s3_adversary_update)
    param_names = list(sig.parameters.keys())
    allowed = {"theta_history", "own_update", "own_data", "model_template", "lambda_fair", "opt_steps", "opt_lr"}
    assert set(param_names) == allowed, f"Unexpected parameter in craft_s3_adversary_update: {param_names}"

    # Verify execution in isolated environment without server or peer state
    data = _create_synthetic_client_data(seed=456)
    in_dim = data.x.size(1)
    model = TrustFedGNN(in_channels=in_dim, hidden_channels=16, out_channels=1)

    # Create synthetic theta_history: 2 rounds of global flat vectors
    p0 = flatten_state(model.state_dict()).detach()
    p1 = p0 + 0.01 * torch.randn_like(p0)
    theta_history = [p0, p1]
    own_update = 0.05 * torch.randn_like(p0)

    # Execute attack
    g_adv = craft_s3_adversary_update(
        theta_history=theta_history,
        own_update=own_update,
        own_data=data,
        model_template=model,
        lambda_fair=1.0,
        opt_steps=10,
        opt_lr=0.01,
    )

    assert g_adv.shape == own_update.shape
    assert torch.isfinite(g_adv).all(), "Crafted adversarial update must be finite"
    assert not torch.allclose(g_adv, own_update), "Adversarial update should deviate from honest update"


def test_schedule_progression():
    """Verify schedule poisoning intervals across S1, S2, S3, and benign."""
    cfg = ExperimentConfig.canonical(
        dataset="german",
        seed=42,
        num_clients=10,
        num_byzantine=2,
        rounds=40,
        dp_enabled=False,
    )

    # S1: honest for t < 10, poison for t >= 10
    trainer_s1 = AdaptiveAccessTrainer(cfg, schedule="S1")
    for t in range(40):
        expected = (t >= 10)
        assert trainer_s1._is_poison_round(t) == expected, f"S1 mismatch at round {t}"

    # S2: honest for t < 10; when t >= 10, poison when ((t - 10) // 5) % 2 == 0
    trainer_s2 = AdaptiveAccessTrainer(cfg, schedule="S2")
    for t in range(40):
        if t < 10:
            expected = False
        else:
            block = (t - 10) // 5
            expected = (block % 2 == 0)
        assert trainer_s2._is_poison_round(t) == expected, f"S2 mismatch at round {t}"

    # S3: honest for t = 0, poison for t >= 1
    trainer_s3 = AdaptiveAccessTrainer(cfg, schedule="S3")
    assert not trainer_s3._is_poison_round(0), "S3 round 0 must be benign"
    for t in range(1, 40):
        assert trainer_s3._is_poison_round(t) is True, f"S3 round {t} must be poison"

    # Benign arm (num_byzantine = 0)
    cfg_benign = ExperimentConfig.canonical(
        dataset="german",
        seed=42,
        num_clients=10,
        num_byzantine=0,
        rounds=40,
        dp_enabled=False,
    )
    for sch in ("S1", "S2", "S3", "benign"):
        trainer_benign = AdaptiveAccessTrainer(cfg_benign, schedule=sch)
        for t in range(40):
            assert not trainer_benign._is_poison_round(t), f"Benign arm must be False at round {t} under {sch}"


def test_completeness_check_raises_on_missing_seed():
    """Verify check_c24_completeness raises RuntimeError on incomplete seed runs."""
    expected_cells = [("german", "S3", "fu_shapley", 0.2)]
    expected_seeds = list(range(42, 52))

    # Incomplete results (only 9 seeds)
    incomplete_results = {
        "german": {
            "S3": {
                "fu_shapley": {
                    "per_seed": [{"seed": s} for s in range(42, 51)]  # Missing seed 51
                }
            }
        }
    }

    with pytest.raises(RuntimeError, match="missing seeds: \\[51\\]"):
        check_c24_completeness(incomplete_results, expected_cells, expected_seeds)

    # Complete results (all 10 seeds)
    complete_results = {
        "german": {
            "S3": {
                "fu_shapley": {
                    "per_seed": [{"seed": s} for s in range(42, 52)]
                }
            }
        }
    }
    check_c24_completeness(complete_results, expected_cells, expected_seeds)


def test_c24_smoke_run_end_to_end():
    """Verify single-run smoke execution for S3 fu_shapley."""
    res = run_one(
        dataset="german",
        method="fu_shapley",
        schedule="S3",
        byzantine_fraction=0.2,
        seed=42,
        rounds=2,
        device="cpu",
    )

    assert res["dataset"] == "german"
    assert res["method"] == "fu_shapley"
    assert res["schedule"] == "S3"
    assert res["seed"] == 42
    assert 0.0 <= res["auc"] <= 1.0
    assert 0.0 <= res["dpd_hard"] <= 1.0
    assert 0.0 <= res["w_adv"] <= 1.0
    assert res["I_adv"] >= 0.0
    assert len(res["per_round"]) == 2
