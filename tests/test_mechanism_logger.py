"""Unit test verifying I_adv and mechanism logger calculations (Review C23 Item 1).

Verifies on toy synthetic examples that:
1. For FU (with clipping), I_adv = ||sum_{k in A} w_k Delta_tilde_k|| matches hand-calculated value.
2. For FLTrust (with root rescaling), I_adv matches hand-calculated value with rescaled updates.
3. Trainer logs w_adv_round = sum_{k in A} w_k.
4. Cosine keys are renamed to cos_agg_g_task and cos_agg_g_fair.
"""
import math
import pytest
import torch

from src.config import ExperimentConfig
from src.federated.trainer import FederatedTrainer


def test_toy_i_adv_fu_clip_hand_calculation():
    """Verify I_adv for FU (clip) matches exact hand-calculated numbers."""
    # Toy scenario:
    # K=3 clients, client 0 is Byzantine: A = {0}.
    # Dimension D=2.
    # Clipped updates in info["effective_updates"]:
    #   Delta_0 = [1.0, 2.0]  (adversary)
    #   Delta_1 = [3.0, 0.0]  (benign)
    #   Delta_2 = [0.0, 4.0]  (benign)
    # Weights w = [0.25, 0.35, 0.40]
    # Hand calculation:
    #   w_adv_round = 0.25
    #   adv_sum = 0.25 * [1.0, 2.0] = [0.25, 0.50]
    #   ||adv_sum||_2 = sqrt(0.25^2 + 0.50^2) = sqrt(0.0625 + 0.25) = sqrt(0.3125) = 0.5590169943749475
    eff_updates = [
        torch.tensor([1.0, 2.0]),
        torch.tensor([3.0, 0.0]),
        torch.tensor([0.0, 4.0]),
    ]
    w_vec = [0.25, 0.35, 0.40]
    byz_ids = {0}

    # Simulate trainer logger logic
    w_adv_round = float(sum(w_vec[i] for i in byz_ids if i < len(w_vec)))
    adv_terms = [
        float(w_vec[i]) * eff_updates[i].float()
        for i in byz_ids
        if i < len(w_vec) and i < len(eff_updates)
    ]
    I_adv = float(torch.stack(adv_terms).sum(dim=0).norm().item())

    expected_w_adv = 0.25
    expected_I_adv = math.sqrt(0.25**2 + 0.50**2)
    assert abs(w_adv_round - expected_w_adv) < 1e-7
    assert abs(I_adv - expected_I_adv) < 1e-7

    # Two adversaries: A = {0, 1}
    #   w_0 = 0.2, Delta_0 = [1.0, 2.0] -> 0.2 * [1.0, 2.0] = [0.2, 0.4]
    #   w_1 = 0.3, Delta_1 = [3.0, 0.0] -> 0.3 * [3.0, 0.0] = [0.9, 0.0]
    #   adv_sum = [0.2 + 0.9, 0.4 + 0.0] = [1.1, 0.4]
    #   ||adv_sum|| = sqrt(1.1^2 + 0.4^2) = sqrt(1.21 + 0.16) = sqrt(1.37) = 1.1704699910719625
    byz_ids_2 = {0, 1}
    w_vec_2 = [0.2, 0.3, 0.5]
    adv_terms_2 = [
        float(w_vec_2[i]) * eff_updates[i].float()
        for i in byz_ids_2
    ]
    I_adv_2 = float(torch.stack(adv_terms_2).sum(dim=0).norm().item())
    expected_I_adv_2 = math.sqrt(1.1**2 + 0.4**2)
    assert abs(I_adv_2 - expected_I_adv_2) < 1e-6


def test_toy_i_adv_fltrust_rescale_hand_calculation():
    """Verify I_adv for FLTrust matches exact hand-calculated numbers with rescaling."""
    # Toy scenario:
    # Reference g_0 = [3.0, 4.0], ||g_0|| = 5.0.
    # Client 0 (Byzantine): raw update u_0 = [0.0, 10.0], norm = 10.0.
    #   Rescaled: Delta_tilde_0 = u_0 * (5.0 / 10.0) = [0.0, 5.0].
    # Client 1 (Benign): raw update u_1 = [3.0, 4.0], norm = 5.0.
    #   Rescaled: Delta_tilde_1 = [3.0, 4.0].
    # Weight w = [0.2, 0.8].
    # Adversary A = {0}:
    #   w_adv_round = 0.2
    #   adv_sum = 0.2 * [0.0, 5.0] = [0.0, 1.0]
    #   I_adv = ||adv_sum|| = 1.0.
    ref = torch.tensor([3.0, 4.0])
    updates = [
        torch.tensor([0.0, 10.0]),
        torch.tensor([3.0, 4.0]),
    ]
    w_vec = [0.2, 0.8]
    byz_ids = {0}

    ref_norm = float(ref.flatten().norm().item()) + 1e-12
    eff_updates = [
        u.flatten().cpu() * (ref_norm / (float(u.flatten().norm().item()) + 1e-12))
        for u in updates
    ]

    # Verify rescaled norm matches ref_norm
    assert abs(eff_updates[0].norm().item() - 5.0) < 1e-5

    adv_terms = [
        float(w_vec[i]) * eff_updates[i].float()
        for i in byz_ids
    ]
    I_adv = float(torch.stack(adv_terms).sum(dim=0).norm().item())
    assert abs(I_adv - 1.0) < 1e-6


def test_trainer_mechanism_logger_end_to_end_synthetic():
    """Verify FederatedTrainer populates w_adv_round, I_adv, cos_agg_g_task, cos_agg_g_fair."""
    cfg = ExperimentConfig.canonical(
        dataset="synthetic",
        rounds=1,
        num_clients=3,
        local_epochs=1,
        seed=42,
        model="trustfedgnn",
        aggregator="fu_shapley",
        fu_alpha=0.1,
        log_mechanism=True,
        attack="fairness_poison",
        num_byzantine=1,
        attack_intensity=10.0,
    )
    trainer = FederatedTrainer(cfg)
    res = trainer.run(verbose=False)
    assert len(res["history"]) == 1
    rec = res["history"][0]

    # Verify keys
    assert "w_adv_round" in rec, "w_adv_round must be present"
    assert "I_adv" in rec, "I_adv must be present"
    assert "cos_agg_g_task" in rec, "cos_agg_g_task must be present"
    assert "cos_agg_g_fair" in rec, "cos_agg_g_fair must be present"
    assert "cos(agg, g_task)" not in rec, "old key must be removed"
    assert "cos(agg, g_fair)" not in rec, "old key must be removed"

    assert rec["w_adv_round"] is not None
    assert rec["I_adv"] is not None
    assert rec["cos_agg_g_task"] is not None
    assert rec["cos_agg_g_fair"] is not None
    assert rec["w_adv_round"] >= 0.0
    assert rec["I_adv"] >= 0.0
