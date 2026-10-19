"""Unit tests for C22 GuardFed Matched runner (K13).
"""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from experiments.campaigns.c22_guardfed_matched import (
    GRID,
    GAMMA_VALUES,
    TAU_VALUES,
    run_part_a_one,
    run_part_b_one,
    run_fltrust_root_part_b_one,
    pick_best_config,
    check_c05_gating,
)


def test_c22_grid_structure():
    assert len(GRID) == 14
    for cfg in GRID:
        assert "gs_tau" in cfg and "gs_gamma" in cfg and "name" in cfg
        assert cfg["gs_tau"] in TAU_VALUES
        assert cfg["gs_gamma"] in GAMMA_VALUES


def test_c22_pick_best_config_logic():
    # Synthetic records: config_a has higher score, config_b has lower score
    records = {
        "cfg_a": [{"val_auc": 0.80, "val_dpd_hard": 0.05, "w_adv": 0.20}],  # score = 0.75
        "cfg_b": [{"val_auc": 0.70, "val_dpd_hard": 0.05, "w_adv": 0.10}],  # score = 0.65
    }
    best, summ = pick_best_config(records)
    assert best == "cfg_a"
    assert summ["cfg_a"]["score"] == 0.75

    # Tie-break test: same score, pick lower w_adv
    records_tie = {
        "cfg_high_w": [{"val_auc": 0.80, "val_dpd_hard": 0.05, "w_adv": 0.30}],  # score = 0.75
        "cfg_low_w":  [{"val_auc": 0.80, "val_dpd_hard": 0.05, "w_adv": 0.10}],  # score = 0.75
    }
    best_tie, summ_tie = pick_best_config(records_tie)
    assert best_tie == "cfg_low_w"


def test_c22_c05_gating_passes():
    """Verify C05 Part B gating reproduces bit-exact."""
    check_c05_gating()


def test_c22_run_part_a_smoke():
    res = run_part_a_one("german", "gs_tau1_gam0", {"gs_tau": 1.0, "gs_gamma": 0.0}, seed=1000, device="cpu")
    assert res["dataset"] == "german"
    assert 0.0 <= res["auc"] <= 1.0
    assert 0.0 <= res["val_auc"] <= 1.0
    assert 0.0 <= res["w_adv"] <= 1.0
    assert 0.0 <= res["empty_s_frac"] <= 1.0
