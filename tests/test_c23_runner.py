"""Unit tests for C23 Mechanism Factorial runner (K13).
"""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from experiments.campaigns.c23_mechanism_factorial import (
    ARMS,
    SCENARIO_ARMS,
    SCENARIOS,
    run_one,
)


def test_c23_arms_defined():
    expected_arms = [
        "F_SR_C", "F_SR_R", "F_RS_C", "F_RS_R",
        "A4_fu_shapley", "A1_fltrust", "A5_fltrust_ema", "R1_fltrust_root", "A0_fedavg",
    ]
    for arm in expected_arms:
        assert arm in ARMS, f"Missing arm {arm}"

    assert len(SCENARIO_ARMS["fairness_poison"]) == 9
    assert set(SCENARIO_ARMS["clean"]) == {"A0_fedavg", "A4_fu_shapley", "R1_fltrust_root"}


def test_c23_run_one_smoke():
    res = run_one(
        dataset="german",
        arm_name="F_SR_R",
        arm_kwargs=ARMS["F_SR_R"],
        scenario_name="fairness_poison",
        attack_kwargs=SCENARIOS["fairness_poison"],
        seed=42,
        device="cpu",
    )
    assert res["dataset"] == "german"
    assert res["arm"] == "F_SR_R"
    assert 0.0 <= res["auc"] <= 1.0
    assert 0.0 <= res["dpd_hard"] <= 1.0
    assert 0.0 <= res["w_adv"] <= 1.0
    assert len(res["per_round"]) == 20
    assert "agg_norm" in res["per_round"][0]
    assert "t_server_s" in res["per_round"][0]
