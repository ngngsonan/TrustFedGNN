"""Gates for the campaigns cited by the article: registry integrity, the C20r
falsified-report protocol, the C24 active-round boundary and the S3 sample-count
correction."""
import inspect
import os

import torch

from experiments.campaigns.analyze_discover_ai import C24_INVALID_CELLS, active_poison_round
from experiments.campaigns.c20_fixed_state_replay import union_lie
from experiments.campaigns.c24_adaptive_access import AdaptiveAccessTrainer
from experiments.campaigns.campaigns import CAMPAIGNS, OUT
from src.federated.aggregation import aggregate
from src.federated.attacks import BEST_RESPONSE_LIE

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def test_registry_ids_unique_and_scripts_exist():
    ids = [c.id for c in CAMPAIGNS]
    assert len(ids) == len(set(ids))
    for c in CAMPAIGNS:
        for cmd in c.cmds + c.smoke_cmds:
            assert cmd[0] == "python"
            assert os.path.exists(os.path.join(REPO, cmd[1])), cmd[1]
        for p in c.outputs:
            assert p.startswith(OUT + "/"), p


def test_union_lie_changes_every_report_field_but_n():
    metas = [{"n": 40, "perf": 0.6, "dpd": 0.3, "eod": 0.2, "eo": 0.2, "loss": 0.4, "loss_pre": 0.7,
              "group1_rate": 0.4},
             {"n": 55, "perf": 0.7, "dpd": 0.1, "eod": 0.1, "eo": 0.1, "loss": 0.3, "loss_pre": 0.6,
              "group1_rate": 0.5}]
    lie = union_lie(metas, {0})
    assert "n" not in lie
    lied = {**metas[0], **lie}
    assert lied != metas[0]
    for key in ("perf", "dpd", "eod", "loss", "loss_pre"):
        assert lied[key] != metas[0][key], key
    assert lie["loss_pre"] == 2 * 0.6


def test_c20r_gives_every_report_free_control_a_falsified_report():
    controls = {"cgsv", "fltrust", "fltrust_root", "fu_shapley", "fu_shapley_alpha0"}
    assert not controls & set(BEST_RESPONSE_LIE), "controls would take the best-response branch"
    src = inspect.getsource(__import__("experiments.campaigns.c20_fixed_state_replay",
                                       fromlist=["run_rule_replay"]).run_rule_replay)
    assert "union_lie(" in src and "set_seed(seed)" in src
    assert 'metas_lying[0] != metas_honest[0]' in src


def test_c24_active_round_boundary():
    # Stored rounds are 1-indexed; the trainer's schedule uses t = stored - 1.
    assert not active_poison_round(10, "S1") and active_poison_round(11, "S1")
    assert active_poison_round(15, "S2") and not active_poison_round(16, "S2")
    assert active_poison_round(21, "S2") and not active_poison_round(20, "S2")
    assert not active_poison_round(1, "S3") and active_poison_round(2, "S3")
    trainer = AdaptiveAccessTrainer.__new__(AdaptiveAccessTrainer)
    trainer.byzantine_ids = [0]
    for sch in ("S1", "S2", "S3"):
        trainer.schedule = sch
        for t in range(40):
            assert trainer._is_poison_round(t) == active_poison_round(t + 1, sch), (sch, t)


def test_s3_report_carries_the_true_sample_count():
    src = inspect.getsource(AdaptiveAccessTrainer._round)
    assert '"n": m["n"]' in src
    assert ("S3", "fedavg") in C24_INVALID_CELLS


def _weights(method, metas, **kw):
    torch.manual_seed(0)
    updates = [torch.randn(50) for _ in metas]
    _, info = aggregate(method, updates, metas, state={}, **kw)
    return info["weights"]


def test_c24_methods_other_than_fedavg_ignore_the_sample_count():
    torch.manual_seed(1)
    ref, g_s = torch.randn(50), torch.randn(50)
    base = [{"n": 100, "perf": 0.8, "dpd": 0.1, "loss": 0.2, "group1_rate": 0.5} for _ in range(4)]
    alt = [dict(m, n=n) for m, n in zip(base, (1, 7, 300, 42))]
    cases = {
        "fltrust_root": dict(delta_root=ref),
        "fu_shapley": dict(g_target=ref, g_task=ref, g_fair=torch.zeros(50), fu_score="cosine",
                           fu_grad_clip=10.0),
        "guardfed_style": dict(g_s=g_s, gs_fair=torch.zeros(4), gs_tau=1.0, gs_gamma=0.0),
    }
    for method, kw in cases.items():
        assert _weights(method, base, **kw) == _weights(method, alt, **kw), method
    assert _weights("fedavg", base) != _weights("fedavg", alt)
