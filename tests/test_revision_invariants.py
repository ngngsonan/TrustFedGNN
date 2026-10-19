"""Invariant tests for Phase 1 revision ablation configs and runners.

Ensures that:
1. Every ablation arm in the grid isolates EXACTLY the intended components.
2. No baseline or ablation arm borrows outside modules inappropriately.
3. All 7 configs instantiate and run correctly for 1 round on synthetic data.
"""
from __future__ import annotations

import os
import re
import sys
import pytest

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import ExperimentConfig
from src.federated import FederatedTrainer


ABLATION_CONFIGS = {
    "C0_FedAvg": dict(
        model="gcn",
        aggregator="fedavg",
        local_fairness=False,
        dp_enabled=False,
    ),
    "C1_FedAvg_FSER": dict(
        model="trustfedgnn",
        aggregator="fedavg",
        local_fairness=False,
        dp_enabled=False,
    ),
    "C2_FedAvg_FTGD": dict(
        model="gcn",
        aggregator="fedavg",
        local_fairness=True,
        dp_enabled=False,
        dp_mode="ftgd",
    ),
    "C3_FedAvg_FTGD_DP": dict(
        model="gcn",
        aggregator="fedavg",
        local_fairness=True,
        dp_enabled=True,
        dp_mode="ftgd",
        dp_epsilon=8.0,
        dp_delta=1e-5,
    ),
    "C4_BFWA_unconstrained": dict(
        model="gcn",
        aggregator="bfwa",
        fairness_budget=1e6,
        local_fairness=False,
        dp_enabled=False,
    ),
    "C5_BFWA_constrained": dict(
        model="gcn",
        aggregator="bfwa",
        fairness_budget=0.05,
        local_fairness=False,
        dp_enabled=False,
    ),
    "C6_Full_TrustFedGNN": dict(
        model="trustfedgnn",
        aggregator="bfwa",
        fairness_budget=0.05,
        dp_enabled=True,
        dp_mode="ftgd",
        dp_epsilon=8.0,
        dp_delta=1e-5,
    ),
}


def test_ablation_grid_pairwise_isolation():
    """Verify that ablation arms strictly isolate the intended independent variables."""
    c0 = ABLATION_CONFIGS["C0_FedAvg"]
    c1 = ABLATION_CONFIGS["C1_FedAvg_FSER"]
    c2 = ABLATION_CONFIGS["C2_FedAvg_FTGD"]
    c3 = ABLATION_CONFIGS["C3_FedAvg_FTGD_DP"]
    c4 = ABLATION_CONFIGS["C4_BFWA_unconstrained"]
    c5 = ABLATION_CONFIGS["C5_BFWA_constrained"]
    c6 = ABLATION_CONFIGS["C6_Full_TrustFedGNN"]

    # C0 vs C1: strictly isolates FSER (model)
    diff_0_1 = {k for k in set(c0) | set(c1) if c0.get(k) != c1.get(k)}
    assert diff_0_1 == {"model"}, f"C0 vs C1 should strictly differ by 'model', got {diff_0_1}"

    # C0 vs C2: strictly isolates FTGD loss objective
    diff_0_2 = {k for k in set(c0) | set(c2) if c0.get(k) != c2.get(k)}
    assert diff_0_2 == {"local_fairness", "dp_mode"}, f"C0 vs C2 diff: {diff_0_2}"

    # C2 vs C3: strictly isolates DP noise
    diff_2_3 = {k for k in set(c2) | set(c3) if c2.get(k) != c3.get(k)}
    assert diff_2_3 == {"dp_enabled", "dp_epsilon", "dp_delta"}, f"C2 vs C3 diff: {diff_2_3}"

    # C0 vs C4: strictly isolates BFWA unconstrained aggregation
    diff_0_4 = {k for k in set(c0) | set(c4) if c0.get(k) != c4.get(k)}
    assert diff_0_4 == {"aggregator", "fairness_budget"}, f"C0 vs C4 diff: {diff_0_4}"

    # C4 vs C5: strictly isolates disparity budget constraint tau
    diff_4_5 = {k for k in set(c4) | set(c5) if c4.get(k) != c5.get(k)}
    assert diff_4_5 == {"fairness_budget"}, f"C4 vs C5 diff: {diff_4_5}"

    # C6: Full TrustFedGNN has all three components active
    assert c6["model"] == "trustfedgnn"
    assert c6["aggregator"] == "bfwa"
    assert c6["fairness_budget"] == 0.05
    assert c6["dp_enabled"] is True
    assert c6["dp_mode"] == "ftgd"


def test_ablation_configs_smoke_run():
    """Smoke test: execute 1 round of each of the 7 configs on synthetic data."""
    for name, overrides in ABLATION_CONFIGS.items():
        cfg = ExperimentConfig.canonical(
            dataset="synthetic",
            rounds=1,
            num_clients=2,
            local_epochs=1,
            seed=42,
            **overrides
        )
        trainer = FederatedTrainer(cfg)
        res = trainer.run(verbose=False)
        assert "final" in res
        assert "auc" in res["final"]
        assert "dpd_hard" in res["final"]
        assert "eod" in res["final"]


def test_robustness_screening_invariants():
    """Verify that robust_bfwa successfully performs distance screening under attack."""
    cfg = ExperimentConfig.canonical(
        dataset="synthetic",
        rounds=2,
        num_clients=4,
        local_epochs=1,
        seed=42,
        model="trustfedgnn",
        aggregator="robust_bfwa",
        attack="gaussian",
        num_byzantine=1,
        krum_f=1,
        attack_intensity=10.0,
    )
    trainer = FederatedTrainer(cfg)
    res = trainer.run(verbose=False)
    assert len(res["history"]) == 2
    # Byzantine client is client 0. Check that client 0 is screened out from kept set
    for r_entry in res["history"]:
        assert "kept" in r_entry, f"r_entry should have 'kept' key, got keys: {list(r_entry.keys())}"
        assert len(r_entry["kept"]) == 3
        # Malicious client 0 with gaussian noise variance 10.0 should be excluded
        assert 0 not in r_entry["kept"], f"Malicious client 0 should be screened out, but kept set is {r_entry['kept']}"



# --------------------------------------------------------------------------- #
# BFWA: the fairness budget tau must actually bind, and the dual multiplier
# must survive across communication rounds.
#
# Both were broken. `gamma = 2/(t+2)` evaluated at t=0 is 1.0, so the first
# Frank-Wolfe step discarded the uniform iterate and jumped onto the vertex
# argmin(-perf + mu*dpd); with mu=0 and a gradient that does not depend on w
# (the objective is linear), every later step re-selected that same vertex.
# `mu` was also a local restarted at 0 on every call, so it never reached the
# value at which the vertex flips. Net effect: sweeping tau over
# {1e6, 0.10, 0.05, 0.02, 0.0} at the shipped fw_iters=20 returned the
# IDENTICAL weight vector every time -- tau had no effect anywhere in the repo.
# --------------------------------------------------------------------------- #
import torch

from src.federated.aggregation import aggregate, bfwa_weights

# A client set with a real utility/fairness trade-off: the most accurate client
# (0) is also the most unfair, the second-most accurate (2) is middling, and the
# fairest clients (1, 3) are the least accurate.
_PERF = torch.tensor([0.90, 0.75, 0.85, 0.70, 0.80])
_DPD = torch.tensor([0.20, 0.02, 0.15, 0.01, 0.10])
_TAUS = [1e6, 0.10, 0.05, 0.02, 0.0]


def test_bfwa_tau_actually_changes_the_weights():
    """Different tau -> different weights at the shipped iteration budget."""
    ws = [bfwa_weights(_PERF, _DPD, tau, iters=20, dual_step=0.1) for tau in _TAUS]
    for i in range(len(ws) - 1):
        assert not torch.allclose(ws[i], ws[i + 1], atol=1e-6), (
            f"tau={_TAUS[i]} and tau={_TAUS[i+1]} gave identical weights "
            f"{ws[i].tolist()} -- the fairness budget is not binding")
    # ... and tightening the budget must move the achieved disparity DOWN.
    gaps = [float(w.dot(_DPD)) for w in ws]
    assert all(gaps[i] >= gaps[i + 1] - 1e-9 for i in range(len(gaps) - 1)), \
        f"weighted DPD must be non-increasing as tau tightens, got {gaps}"
    assert gaps[0] > gaps[-1] + 1e-3, \
        f"tau=0 must buy a strictly smaller disparity than tau=1e6, got {gaps}"


def test_bfwa_reports_constraint_residual_and_feasibility():
    """The per-round residual/feasibility the manuscript promises to report."""
    updates = [torch.randn(8) for _ in range(5)]
    meta = [{"n": 100, "perf": float(p), "dpd": float(d), "loss": 1.0 - float(p)}
            for p, d in zip(_PERF, _DPD)]
    for method in ("bfwa", "robust_bfwa"):
        _, info = aggregate(method, updates, meta, tau=0.05, fw_iters=20,
                            dual_step=0.1, krum_f=1, state={})
        for key in ("constraint_residual", "feasible",
                    "constraint_residual_preclamp", "feasible_preclamp"):
            assert key in info, f"{method} must report {key}"
        w = torch.tensor(info["weights"])
        # post-clamp residual is computed on the returned weights
        assert abs(info["constraint_residual"] - (float(w.dot(_DPD)) - 0.05)) < 1e-5
        assert info["feasible"] == (info["constraint_residual"] <= 0.0)
        assert info["feasible_preclamp"] == (info["constraint_residual_preclamp"] <= 0.0)

    # A budget nothing can violate is reported feasible; tau=0 with a strictly
    # positive disparity everywhere cannot be.
    _, loose = aggregate("bfwa", updates, meta, tau=1e6, state={})
    _, tight = aggregate("bfwa", updates, meta, tau=0.0, state={})
    assert loose["feasible"] is True and tight["feasible"] is False


def test_bfwa_dual_persists_across_rounds():
    """mu must accumulate across rounds through `state`, like fedgraphfair_lambda."""
    updates = [torch.randn(8) for _ in range(5)]
    meta = [{"n": 100, "perf": float(p), "dpd": float(d), "loss": 1.0 - float(p)}
            for p, d in zip(_PERF, _DPD)]
    kw = dict(tau=0.02, fw_iters=20, dual_step=0.1)

    state = {}
    persisted = [aggregate("bfwa", updates, meta, state=state, **kw)[1]
                 for _ in range(2)]
    assert "bfwa_mu" in state and state["bfwa_mu"] > 0.0
    assert persisted[1]["bfwa_mu"] > persisted[0]["bfwa_mu"], \
        "the dual multiplier must keep ascending across rounds"

    independent = [aggregate("bfwa", updates, meta, state=None, **kw)[1]
                   for _ in range(2)]
    assert independent[0]["weights"] == independent[1]["weights"], \
        "without state, every round restarts the dual from 0 (sanity)"
    assert persisted[1]["weights"] != independent[1]["weights"], \
        "persisting mu must change round 2's weights -- otherwise it does nothing"
    # Carrying the dual over tightens the constraint, it does not loosen it.
    assert persisted[1]["constraint_residual"] < independent[1]["constraint_residual"]

    # bfwa_persist_dual=False reproduces the old reset-every-round behaviour.
    off_state = {}
    off = [aggregate("bfwa", updates, meta, state=off_state,
                     bfwa_persist_dual=False, **kw)[1] for _ in range(2)]
    assert "bfwa_mu" not in off_state
    assert off[0]["weights"] == off[1]["weights"] == independent[1]["weights"]


def test_bfwa_and_robust_bfwa_keep_separate_duals():
    """The two rules solve different subproblems and must not share mu."""
    updates = [torch.randn(8) for _ in range(5)]
    meta = [{"n": 100, "perf": float(p), "dpd": float(d), "loss": 1.0 - float(p)}
            for p, d in zip(_PERF, _DPD)]
    state = {}
    aggregate("bfwa", updates, meta, tau=0.02, state=state)
    aggregate("robust_bfwa", updates, meta, tau=0.02, krum_f=1, state=state)
    assert {"bfwa_mu", "robust_bfwa_mu"} <= set(state)
    assert state["bfwa_mu"] != state["robust_bfwa_mu"]


def test_slack_cli_defaults_match_function_defaults():
    """A no-flag run must measure what the function signature says it measures.

    bfwa_slack_analysis exposes the same knobs twice: once as defaults on
    analyze_bfwa_slack, once as argparse defaults in main(). A revision that
    corrected the first and left the second stale produced a run that reported
    Bail on two seeds while the caller believed it was German on ten -- a
    complete, plausible-looking table of the wrong experiment, with nothing in
    the artifact to reveal it. This locks the two together.
    """
    import argparse
    import inspect
    from unittest import mock

    import experiments.revision.bfwa_slack_analysis as mod

    sig = inspect.signature(mod.analyze_bfwa_slack)
    fn_defaults = {k: v.default for k, v in sig.parameters.items()
                   if v.default is not inspect.Parameter.empty}

    captured = {}
    real_parse = argparse.ArgumentParser.parse_args

    def capture(self, *a, **kw):
        ns = real_parse(self, [])                      # defaults only
        captured.update(vars(ns))
        raise SystemExit(0)                            # stop before running

    with mock.patch.object(argparse.ArgumentParser, "parse_args", capture), \
            mock.patch.object(mod, "run_bfwa_slack_experiment", lambda **kw: None):
        try:
            mod.main()
        except SystemExit:
            pass

    assert captured, "main() did not reach parse_args"
    for name in ("dataset", "rounds", "tau"):
        assert captured[name] == fn_defaults[name], (
            f"CLI default for --{name} is {captured[name]!r} but "
            f"analyze_bfwa_slack defaults to {fn_defaults[name]!r}")
    assert tuple(captured["seeds"]) == tuple(fn_defaults["seeds"])
    assert tuple(captured["epsilons"]) == tuple(fn_defaults["epsilons"])
    assert captured["num_clients"] == fn_defaults["num_clients"]


def _capture_weights(rule, lie):
    """Weights under an honest report and under ``lie``, same updates and seed."""
    import copy
    from src.federated.aggregation import aggregate
    torch.manual_seed(0)
    ups = [torch.randn(32) for _ in range(5)]
    gt = torch.randn(32)
    honest = [{"n": 100, "perf": 0.70 + 0.02 * i, "dpd": 0.10 + 0.03 * i, "eod": 0.08,
               "loss": 0.30 - 0.02 * i, "group1_rate": 0.3 + 0.1 * i} for i in range(5)]
    lying = copy.deepcopy(honest)
    lying[0].update(lie(lying, {0}))
    kw = dict(g_target=gt, tau=0.05)
    wh = aggregate(rule, ups, copy.deepcopy(honest), state={}, **kw)[1].get("weights")
    wl = aggregate(rule, ups, lying, state={}, **kw)[1].get("weights")
    return wh, wl


def test_metadata_reading_rules_move_under_a_false_report():
    """Reading a self-reported field means the weights are a function of it.

    A null result here would otherwise be unfalsifiable: before this was added,
    ``loss`` was overwritten with the honest value on every call, so qffl and
    fedgraphfair could not be attacked at all and would have been reported as
    immune -- an artefact of the harness, not a property of the rules.
    """
    from src.federated.attacks import BEST_RESPONSE_LIE
    for rule in ("fairfed", "qffl", "f2gnn", "fedgraphfair", "popets_fairfed", "bfwa"):
        wh, wl = _capture_weights(rule, BEST_RESPONSE_LIE[rule])
        assert wh is not None and wl is not None, f"{rule} exposed no weights"
        moved = max(abs(a - b) for a, b in zip(wh, wl))
        assert moved > 0.0, (
            f"{rule} reads a client-reported field but its weights did not move "
            f"under the best-response lie -- the attack is not reaching the channel")


def test_metadata_blind_rules_are_bit_exact_under_a_false_report():
    """cgsv and fltrust score gradients only; a false report must be inert."""
    from src.federated.attacks import BEST_RESPONSE_LIE, LEGACY_LIE
    for rule in ("cgsv", "fltrust"):
        wh, wl = _capture_weights(rule, BEST_RESPONSE_LIE.get(rule, LEGACY_LIE))
        assert wh == wl, f"{rule} is supposed to read no metadata but its weights moved"


def test_popets_polynomial_is_nearly_inert_at_realistic_disparities():
    """Records a finding, so that changing it is a deliberate act.

    PoPETs' FHE-friendly surrogate replaces FairFed's exp(-beta|F_i-F_g|) with
    -beta(F_i-F_g)^2 + 1. At a 0.15 gap and beta=1 the polynomial spans 0.0225
    against the exponential's 0.1393 -- about a sixth of the steering range. The
    rule therefore barely moves under a false report, but for the same reason it
    barely steers on fairness at all. Read it as inertness, not robustness.
    """
    from src.federated.attacks import BEST_RESPONSE_LIE
    wh, wl = _capture_weights("popets_fairfed", BEST_RESPONSE_LIE["popets_fairfed"])
    moved = max(abs(a - b) for a, b in zip(wh, wl))
    assert 0.0 < moved < 0.01, (
        f"popets_fairfed moved by {moved}; it was ~5e-4 when this was characterised. "
        f"If the weighting changed, re-derive the claim rather than "
        f"widening this bound.")


def test_poison_updates_actually_applies_the_best_response_lie():
    """Locks the integration, not just the table.

    The table can be correct while nothing calls it. The original defect was of
    exactly this shape: poison_updates wrote dpd/eod/perf and never loss, so the
    two rules that read loss were unattackable and would have been written up as
    immune.
    """
    from src.federated.attacks import poison_updates
    base = [{"n": 100, "perf": 0.7, "dpd": 0.12, "eod": 0.08, "loss": 0.3,
             "group1_rate": 0.4} for _ in range(4)]
    for rule, field in (("qffl", "loss"), ("fedgraphfair", "loss"),
                        ("fairfed", "dpd"), ("f2gnn", "group1_rate")):
        metas = [dict(m) for m in base]
        ups = [torch.randn(8) for _ in range(4)]
        _, out = poison_updates("fairness_poison", ups, metas, [0], meta_lie=rule)
        assert out[0][field] != base[0][field], (
            f"poison_updates did not change {field!r} for meta_lie={rule!r}; "
            f"the lie table is not reaching the transmitted report")
        assert out[1][field] == base[1][field], "a benign client's report was altered"


def test_honest_report_control_leaves_every_field_untouched():
    """The control arm must poison the update and nothing else."""
    from src.federated.attacks import poison_updates
    metas = [{"n": 100, "perf": 0.7, "dpd": 0.12, "eod": 0.08, "loss": 0.3} for _ in range(3)]
    before = [dict(m) for m in metas]
    ups = [torch.randn(8) for _ in range(3)]
    _, out = poison_updates("fairness_poison_honest_report", ups, metas, [0], meta_lie="fairfed")
    assert out == before, "the honest-report control altered the metadata channel"


def test_metadata_contrast_pairs_on_the_intersection_of_finite_seeds():
    """A diverged run must drop the PAIR, never leak NaN into the statistics.

    q-FedAvg diverges on German for different seeds in different arms, so the
    two arms' means were being taken over different seed sets while the
    artifact looked complete. The contrast must pair on seeds finite in both
    arms and say which ones it dropped.
    """
    import math
    from experiments.revision.metadata_capture_endtoend import paired_contrasts

    lie = [{"seed": 42, "w_adv": 0.60}, {"seed": 43, "w_adv": float("nan")},
           {"seed": 44, "w_adv": 0.50}, {"seed": 45, "w_adv": 0.80}]
    hon = [{"seed": 42, "w_adv": 0.30}, {"seed": 43, "w_adv": 0.20},
           {"seed": 44, "w_adv": 0.50}, {"seed": 45, "w_adv": float("nan")}]

    r = paired_contrasts(lie, hon, metrics=("w_adv",))["w_adv"]
    assert r["seeds_used"] == [42, 44], r["seeds_used"]
    assert r["seeds_dropped"] == [43, 45], r["seeds_dropped"]
    assert math.isfinite(r["wilcoxon_p"]), "a NaN seed leaked into the p-value"
    assert math.isfinite(r["mean_delta"]), "a NaN seed leaked into the mean"
    # seed 44 is an exact tie; the signed-rank test drops it, so the attainable
    # p-floor is set by n_nonzero_pairs, not n_pairs. The write-up depends on
    # this distinction, so lock it.
    assert r["n_pairs"] == 2 and r["n_nonzero_pairs"] == 1, r
    assert r["n_positive"] == 1 and r["n_negative"] == 0, r


def test_folded_normal_mean_matches_its_two_limits_and_is_even():
    """Lemma 4.1's exact expectation, pinned at both ends.

    Theorem 4 uses the sigma >> |delta| limit, sigma*sqrt(2/pi). Comparing a
    measurement against that LIMIT can only agree inside the regime whose
    assumption it encodes, which is close to circular and a referee may say so.
    The exact expression carries no regime condition, so RUN-E3c compares
    against it -- and it is only trustworthy if it reduces to the right thing at
    both ends.
    """
    import math
    from experiments.revision.bfwa_slack_analysis import folded_normal_mean

    # noise-dominant end: delta = 0 recovers Theorem 4's constant exactly
    assert folded_normal_mean(0.0, 1.0) == pytest.approx(math.sqrt(2 / math.pi), rel=1e-12)
    assert folded_normal_mean(0.0, 0.3) == pytest.approx(0.3 * math.sqrt(2 / math.pi), rel=1e-12)

    # signal-dominant end: delta >> sigma recovers |delta|
    assert folded_normal_mean(5.0, 1e-4) == pytest.approx(5.0, rel=1e-9)

    # even in delta, so an unsigned per-client disparity is sufficient input
    for d, s_ in ((0.3, 0.2), (1.0, 2.0), (0.05, 0.05)):
        assert folded_normal_mean(d, s_) == folded_normal_mean(-d, s_)

    # and it must never fall below the noise-only value for a fixed sigma:
    # adding a real disparity can only increase the expected released magnitude
    base = folded_normal_mean(0.0, 0.5)
    for d in (0.1, 0.5, 2.0):
        assert folded_normal_mean(d, 0.5) >= base


def test_build_manifest_reports_the_device_it_was_told_to_use():
    """R13. build_manifest has 85 callers and had no test at all.

    It reads the device from FEDFAIR_DEVICE and silently defaults to "cpu", so
    run_revision_gpu.py -- which never set the variable -- wrote device="cpu"
    into the manifest of a real T4 run while stdout printed "Device: CUDA". The
    hand-reconstructed artifact recorded "cuda" correctly and the measured one
    recorded it wrongly.
    """
    from unittest.mock import patch
    from src.utils.provenance import build_manifest

    prev = os.environ.get("FEDFAIR_DEVICE")
    try:
        # F2: If CUDA is available, "cuda" is reported; if not, falls back to "cpu"
        with patch("torch.cuda.is_available", return_value=True):
            os.environ["FEDFAIR_DEVICE"] = "cuda"
            assert build_manifest()["device"] == "cuda"

        with patch("torch.cuda.is_available", return_value=False):
            os.environ["FEDFAIR_DEVICE"] = "cuda"
            assert build_manifest()["device"] == "cpu", "CUDA requested on non-CUDA torch must record cpu"

            os.environ["FEDFAIR_DEVICE"] = "cpu"
            assert build_manifest()["device"] == "cpu"

            os.environ["FEDFAIR_DEVICE"] = "mps"
            assert build_manifest()["device"] == "mps"

        os.environ.pop("FEDFAIR_DEVICE", None)
        with patch("torch.cuda.is_available", return_value=False):
            m = build_manifest()
            assert m["device"] == "cpu", "the documented default changed"


        # the fields G-prov reads must exist and be machine-generated
        for k in ("git_commit", "git_dirty", "timestamp", "torch_version",
                  "python_version", "platform"):
            assert k in m, f"manifest lost field {k!r}, which G-prov checks"
        # timestamp must carry microseconds and a +00:00 offset -- a Z-suffixed,
        # whole-second value is what a hand-written manifest looks like
        assert re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{1,6}\+00:00$",
                        m["timestamp"]), m["timestamp"]
        # platform.platform() is a long descriptive string, not "Linux-x86_64"
        assert m["platform"].count("-") >= 2, m["platform"]

        # extras win over the computed fields, which is how experiment/args ride along
        assert build_manifest(experiment="x")["experiment"] == "x"
    finally:
        if prev is None:
            os.environ.pop("FEDFAIR_DEVICE", None)
        else:
            os.environ["FEDFAIR_DEVICE"] = prev


