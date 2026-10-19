"""Fast, offline unit tests (synthetic data only -- no downloads).

Run: pytest -q
"""
import math
import warnings

import numpy as np
import pytest
import torch

from src.config import ExperimentConfig, set_seed
from src.data.datasets import load_synthetic
from src.data.partition import (carve_server_holdout, partition_graph,
                                partition_stats)
from src.models import build_model
from src.utils import metrics as M
from src.trust import privacy
from src.trust.trust_score import trust_score as compute_trust
from src.federated import aggregate


# ----------------------------- metrics ----------------------------- #
def test_perfect_and_random_auc():
    y = np.array([0, 0, 1, 1])
    assert M.auc_roc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert abs(M.auc_roc(y, np.array([0.5, 0.5, 0.5, 0.5])) - 0.5) < 1e-9


def test_dpd_zero_when_equal():
    s = np.array([0, 0, 1, 1])
    assert M.demographic_parity_difference(np.array([0.5, 0.5, 0.5, 0.5]), s) == 0.0
    assert abs(M.demographic_parity_difference(np.array([1.0, 1.0, 0.0, 0.0]), s) - 1.0) < 1e-9


def test_metrics_report_divergence_instead_of_hiding_it():
    """SPEC 4.0(c). This test previously asserted the opposite -- that every
    metric stays finite on NaN/inf input -- which made a bug into a requirement:
    non-finite scores were mapped onto the constant 0.5, so a diverged model
    scored auc=0.5 *and dpd=0.0*, i.e. chance accuracy and perfect fairness. It
    is a false positive in our own favour, and it is what the sign_flip rows of
    incentive_audit reported for 80% of rounds. A dead model has no defined
    utility or fairness; it must surface as NaN and be excluded from aggregates,
    not averaged in as a zero disparity."""
    y = np.array([0, 1, 0, 1])
    s = np.array([0, 1, 0, 1])

    out = M.all_metrics(y, np.array([np.nan, np.inf, 0.2, 0.8]), s)
    assert out["diverged"] == 1.0
    assert not np.isfinite(out["auc"]), "diverged AUC must be NaN, never 0.5"
    assert not np.isfinite(out["dpd"]), "diverged DPD must be NaN, never 0.0"

    ok = M.all_metrics(y, np.array([0.1, 0.9, 0.2, 0.8]), s)
    assert ok["diverged"] == 0.0
    # The *global* metrics of a healthy run stay finite. The per-group ones added
    # for levelling-down detection are exempt on purpose: this fixture gives
    # group S=0 the labels [0, 0], and a ranking metric on a single-class subset
    # is undefined -- reporting 0.5/0.0 there would invent a measurement the data
    # cannot support (the same argument as SPEC 4.0(c) above, one level down).
    per_group = {"auc_group0", "auc_group1", "ap_group0", "ap_group1"}
    assert all(np.isfinite(v) for k, v in ok.items() if k not in per_group), \
        "healthy runs stay finite"


def test_equalized_odds_bounds():
    y = np.array([1, 1, 0, 0]); s = np.array([0, 1, 0, 1])
    v = M.equalized_odds(y, np.array([0.9, 0.1, 0.1, 0.9]), s)
    assert 0.0 <= v <= 1.0


# ----------------------------- privacy ----------------------------- #
def test_epsilon_monotone_in_noise():
    e_lo = privacy.compute_epsilon(1.0, 50, 1e-5)
    e_hi = privacy.compute_epsilon(5.0, 50, 1e-5)
    assert e_hi < e_lo            # more noise -> smaller epsilon
    assert privacy.compute_epsilon(2.0, 100, 1e-5) > privacy.compute_epsilon(2.0, 10, 1e-5)


def test_calibration_hits_target():
    z = privacy.calibrate_noise_multiplier(4.0, 60, 1e-5)
    assert abs(privacy.compute_epsilon(z, 60, 1e-5) - 4.0) < 0.2


# --------------------------- aggregation --------------------------- #
def test_fedavg_is_weighted_mean():
    ups = [torch.ones(5), torch.zeros(5)]
    meta = [{"n": 3}, {"n": 1}]
    agg, _ = aggregate("fedavg", ups, meta)
    assert torch.allclose(agg, torch.full((5,), 0.75), atol=1e-6)


def test_krum_rejects_outlier():
    torch.manual_seed(0)
    honest = [torch.randn(8) * 0.01 for _ in range(5)]
    outlier = torch.ones(8) * 100.0
    ups = honest + [outlier]
    meta = [{"n": 1}] * 6
    agg, info = aggregate("krum", ups, meta, krum_f=1)
    assert info["selected"] != 5        # never selects the outlier
    assert agg.abs().max() < 1.0


def test_median_robust_to_outlier():
    ups = [torch.zeros(4), torch.zeros(4), torch.full((4,), 1e6)]
    agg, _ = aggregate("median", ups, [{"n": 1}] * 3)
    assert torch.allclose(agg, torch.zeros(4))


def test_bfwa_weights_simplex():
    from src.federated.aggregation import bfwa_weights
    w = bfwa_weights(torch.tensor([0.9, 0.6, 0.7]), torch.tensor([0.2, 0.01, 0.1]), tau=0.05)
    assert abs(float(w.sum()) - 1.0) < 1e-5 and (w >= 0).all()


# ------------------------------ trust ------------------------------ #
def test_trust_score_bounds():
    m = {"auc": 0.9, "dpd": 0.01, "eod": 0.02}
    t = compute_trust(m, epsilon=1.0, ece=0.02)
    assert 0.0 <= t <= 1.0


def test_trust_geomean_penalises_zero_axis():
    # hold privacy fixed (eps) so the comparison isolates the fairness axis
    good = {"auc": 0.9, "dpd": 0.0, "eod": 0.0}
    unfair = {"auc": 0.9, "dpd": 0.5, "eod": 0.5}
    assert compute_trust(good, p=0.0, epsilon=1.0) > compute_trust(unfair, p=0.0, epsilon=1.0)


def test_no_privacy_scores_zero_privacy_axis():
    from src.trust.trust_score import sub_scores
    assert sub_scores({"auc": 0.9, "dpd": 0.01, "eod": 0.01})["privacy"] == 0.0
    assert sub_scores({"auc": 0.9, "dpd": 0.01, "eod": 0.01}, epsilon=1.0)["privacy"] > 0.5


# ------------------------- data & partition ------------------------ #
def test_synthetic_shapes_and_bias():
    d = load_synthetic(seed=0, num_nodes=400, d=16)
    assert d.x.shape == (400, 16)
    assert d.y.shape[0] == 400 and set(d.y.unique().tolist()) <= {0.0, 1.0}
    assert d.sensitive_attr.shape[0] == 400


def test_partition_covers_all_nodes_and_both_classes():
    set_seed(0)
    d = load_synthetic(seed=0, num_nodes=600, d=8)
    parts = partition_graph(d, 4, "dirichlet", 0.5, seed=0)
    assert sum(p.num_nodes for p in parts) == 600
    for p in parts:                       # floor guarantees both classes present
        assert len(p.y.unique()) == 2


# --------------------- server holdout (A1 / C16 / I5) --------------------- #
def _val_budget(d):
    """How many validation nodes carve_server_holdout is allowed to take."""
    return int(d.val_mask.sum()) // 2


def test_holdout_honours_request_within_budget():
    """A request the split can afford must be granted exactly -- and quietly."""
    d = load_synthetic(seed=0, num_nodes=600, d=8)
    ask = _val_budget(d) // 2                       # comfortably inside budget
    assert ask > 0
    with warnings.catch_warnings():
        warnings.simplefilter("error")              # any warning fails the test
        holdout, rest = carve_server_holdout(d, ask, seed=0)
    assert holdout is not None
    assert holdout.num_nodes == ask
    assert holdout.granted_size == ask == holdout.requested_size
    assert holdout.truncated is False


def test_holdout_never_truncates_silently():
    """C16. The cap at half the validation split is legitimate; clipping the
    request *without saying so* is not. Before this test, german's 250-node val
    split silently turned fu_holdout_size=250 into 125, so a D7/D8 ablation
    would have reported 'a bigger holdout does not help' while measuring a
    holdout that never grew. Two things must hold: the caller is warned, and
    the granted size is readable from the returned object."""
    d = load_synthetic(seed=0, num_nodes=600, d=8)
    budget = _val_budget(d)
    ask = budget * 2 + 10                           # impossible on purpose

    with pytest.warns(RuntimeWarning, match="server-holdout"):
        holdout, rest = carve_server_holdout(d, ask, seed=0)

    assert holdout is not None
    assert holdout.requested_size == ask
    assert holdout.granted_size < ask               # it WAS clipped
    assert holdout.granted_size == holdout.num_nodes    # and says by how much
    assert holdout.truncated is True


def test_holdout_is_node_disjoint_from_every_client():
    """I5 / A1. The server's yardstick must not be data any client -- least of
    all a Byzantine one -- also holds. Node ids are relabelled by the induced
    subgraph, so identity is carried by the (unique) synthetic feature rows."""
    set_seed(0)
    d = load_synthetic(seed=0, num_nodes=600, d=8)
    holdout, rest = carve_server_holdout(d, _val_budget(d) // 2, seed=0)
    assert holdout is not None
    parts = partition_graph(rest, 4, "dirichlet", 0.5, seed=0)

    # (a) the partition of `rest` accounts for every non-holdout node
    assert holdout.num_nodes + sum(p.num_nodes for p in parts) == d.num_nodes

    # (b) no feature row is present on both sides
    def rows(t):
        return {tuple(r) for r in t.tolist()}
    server_rows = rows(holdout.x)
    assert len(server_rows) == holdout.num_nodes        # rows really are unique
    for i, p in enumerate(parts):
        assert not (server_rows & rows(p.x)), f"client {i} shares holdout nodes"


# ------------------------------ models ----------------------------- #
def test_models_forward_probabilities():
    d = load_synthetic(seed=0, num_nodes=200, d=8)
    cfg = ExperimentConfig(hidden_channels=16, num_layers=2, heads=2)
    for name in ["gcn", "gat", "trustfedgnn", "fairgnn", "fairsin"]:
        model = build_model(name, 8, cfg)
        out = model(d.x, d.edge_index, d.sensitive_attr)
        assert out.shape[0] == 200
        assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0


# ------------------- per-group utility metrics (D1) ------------------- #
def test_per_group_utility_metrics_are_a_strict_addition():
    """all_metrics gains auc/ap per sensitive group without disturbing any
    existing key -- global utility can hide 'levelling down', where a fairness
    gap closes because the worse-off group got *worse*."""
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    s = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    # group 0 ranked perfectly, group 1 ranked backwards
    sc = np.array([0.1, 0.9, 0.2, 0.8, 0.9, 0.1, 0.8, 0.2])
    out = M.all_metrics(y, sc, s)

    legacy = {"diverged", "auc", "ap", "f1", "acc", "fpr@80tpr", "dpd",
              "dpd_soft", "dpd_hard", "eod", "eo", "pred_std"}
    assert legacy <= set(out), "existing keys must survive unchanged"
    assert {"auc_group0", "auc_group1", "ap_group0", "ap_group1"} <= set(out)

    assert out["auc_group0"] == pytest.approx(1.0)
    assert out["auc_group1"] == pytest.approx(0.0)
    # the global AUC alone cannot tell these two groups apart
    assert out["auc_group0"] > out["auc"] > out["auc_group1"]
    assert out["ap_group0"] > out["ap_group1"]


def test_per_group_metrics_are_nan_when_undefined():
    """A group that is empty, or carries one label class, has no AUC/AP. That
    must read as NaN, never as 0.5 / 0.0 -- a fabricated 'chance' number would
    be averaged into cross-seed aggregates as if it were measured."""
    y = np.array([0, 0, 0, 1])
    s = np.array([0, 0, 1, 1])            # group 0 is all-negative
    out = M.all_metrics(y, np.array([0.2, 0.3, 0.4, 0.9]), s)
    assert math.isnan(out["auc_group0"]) and math.isnan(out["ap_group0"])
    assert np.isfinite(out["auc_group1"])

    empty = M.all_metrics(np.array([0, 1]), np.array([0.2, 0.9]), np.array([0, 0]))
    assert math.isnan(empty["auc_group1"]), "an empty group is not a measurement"

    # a diverged run keeps a stable key set, all NaN
    bad = M.all_metrics(y, np.array([np.nan, 0.1, 0.2, 0.3]), s)
    assert bad["diverged"] == 1.0
    for k in ("auc_group0", "auc_group1", "ap_group0", "ap_group1"):
        assert k in bad and math.isnan(bad[k])


# --------------------- FTGD gradient surgery (C) --------------------- #
def _ftgd_client(**over):
    """A trustfedgnn client on tiny synthetic data, noise-free and dropout-free
    so both FTGD paths are exactly reproducible."""
    from src.federated.client import Client
    d = load_synthetic(seed=0, num_nodes=90)
    set_seed(7)
    cfg = ExperimentConfig(model="trustfedgnn", dp_mode="ftgd", dp_enabled=False,
                           hidden_channels=16, num_layers=2, heads=2, dropout=0.0,
                           rounds=1, local_epochs=1, **over)
    return Client(0, d, cfg)


def test_sampled_ftgd_path_does_the_same_surgery_as_full_batch():
    """C1. ``_ftgd_batch`` used to backprop task + lambda*|mu0-mu1| directly,
    with no task/fairness decomposition at all -- so every sampled run (the
    ogbn-products-scale ones) trained a different algorithm from the full-batch
    runs it was compared with. On matched inputs the two paths must now produce
    identical gradients, beta included."""
    a, b = _ftgd_client(), _ftgd_client()
    b.set_flat(a.get_flat())                      # identical weights
    x, ei, s = a.data.x, a.data.edge_index, a.data.sensitive_attr
    bs = 40
    m = torch.zeros(x.size(0), dtype=torch.bool)
    m[:bs] = True                                 # batch == the first bs nodes
    opt_a = torch.optim.SGD(a.model.parameters(), lr=0.0)   # lr=0: compare grads
    opt_b = torch.optim.SGD(b.model.parameters(), lr=0.0)
    a.model.train(); b.model.train()

    a._ftgd_step(opt_a, x, ei, s, a._y, m)
    b._ftgd_batch(opt_b, x, ei, s, b._y[m].float(), s[m], bs)

    g_a = torch.cat([p.grad.flatten() for p in a.model.parameters()])
    g_b = torch.cat([p.grad.flatten() for p in b.model.parameters()])
    assert torch.allclose(g_a, g_b, atol=1e-6), "sampled path must match full-batch"
    assert a.model.layers[0].beta.grad is not None
    assert torch.allclose(a.model.layers[0].beta.grad, b.model.layers[0].beta.grad)


def test_min_fair_norm_skips_projection_cleanly_and_is_counted():
    """C2. A near-zero ||g_fair|| used to be papered over with ``+1e-12`` in the
    denominator, which scales the projection toward numerical garbage instead of
    declining to project. Below cfg.ftgd_min_fair_norm the step must degrade to
    plain weighted-sum training, and the skip must be countable."""
    c = _ftgd_client(ftgd_min_fair_norm=1e-8)
    g_total = torch.tensor([1.0, 2.0, 3.0])
    assert c._ftgd_skipped_projection_count == 0

    out = c._gradient_surgery(g_total, torch.tensor([1e-12, 0.0, 0.0]))
    assert torch.equal(out, g_total), "must fall back to the un-decomposed sum"
    assert c._ftgd_skipped_projection_count == 1

    g_fair = torch.tensor([0.0, 1.0, 0.0])
    out = c._gradient_surgery(g_total, g_fair)
    assert not torch.equal(out, g_total), "a healthy g_fair must still be projected"
    assert c._ftgd_skipped_projection_count == 1, "no spurious skip"
    # g_task = g_total - <g_total,g_fair>/||g_fair||^2 * g_fair; g_final = g_task + g_fair
    assert torch.allclose(out, torch.tensor([1.0, 1.0, 3.0]))


def test_ftgd_projection_conflict_only_projects_on_conflict():
    """C3. PCGrad (Yu et al., NeurIPS'20): when the task and fairness gradients
    already agree there is no conflict to resolve, and projecting deletes a
    cooperative component for no fairness gain. 'always' reproduces the
    published unconditional rule."""
    g_fair = torch.tensor([1.0, 0.0])

    agree = torch.tensor([2.0, 0.0]) + g_fair          # <g_task, g_fair> > 0
    assert torch.equal(_ftgd_client(ftgd_projection="conflict")
                       ._gradient_surgery(agree, g_fair), agree)
    assert not torch.equal(_ftgd_client(ftgd_projection="always")
                           ._gradient_surgery(agree, g_fair), agree)

    clash = torch.tensor([-2.0, 0.0]) + g_fair         # <g_task, g_fair> < 0
    projected = _ftgd_client(ftgd_projection="conflict")._gradient_surgery(clash, g_fair)
    assert not torch.equal(projected, clash), "a real conflict must be projected"
    assert torch.allclose(
        projected, _ftgd_client(ftgd_projection="always")._gradient_surgery(clash, g_fair)), \
        "under conflict the two modes must agree"
