"""Invariants for how ``dp_mode`` and ``dp_enabled`` interact on the client.

``dp_mode`` names the local *training algorithm*; ``dp_enabled`` says whether the
privacy *mechanism* (clip + Gaussian noise) is live. Conflating the two is how a
config ends up clipping gradients to ``dp_clip`` while adding zero noise -- a
variant that is neither the published private method nor plain SGD, and that is
silently reportable as "DP" because the mode name still says so.

These tests pin the contract:
  * ``dp_active`` is the single switch every clip/noise site gates on;
  * FTGD survives ``dp_enabled=False`` at sigma=0 (the `fedfairgnn-nodp` arm);
  * a privacy *mechanism* asked to run without privacy warns loudly and does not
    clip.
"""
from __future__ import annotations

import os
import sys

import warnings

import pytest
import torch

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import ExperimentConfig
from src.data.datasets import load_synthetic
from src.federated.client import Client


def _client(**overrides):
    data = load_synthetic(seed=0, num_nodes=60)
    cfg = ExperimentConfig(**overrides)
    return Client(0, data, cfg)


def test_dp_disabled_means_no_noise_anywhere():
    """dp_enabled=False must leave sigma at exactly zero for every mode."""
    for mode in ["auto", "none", "ftgd", "gradient", "puffle"]:
        with pytest.warns(RuntimeWarning) if mode in ("gradient", "puffle") \
                else _noop_ctx():
            c = _client(model="gcn", dp_enabled=False, dp_mode=mode)
        assert c.dp_active is False, f"mode={mode}: dp_active must be False"
        assert c.dp_sigma == 0.0, f"mode={mode}: dp_sigma must be 0"
        assert c.noise_multiplier == 0.0, f"mode={mode}: noise_multiplier must be 0"


def test_ftgd_survives_dp_disabled():
    """`fedfairgnn-nodp` = FTGD gradient geometry at sigma=0, no warning.

    The arm must differ from `fedfairgnn` by dp_enabled alone, so the mode has to
    stay 'ftgd' rather than collapsing to 'none'.
    """
    c = _client(model="trustfedgnn", dp_enabled=False, dp_mode="ftgd")
    assert c.dp_mode == "ftgd"
    assert c.dp_active is False


def test_dp_mechanism_without_dp_warns():
    """'gradient'/'puffle' ARE the privacy mechanism: refuse to run it silently."""
    for mode in ["gradient", "puffle"]:
        with pytest.warns(RuntimeWarning, match="not differentially private"):
            c = _client(model="gcn", dp_enabled=False, dp_mode=mode)
        assert c.dp_active is False


def test_privatise_grads_is_a_noop_when_dp_inactive():
    """No clipping without noise: gradients must pass through byte-identically."""
    with pytest.warns(RuntimeWarning):
        c = _client(model="gcn", dp_enabled=False, dp_mode="gradient", dp_clip=1e-6)
    for p in c.model.parameters():
        p.grad = torch.full_like(p, 100.0)      # way past any sane clip norm
    before = [p.grad.clone() for p in c.model.parameters()]
    c._privatise_grads()
    for b, p in zip(before, c.model.parameters()):
        assert torch.equal(b, p.grad), "dp_active=False must not touch gradients"


def test_privatise_grads_clips_to_dp_clip_when_active():
    """Clipping bounds ||g|| by dp_clip. Isolated from the noise (sigma forced 0)
    so the bound is exact -- with noise live the perturbation legitimately
    dominates and no norm bound on the released vector holds."""
    c = _client(model="gcn", dp_enabled=True, dp_mode="gradient",
                dp_clip=1.0, dp_epsilon=8.0, dp_delta=1e-5)
    assert c.dp_active is True and c.dp_sigma > 0
    c.dp_sigma = 0.0                     # isolate the clip
    for p in c.model.parameters():
        p.grad = torch.full_like(p, 100.0)
    c._privatise_grads()
    flat = torch.cat([p.grad.flatten() for p in c.model.parameters()])
    assert flat.norm(2).item() == pytest.approx(1.0, rel=1e-5), \
        "clipped gradient norm must equal dp_clip"


def test_privatise_grads_actually_adds_noise_when_active():
    """With the mechanism live the released gradient must be perturbed."""
    c = _client(model="gcn", dp_enabled=True, dp_mode="gradient",
                dp_clip=1.0, dp_epsilon=8.0, dp_delta=1e-5)
    for p in c.model.parameters():
        p.grad = torch.full_like(p, 100.0)
    c._privatise_grads()
    flat = torch.cat([p.grad.flatten() for p in c.model.parameters()])
    assert torch.isfinite(flat).all()
    assert flat.norm(2).item() > 1.0, "noise must move the vector off the clip ball"


def test_auto_mode_still_resolves_as_before():
    """dp_mode='auto' keeps its old meaning: ftgd for fair backbones, else gradient."""
    fair = _client(model="trustfedgnn", dp_enabled=True, dp_mode="auto")
    assert fair.dp_mode == "ftgd"
    plain = _client(model="gcn", dp_enabled=True, dp_mode="auto", local_fairness=False)
    assert plain.dp_mode == "gradient"
    off = _client(model="gcn", dp_enabled=False, dp_mode="auto")
    assert off.dp_mode == "none"


# --------------------------------------------------------------------------- #
# What the client actually RELEASES (Fix A) and what the released statistic is
# a function of (Fix B).
# --------------------------------------------------------------------------- #
def _ftgd_client(nodes=90, **overrides):
    """A trustfedgnn/FTGD client on tiny synthetic data."""
    data = load_synthetic(seed=0, num_nodes=nodes)
    torch.manual_seed(7)
    base = dict(model="trustfedgnn", dp_mode="ftgd", hidden_channels=16,
                num_layers=2, heads=2, dropout=0.0, rounds=1, local_epochs=1)
    base.update(overrides)
    return Client(0, data, ExperimentConfig(**base))


def _run_one_ftgd_step(c):
    opt = torch.optim.SGD(c.model.parameters(), lr=0.0)
    c.model.train()
    c._ftgd_step(opt, c.data.x, c.data.edge_index, c.data.sensitive_attr,
                 c._y, c.data.train_mask)


def test_meta_reports_the_disparity_that_was_actually_released():
    """Fix A. The Gaussian mechanism was noising (mu0, mu1) for the *local loss*
    only -- those numbers left no trace -- while meta() shipped the server a raw
    dpd from an un-noised evaluate() forward pass every round. The privacy
    accounting therefore protected a statistic that was never transmitted."""
    c = _ftgd_client(dp_enabled=True, dp_epsilon=8.0)
    assert c._last_privatised_dpd is None, "nothing released before the first step"
    assert c.meta()["dpd"] == pytest.approx(c.evaluate("val")["dpd"]), \
        "with no privatised statistic yet, the un-noised fallback is all there is"

    _run_one_ftgd_step(c)
    assert c._last_privatised_dpd is not None
    m = c.meta()
    assert m["dpd"] == pytest.approx(c._last_privatised_dpd), \
        "meta() must report the post-noise disparity FTGD released"
    assert m["dpd"] != pytest.approx(c.evaluate("val")["dpd"]), \
        "and therefore not the raw val-split statistic"


def test_privatised_dpd_reporting_is_gated_and_scoped():
    """The fallback is exact: off by flag, off without DP, off outside FTGD."""
    # flag off -> pre-fix behaviour reproduced
    c = _ftgd_client(dp_enabled=True, dp_epsilon=8.0, report_privatised_dpd=False)
    _run_one_ftgd_step(c)
    assert c._last_privatised_dpd is not None
    assert c.meta()["dpd"] == pytest.approx(c.evaluate("val")["dpd"])

    # DP off -> there IS no privatised statistic; do not invent one
    c = _ftgd_client(dp_enabled=False)
    _run_one_ftgd_step(c)
    assert c.dp_sigma == 0.0
    assert c._last_privatised_dpd is None, "sigma=0 releases nothing to report"
    assert c.meta()["dpd"] == pytest.approx(c.evaluate("val")["dpd"])

    # a non-FTGD mode never computes the quantity
    c = _ftgd_client(dp_enabled=True, dp_mode="gradient", dp_epsilon=8.0)
    c.train()
    assert c._last_privatised_dpd is None
    assert c.meta()["dpd"] == pytest.approx(c.evaluate("val")["dpd"])


def test_group1_rate_is_not_transmitted_by_default():
    """Fix A3. group1_rate is the client's exact sensitive-attribute marginal,
    in the clear, unaccounted, every round. Omitted unless explicitly requested
    -- omitted, not zero-filled, so a stand-in can never be mistaken for a
    measurement. Consumers already read it defensively."""
    c = _ftgd_client(dp_enabled=True, dp_epsilon=8.0)
    m = c.meta()
    assert "group1_rate" not in m, "the raw s marginal must not be on the wire"
    assert {"n", "perf", "dpd", "eod", "eo", "loss", "diverged"} <= set(m)

    # FairFed-style aggregation still runs with the key absent
    from src.federated import aggregate
    ups = [torch.ones(4), torch.zeros(4)]
    agg, _ = aggregate("fairfed", ups, [dict(m), dict(m)])
    assert torch.isfinite(agg).all()

    on = _ftgd_client(dp_enabled=True, dp_epsilon=8.0, report_group_rate=True)
    assert 0.0 <= on.meta()["group1_rate"] <= 1.0


def test_fser_makes_predictions_depend_on_s():
    """Fix B, premise. The FTGD sensitivity bound needs y_hat _|_ s, so that
    flipping one node's group moves exactly that node between the two means.
    FSER reads s inside the attention, so a single flip propagates through the
    L-hop neighbourhood -- this test documents the violation the fix routes
    around, and pins that the s-blind pass does not have it."""
    c = _ftgd_client(dp_enabled=True, dp_epsilon=8.0)
    c.model.eval()
    x, ei = c.data.x, c.data.edge_index
    s = c.data.sensitive_attr.clone()
    with torch.no_grad():
        p0 = c.model(x, ei, s)
        s_flip = s.clone(); s_flip[0] = 1 - s_flip[0]
        p1 = c.model(x, ei, s_flip)
        moved = int((p1 - p0).abs().gt(1e-9).sum())
        assert moved > 1, "FSER really does spread one flip beyond its own node"

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            b0 = c.model(x, ei, None)
            b1 = c.model(x, ei, None)
        assert torch.equal(b0, b1)
        assert not torch.allclose(b0, p0), "the s-blind pass is a different function"


def test_released_statistic_is_computed_s_blind_when_dp_is_live():
    """Fix B. The RELEASED group means must come from a forward pass that never
    reads s, while FSER keeps using the real s in the task pathway -- and the
    task pathway must still receive gradients, beta included."""
    seen = []

    def _instrument(client):
        inner = client.model.forward
        def fwd(x, ei, sa=None, *a, **k):
            seen.append(sa is None)
            return inner(x, ei, sa, *a, **k)
        client.model.forward = fwd

    c = _ftgd_client(dp_enabled=True, dp_epsilon=8.0, dp_statistic_s_blind=True)
    _instrument(c)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        _run_one_ftgd_step(c)
    assert len(seen) == 2, "one task forward with s, one s-blind release forward"
    assert seen == [False, True]
    beta_grad = c.model.layers[0].beta.grad
    assert beta_grad is not None and float(beta_grad.abs()) > 0.0, \
        "the task pathway must still train FSER's beta"
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in c.model.parameters())

    # flag off -> the single, sensitivity-broken forward pass of the old code
    seen.clear()
    old = _ftgd_client(dp_enabled=True, dp_epsilon=8.0, dp_statistic_s_blind=False)
    _instrument(old)
    _run_one_ftgd_step(old)
    assert seen == [False], "pre-fix behaviour reuses the s-dependent predictions"

    # no second pass when there is no privacy mechanism to justify it
    seen.clear()
    nodp = _ftgd_client(dp_enabled=False, dp_statistic_s_blind=True)
    _instrument(nodp)
    _run_one_ftgd_step(nodp)
    assert seen == [False]


class _noop_ctx:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False
