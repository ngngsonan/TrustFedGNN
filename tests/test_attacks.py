"""Calibration tests for the two *stealth* Byzantine attacks (IPM, ALIE).

Both attacks are defined by staying inside the benign population's spread. A
mis-sized parameter does not make them "stronger" -- it turns them into a
different (trivially detectable) attack, so the calibration is part of the
attack's identity and is worth pinning down in tests.
"""
import math

import pytest
import torch

from src.config import ExperimentConfig
from src.federated.attacks import alie_z_from_counts, poison_updates


def _benign_updates(n=8, dim=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(dim, generator=g) * 0.1 + 0.5 for _ in range(n)]


# --------------------------------------------------------------------------- #
# IPM (Xie et al., "Fall of Empires"): g_byz = -eps * mean(benign), eps < 1
# --------------------------------------------------------------------------- #
def test_ipm_norm_is_epsilon_times_benign_mean_norm():
    """||g_byz|| must equal eps * ||mean(benign)|| -- 0.5x, not 10x."""
    updates = _benign_updates()
    metas = [{} for _ in updates]
    byz = [0, 1]
    benign_mean = torch.stack([updates[i] for i in range(len(updates))
                               if i not in byz]).mean(0)

    out, _ = poison_updates("ipm", updates, metas, byz, intensity=10.0,
                            ipm_epsilon=0.5)

    ref = float(benign_mean.norm())
    for i in byz:
        assert float(out[i].norm()) == pytest.approx(0.5 * ref, rel=1e-5)
        # and the direction is the reverse of the benign consensus
        cos = torch.nn.functional.cosine_similarity(
            out[i].flatten(), benign_mean.flatten(), dim=0)
        assert float(cos) < -0.99


def test_ipm_stays_inside_the_benign_radius_unlike_the_old_intensity():
    """eps=0.5 keeps the update sub-unit-scale; the old eps=intensity=10 did not."""
    updates = _benign_updates()
    metas = [{} for _ in updates]
    byz = [0]
    benign_mean = torch.stack([updates[i] for i in range(len(updates))
                               if i not in byz]).mean(0)
    ref = float(benign_mean.norm())

    stealthy, _ = poison_updates("ipm", updates, metas, byz, ipm_epsilon=0.5)
    loud, _ = poison_updates("ipm", updates, metas, byz, ipm_epsilon=10.0)

    assert float(stealthy[0].norm()) < ref          # inside the benign radius
    assert float(loud[0].norm()) > 5 * ref          # degenerate scaling attack


def test_ipm_epsilon_default_matches_config_default():
    """The function default must not silently disagree with the config default."""
    updates = _benign_updates()
    metas = [{} for _ in updates]
    cfg_eps = ExperimentConfig().ipm_epsilon
    assert cfg_eps == 0.5
    a, _ = poison_updates("ipm", updates, metas, [0])
    b, _ = poison_updates("ipm", updates, metas, [0], ipm_epsilon=cfg_eps)
    assert torch.allclose(a[0], b[0])


# --------------------------------------------------------------------------- #
# ALIE (Baruch et al.): z solved from (n, f), not hardcoded
# --------------------------------------------------------------------------- #
def test_alie_z_matches_the_closed_form():
    """z = Phi^-1((n - f - s) / (n - f)) with s = floor(n/2 + 1) - f."""
    norm = pytest.importorskip("scipy.stats").norm
    for n, f in [(10, 1), (20, 4), (50, 12), (25, 5)]:
        s = math.floor(n / 2 + 1) - f
        expected = float(norm.ppf((n - f - s) / (n - f)))
        assert alie_z_from_counts(n, f) == pytest.approx(expected, abs=1e-9)


def test_alie_z_is_monotone_in_the_byzantine_fraction():
    """z is non-decreasing in f for fixed n.

    The numerator of the quantile argument, ``n - f - s``, collapses to the
    f-independent ``n - floor(n/2 + 1)``, while the denominator ``n - f``
    shrinks with f. So controlling a larger share of the population lets the
    attacker push the mean *further* while still hiding inside the surviving
    benign spread. (The task brief guessed the opposite direction; the closed
    form above -- and the intuition that fewer benign workers are then needed
    on the attacker's side of the median -- says otherwise.)
    """
    for n in (10, 20, 50):
        zs = [alie_z_from_counts(n, f) for f in range(0, n // 2)]
        assert all(b >= a - 1e-12 for a, b in zip(zs, zs[1:])), (n, zs)
        assert zs[-1] > zs[0]          # strictly moves, i.e. f actually matters


def test_alie_z_depends_on_n_and_is_not_the_old_hardcoded_constant():
    z_small = alie_z_from_counts(10, 2)
    z_large = alie_z_from_counts(50, 10)
    assert z_small != z_large
    # the old code used 1.5 regardless of (n, f); nothing in a plausible range
    # of populations reproduces that
    for n in (5, 10, 20, 50, 100):
        for f in range(0, n // 2):
            assert abs(alie_z_from_counts(n, f) - 1.5) > 1e-6


def test_alie_z_degenerate_inputs():
    assert alie_z_from_counts(0, 0) == 0.0
    assert alie_z_from_counts(5, 5) == 0.0
    assert alie_z_from_counts(5, 9) == 0.0


def test_alie_uses_calibrated_z_by_default_and_honours_an_explicit_override():
    updates = _benign_updates(n=10)
    metas = [{} for _ in updates]
    byz = [0, 1, 2]
    others = [updates[i] for i in range(len(updates)) if i not in byz]
    bmean = torch.stack(others).mean(0)
    bstd = torch.stack(others).std(0) + 1e-8

    auto, _ = poison_updates("alie", updates, metas, byz)
    z = alie_z_from_counts(len(updates), len(byz))
    assert torch.allclose(auto[0], bmean - z * bstd, atol=1e-6)

    fixed, _ = poison_updates("alie", updates, metas, byz, alie_z=1.5)
    assert torch.allclose(fixed[0], bmean - 1.5 * bstd, atol=1e-6)
    assert not torch.allclose(auto[0], fixed[0])


def test_alie_output_stays_within_the_benign_range_when_calibrated():
    """The point of the calibration: the poisoned update is not an outlier."""
    updates = _benign_updates(n=20, dim=128, seed=3)
    metas = [{} for _ in updates]
    byz = list(range(6))
    benign = torch.stack([updates[i] for i in range(len(updates))
                          if i not in byz])
    out, _ = poison_updates("alie", updates, metas, byz)

    bmean = benign.mean(0)
    dists = (benign - bmean).norm(dim=1)
    assert float((out[0] - bmean).norm()) <= float(dists.max())

    loud, _ = poison_updates("alie", updates, metas, byz, alie_z=1.5)
    assert float((loud[0] - bmean).norm()) > float(dists.max())


def test_other_attacks_are_unaffected_by_the_new_parameters():
    updates = _benign_updates()
    metas = [{} for _ in updates]
    for atk in ("gaussian", "scaling", "sign_flip", "fairness_poison"):
        torch.manual_seed(0)
        a, ma = poison_updates(atk, updates, metas, [0], intensity=10.0)
        torch.manual_seed(0)
        b, mb = poison_updates(atk, updates, metas, [0], intensity=10.0,
                               ipm_epsilon=0.123, alie_z=9.9)
        assert torch.allclose(a[0], b[0])
        assert ma[0] == mb[0]
