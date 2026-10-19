"""Unit tests for heterogeneous privacy contract (C21 / Theorem 1).

Tests:
1. Equal sigmas match the equal-noise formula exactly.
2. Monotonicity: larger noise monotonically increases minimax floor R*.
3. Small sigma collapse: a single low-noise client pulls the minimax floor to 0.
4. Counterexample: weighted-average equal-noise approximation yields an overly
   optimistic bound strictly higher than the true heterogeneous minimax risk floor.
"""
from __future__ import annotations

import math
import pytest


def calc_r_het(sigmas: list[float], tau: float = 0.05) -> float:
    valid = [s for s in sigmas if s > 0]
    if not valid:
        return 0.0
    S = sum(1.0 / (s ** 2) for s in valid)
    return max(0.0, 0.5 * (1.0 - (tau / 2.0) * math.sqrt(S)))


def calc_r_equal(sigmas: list[float], weights: list[float], tau: float = 0.05) -> float:
    K = len(sigmas)
    sigma_w = sum(w * s for w, s in zip(weights, sigmas))
    if sigma_w <= 0:
        return 0.0
    return max(0.0, 0.5 * (1.0 - (tau * math.sqrt(K)) / (2.0 * sigma_w)))


def test_equal_sigmas_match_equal_noise_formula():
    """When all clients have identical noise, R*_het must match R*_equal exactly."""
    K = 10
    sigma_val = 4.5
    tau = 0.05
    sigmas = [sigma_val] * K
    weights = [1.0 / K] * K

    r_het = calc_r_het(sigmas, tau)
    r_equal = calc_r_equal(sigmas, weights, tau)

    assert math.isclose(r_het, r_equal, rel_tol=1e-12)
    assert r_het > 0.0


def test_monotonicity_in_sigma():
    """Increasing noise on clients must increase (or keep at 0) the minimax bound."""
    tau = 0.05
    base_sigmas = [1.0, 2.0, 3.0, 4.0, 5.0]
    elevated_sigmas = [1.5, 2.5, 3.5, 4.5, 5.5]

    r_base = calc_r_het(base_sigmas, tau)
    r_elevated = calc_r_het(elevated_sigmas, tau)

    assert r_elevated > r_base


def test_single_small_sigma_collapses_bound():
    """A single client with negligible noise pulls the whole contract bound to 0."""
    tau = 0.05
    sigmas_healthy = [5.0] * 10
    sigmas_leaked = [5.0] * 9 + [1e-6]

    assert calc_r_het(sigmas_healthy, tau) > 0.4
    assert calc_r_het(sigmas_leaked, tau) == 0.0


def test_counterexample_weighted_average_higher_than_heterogeneous():
    """Demonstrates that weighted average sigma over-estimates the true heterogeneous bound."""
    tau = 0.1
    K = 2
    weights = [0.5, 0.5]
    sigmas = [0.1, 10.0]

    r_het = calc_r_het(sigmas, tau)
    r_equal = calc_r_equal(sigmas, weights, tau)

    # Equal noise uses sigma_w = 5.05 -> bound ~ 0.493
    # Heterogeneous uses S = 100.01 -> bound ~ 0.250
    assert r_equal > r_het
    assert r_equal > 0.45
    assert r_het < 0.30
