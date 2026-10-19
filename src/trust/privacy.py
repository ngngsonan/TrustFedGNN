"""Differential-privacy accounting via Rényi Differential Privacy (RDP).

This replaces the manuscript's previously *unimplemented* "moments accountant"
claim with a real, self-contained RDP accountant for the Gaussian mechanism
used by FTGD (noise is added only to the clipped fairness-gradient subspace).

For the Gaussian mechanism with noise multiplier ``z = sigma / C`` (C = L2
clipping bound = sensitivity), the RDP at order alpha is

    eps_RDP(alpha) = alpha / (2 z**2).

Over ``T`` sequential releases RDP composes additively: ``T * alpha / (2 z**2)``.
The tight RDP -> (eps, delta)-DP conversion (Canonne, Kamath, Steinke 2020) is

    eps = eps_RDP(alpha) + log((alpha - 1)/alpha) - (log delta + log alpha)/(alpha - 1),

minimised over a grid of orders ``alpha > 1``.

We do not assume privacy amplification by subsampling (local training is
full-batch here), so the reported epsilon is a conservative upper bound.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List

import numpy as np

_DEFAULT_ORDERS = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 256))


def gaussian_rdp(noise_multiplier: float, steps: int, orders=None) -> np.ndarray:
    orders = np.asarray(orders if orders is not None else _DEFAULT_ORDERS, dtype=float)
    if noise_multiplier <= 0:
        return np.full_like(orders, np.inf)
    return steps * orders / (2.0 * noise_multiplier ** 2)


def rdp_to_dp(rdp: np.ndarray, delta: float, orders=None) -> float:
    """Convert per-order RDP to (eps, delta)-DP, minimising over orders."""
    orders = np.asarray(orders if orders is not None else _DEFAULT_ORDERS, dtype=float)
    rdp = np.asarray(rdp, dtype=float)
    eps = rdp + np.log1p(-1.0 / orders) - (math.log(delta) + np.log(orders)) / (orders - 1.0)
    eps = np.where(np.isfinite(eps), eps, np.inf)
    return float(np.min(eps))


def compute_epsilon(noise_multiplier: float, steps: int, delta: float = 1e-5) -> float:
    """(eps, delta)-DP guarantee for ``steps`` Gaussian releases."""
    if steps <= 0:
        return 0.0
    return rdp_to_dp(gaussian_rdp(noise_multiplier, steps), delta)


def calibrate_noise_multiplier(target_epsilon: float, steps: int, delta: float = 1e-5,
                               lo: float = 0.1, hi: float = 200.0, tol: float = 1e-3) -> float:
    """Smallest noise multiplier z achieving (target_epsilon, delta)-DP over steps."""
    if target_epsilon <= 0:
        return hi
    # eps decreases monotonically in z -> binary search
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        eps = compute_epsilon(mid, steps, delta)
        if eps > target_epsilon:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return hi


@dataclass
class PrivacyAccountant:
    """Tracks cumulative DP spend across federated rounds.

    ``noise_multiplier`` is z = sigma / C. Call :meth:`step` once per privatised
    release (per local epoch that adds DP noise) and read :meth:`epsilon` for the
    current (eps, delta) guarantee.
    """

    noise_multiplier: float
    delta: float = 1e-5
    steps: int = 0
    history: List[float] = field(default_factory=list)

    def step(self, n: int = 1) -> None:
        self.steps += n

    def epsilon(self) -> float:
        eps = compute_epsilon(self.noise_multiplier, self.steps, self.delta)
        self.history.append(eps)
        return eps

    def summary(self) -> dict:
        return {
            "noise_multiplier": self.noise_multiplier,
            "steps": self.steps,
            "delta": self.delta,
            "epsilon": self.epsilon(),
        }


def sigma_from_epsilon_classic(epsilon: float, delta: float, clip: float) -> float:
    """Classical single-shot Gaussian mechanism std (Dwork & Roth).

    sigma = C * sqrt(2 ln(1.25/delta)) / epsilon. Used by FTGD to set per-step
    noise; the *cumulative* guarantee across rounds is then reported via the
    RDP accountant above (which is much tighter than naive composition).
    """
    return clip * math.sqrt(2.0 * math.log(1.25 / delta)) / max(epsilon, 1e-8)
