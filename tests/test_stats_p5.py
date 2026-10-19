"""Unit tests for stats_p5.py against known answers (P5-K11.1)."""
import pytest
import numpy as np

from experiments.campaigns.stats_p5 import (
    paired_bootstrap_ci,
    wilcoxon_p,
    sign_test_p,
    holm,
    noninferiority,
    factorial_2x2,
)


def test_paired_bootstrap_ci_constant_and_symmetric():
    # Constant array has identical bounds
    lo, hi = paired_bootstrap_ci([5.0] * 20, n=1000, seed=42)
    assert lo == 5.0
    assert hi == 5.0

    # Symmetric array centered at 0 covers 0
    lo, hi = paired_bootstrap_ci([-2.0, -1.0, 1.0, 2.0], n=2000, seed=42)
    assert lo < 0.0 < hi


def test_wilcoxon_p_known():
    # All positive differences -> highly significant
    d_pos = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    p = wilcoxon_p(d_pos)
    assert p < 0.01

    # Symmetric around zero -> p-value near 1.0
    d_sym = [-2.0, -1.0, 1.0, 2.0]
    p_sym = wilcoxon_p(d_sym)
    assert p_sym > 0.5

    # All zeros -> returns 1.0
    assert wilcoxon_p([0.0, 0.0, 0.0]) == 1.0


def test_sign_test_p_known():
    # 10 positive, 0 negative: p = 2 * (0.5)^10 = 0.001953125
    d = [1.0] * 10
    p = sign_test_p(d)
    assert abs(p - 0.001953125) < 1e-6

    # 5 positive, 5 negative: p = 1.0
    d_bal = [1.0] * 5 + [-1.0] * 5
    assert sign_test_p(d_bal) == 1.0

    # Zeros dropped: 5 pos, 5 neg, 10 zeros -> still p = 1.0
    d_zeros = [1.0] * 5 + [-1.0] * 5 + [0.0] * 10
    assert sign_test_p(d_zeros) == 1.0


def test_holm_stepdown_exact():
    # Known test case
    raw_p = [0.01, 0.04, 0.03, 0.005, 0.2]
    # Sorted: (0.005, *5=0.025), (0.01, *4=0.04), (0.03, *3=0.09), (0.04, *2=0.08->0.09 max), (0.2, *1=0.2)
    # Mapping back to input order:
    # 0.01 -> 0.04
    # 0.04 -> 0.09
    # 0.03 -> 0.09
    # 0.005 -> 0.025
    # 0.2 -> 0.2
    expected = [0.04, 0.09, 0.09, 0.025, 0.2]
    adj = holm(raw_p)
    np.testing.assert_allclose(adj, expected)


def test_noninferiority_pass_and_fail():
    # Pass: delta=0.01, difference is +0.05
    res_pass = noninferiority([0.05] * 20, delta=0.01, n=1000)
    assert res_pass["pass"] is True
    assert abs(res_pass["ci_lower"] - 0.05) < 1e-6

    # Fail: delta=0.01, difference is -0.05 (ci_lower < -0.01)
    res_fail = noninferiority([-0.05] * 20, delta=0.01, n=1000)
    assert res_fail["pass"] is False
    assert abs(res_fail["ci_lower"] - (-0.05)) < 1e-6


def test_factorial_2x2_known():
    # Synthetic values per seed
    y = {
        "RS_R": [10.0, 12.0],
        "RS_C": [6.0, 8.0],
        "SR_R": [4.0, 5.0],
        "SR_C": [2.0, 3.0],
    }
    # Seed 0:
    # order: 0.5 * ((10 - 4) + (6 - 2)) = 0.5 * (6 + 4) = 5.0
    # mag:   0.5 * ((10 - 6) + (4 - 2)) = 0.5 * (4 + 2) = 3.0
    # inter: (10 - 6) - (4 - 2) = 4 - 2 = 2.0
    # Seed 1:
    # order: 0.5 * ((12 - 5) + (8 - 3)) = 0.5 * (7 + 5) = 6.0
    # mag:   0.5 * ((12 - 8) + (5 - 3)) = 0.5 * (4 + 2) = 3.0
    # inter: (12 - 8) - (5 - 3) = 4 - 2 = 2.0
    res = factorial_2x2(y, n_boot=1000, seed=42)

    assert abs(res["order"]["mean"] - 5.5) < 1e-6
    assert abs(res["magnitude"]["mean"] - 3.0) < 1e-6
    assert abs(res["interaction"]["mean"] - 2.0) < 1e-6
    assert len(res["order"]["ci95"]) == 2
