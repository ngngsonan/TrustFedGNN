import pytest
import torch
import numpy as np
from src.federated.aggregation import aggregate

def test_flame_basic_properties():
    K = 10
    d = 100
    torch.manual_seed(42)
    updates = [torch.randn(d) for _ in range(K)]
    metas = [{"n": 100, "perf": 0.8, "dpd": 0.05} for _ in range(K)]

    agg, info = aggregate("flame", updates, metas)
    assert agg.shape == (d,)
    assert not torch.isnan(agg).any()
    assert "weights" in info
    assert "selected" in info
    assert len(info["weights"]) == K
    assert pytest.approx(sum(info["weights"]), abs=1e-5) == 1.0

def test_flame_outlier_rejection():
    # 8 benign updates close to each other, 2 huge outlier updates
    K = 10
    d = 50
    torch.manual_seed(42)
    base = torch.ones(d)
    updates = [base + 0.05 * torch.randn(d) for _ in range(8)]
    # Add 2 orthogonal / far outliers
    updates.append(-10.0 * base + torch.randn(d))
    updates.append(-10.0 * base + torch.randn(d))
    metas = [{"n": 100, "perf": 0.8, "dpd": 0.05} for _ in range(K)]

    agg, info = aggregate("flame", updates, metas)
    selected = set(info["selected"])
    # FLAME should select the majority benign cluster (indices 0..7) and reject 8, 9
    assert 8 not in selected
    assert 9 not in selected
    assert len(selected) == 8
    assert info["weights"][8] == 0.0
    assert info["weights"][9] == 0.0

def test_flame_stealth_poison_vulnerability():
    # Demonstrating the core scientific premise of Section 2.3:
    # Adversary updates crafted with cosine proximity rho ~ 0.85 are clustered TOGETHER
    # with benign clients, so FLAME assigns them uniform weight ~ 1/K!
    K = 10
    d = 100
    torch.manual_seed(42)
    benign = [torch.randn(d) for _ in range(9)]
    bmed = torch.stack(benign).median(0).values
    
    # Adversary nuzzling close to the median (stealth proximity)
    adv = bmed + 0.01 * torch.randn(d)
    updates = benign + [adv]
    metas = [{"n": 100} for _ in range(K)]

    agg, info = aggregate("flame", updates, metas)
    assert 9 in info["selected"]
    assert info["weights"][9] > 0.0
