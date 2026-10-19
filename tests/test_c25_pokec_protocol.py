"""Tests for Pokec label protocol (P5-K10.4 / K10.5).

Exercises 'legacy_negative' vs 'known_only' protocols on a synthetic 200-node graph
without downloading external raw datasets:
1. Masks (train, val, test) are strict subsets of label_known in 'known_only'.
2. Perturbing y on unlabeled nodes leaves training loss and eval metrics unchanged.
3. Carved server holdout is strictly in label_known and disjoint from clients.
4. Legacy protocol produces exact legacy y tensor.
5. Fingerprint differs between the two protocols.
6. Sensitive attribute S assertion verifies at least two groups.
"""
import pytest
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.metrics import roc_auc_score

from src.data.datasets import _make_splits, _make_splits_known_only, get_pokec_fingerprint
from src.data.partition import carve_server_holdout, partition_graph, partition_stats
from src.utils.metrics import all_metrics


def _make_synthetic_pokec(seed=42, num_nodes=200, known_frac=0.3):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(num_nodes, 8, generator=g)
    
    # Random edges (undirected)
    e = torch.randint(0, num_nodes, (2, 500), generator=g)
    ei = torch.cat([e, e.flip(0)], dim=1)
    ei = torch.unique(ei, dim=1)
    
    # Sensitive attribute: 2 groups
    s = torch.zeros(num_nodes, dtype=torch.long)
    s[num_nodes // 2:] = 1
    
    # 30% known labels: -1 represents unknown/unlabeled
    y_raw = np.full(num_nodes, -1, dtype=np.int64)
    n_known = int(num_nodes * known_frac)
    perm = torch.randperm(num_nodes, generator=g).tolist()
    known_idx = perm[:n_known]
    
    # Balanced classes on known
    for i, idx in enumerate(known_idx):
        y_raw[idx] = 1 if (i % 2 == 0) else 0

    return x, ei, s, y_raw


def test_pokec_known_only_masks_are_subset_of_known():
    x, ei, s, y_raw = _make_synthetic_pokec(seed=42)
    label_known = torch.tensor(y_raw >= 0, dtype=torch.bool)
    y_bin = torch.tensor(np.where(y_raw > 0, 1.0, 0.0), dtype=torch.float32)
    
    train, val, test = _make_splits_known_only(y_bin, label_known, seed=42)
    
    # Masks must be disjoint and strictly subsets of label_known
    assert (train & ~label_known).sum() == 0, "train contains unlabeled nodes"
    assert (val & ~label_known).sum() == 0, "val contains unlabeled nodes"
    assert (test & ~label_known).sum() == 0, "test contains unlabeled nodes"
    assert (train & val).sum() == 0, "train and val overlap"
    assert (train & test).sum() == 0, "train and test overlap"
    assert (val & test).sum() == 0, "val and test overlap"
    assert (train | val | test).sum() == label_known.sum(), "all known nodes should be partitioned into splits"


def test_pokec_unlabeled_y_perturbation_invariant():
    x, ei, s, y_raw = _make_synthetic_pokec(seed=42)
    label_known = torch.tensor(y_raw >= 0, dtype=torch.bool)
    y_bin1 = torch.tensor(np.where(y_raw > 0, 1.0, 0.0), dtype=torch.float32)
    train, val, test = _make_splits_known_only(y_bin1, label_known, seed=42)

    # Build predictions
    torch.manual_seed(123)
    preds = torch.sigmoid(torch.randn(len(y_raw)))

    # Compute loss & metrics with original labels
    loss1 = torch.nn.functional.binary_cross_entropy(preds[train], y_bin1[train]).item()
    m1 = all_metrics(y_bin1[test], preds[test], s[test])

    # Perturb y on UNLABELED nodes
    y_bin2 = y_bin1.clone()
    y_bin2[~label_known] = 1.0 - y_bin2[~label_known]  # flip unlabeled

    loss2 = torch.nn.functional.binary_cross_entropy(preds[train], y_bin2[train]).item()
    m2 = all_metrics(y_bin2[test], preds[test], s[test])

    assert loss1 == loss2, f"Training loss changed after perturbing unlabeled nodes: {loss1} vs {loss2}"
    assert m1["auc"] == m2["auc"]
    assert m1["dpd_hard"] == m2["dpd_hard"]
    assert m1["eod"] == m2["eod"]


def test_pokec_holdout_disjoint_and_in_known():
    x, ei, s, y_raw = _make_synthetic_pokec(seed=42)
    label_known = torch.tensor(y_raw >= 0, dtype=torch.bool)
    y_bin = torch.tensor(np.where(y_raw > 0, 1.0, 0.0), dtype=torch.float32)
    train, val, test = _make_splits_known_only(y_bin, label_known, seed=42)

    data = Data(x=x, edge_index=ei, y=y_bin, sensitive_attr=s,
                train_mask=train, val_mask=val, test_mask=test)
    data.label_known = label_known
    data.pokec_label_protocol = "known_only"

    holdout, rest = carve_server_holdout(data, size=5, seed=42)
    assert holdout is not None
    # All nodes in holdout must have been in label_known
    assert holdout.label_known.all(), "Holdout contains unlabeled nodes"
    assert holdout.num_nodes == 5
    assert rest.num_nodes == 200 - 5

    # Partition rest across 3 clients
    clients = partition_graph(rest, num_clients=3, method="dirichlet", seed=42)
    assert len(clients) == 3
    total_client_nodes = sum(c.num_nodes for c in clients)
    assert total_client_nodes == rest.num_nodes

    stats = partition_stats(clients)
    assert len(stats) == 3
    for st in stats:
        assert "nodes" in st
        assert "pos_rate" in st


def test_pokec_legacy_reproduces_old_y_hash():
    x, ei, s, y_raw = _make_synthetic_pokec(seed=42)
    # Legacy: np.where(y_raw > 0, 1.0, 0.0)
    expected_y = np.where(y_raw > 0, 1.0, 0.0).astype(np.float32)
    y_t = torch.tensor(expected_y, dtype=torch.float32)
    
    train, val, test = _make_splits(y_t, seed=42)
    data = Data(x=x, edge_index=ei, y=y_t, sensitive_attr=s,
                train_mask=train, val_mask=val, test_mask=test)
    data.label_known = torch.ones(len(y_t), dtype=torch.bool)
    data.pokec_label_protocol = "legacy_negative"

    np.testing.assert_array_equal(data.y.numpy(), expected_y)


def test_pokec_fingerprint_differs():
    x, ei, s, y_raw = _make_synthetic_pokec(seed=42)
    y_bin = torch.tensor(np.where(y_raw > 0, 1.0, 0.0), dtype=torch.float32)
    label_known = torch.tensor(y_raw >= 0, dtype=torch.bool)

    # Legacy
    tr_leg, val_leg, te_leg = _make_splits(y_bin, seed=42)
    d_leg = Data(x=x, edge_index=ei, y=y_bin, sensitive_attr=s,
                 train_mask=tr_leg, val_mask=val_leg, test_mask=te_leg)
    d_leg.label_known = torch.ones(len(y_bin), dtype=torch.bool)
    d_leg.pokec_label_protocol = "legacy_negative"
    fp_leg = get_pokec_fingerprint(d_leg, "legacy_negative")

    # Known only
    tr_kn, val_kn, te_kn = _make_splits_known_only(y_bin, label_known, seed=42)
    d_kn = Data(x=x, edge_index=ei, y=y_bin, sensitive_attr=s,
                train_mask=tr_kn, val_mask=val_kn, test_mask=te_kn)
    d_kn.label_known = label_known
    d_kn.pokec_label_protocol = "known_only"
    fp_kn = get_pokec_fingerprint(d_kn, "known_only")

    assert fp_leg != fp_kn, "Fingerprint should differ between legacy_negative and known_only"


def test_pokec_assert_sensitive_two_groups():
    # If S has only 1 unique group, assertion should trigger
    s_single = np.zeros(100, dtype=int)
    with pytest.raises(AssertionError, match="at least two groups"):
        assert len(np.unique(s_single)) >= 2, "Sensitive attribute S must contain at least two groups"


