"""Fidelity fixtures for aggregation rules against source equations.

Each fixture tests a K=3 toy example against a numpy implementation computed
directly from the source paper equation (or upstream code).
Tolerance is atol=1e-6.
"""
import numpy as np
import pytest
import torch
import torch.nn as nn

from src.federated.aggregation import aggregate


def test_fairfed_fixture():
    """FairFed (Ezzeldin et al., AAAI 2023, Eq. 5-7):
    w_k = max(0, n_k/N - beta * (dpd_k - mean_dpd))
    w_k = w_k / sum(w_k)
    """
    K = 3
    dim = 4
    # Fixed updates
    updates = [torch.tensor([1.0, 2.0, 0.0, -1.0]),
               torch.tensor([0.5, -0.5, 1.0, 2.0]),
               torch.tensor([-1.0, 0.0, 2.0, 1.0])]
    meta = [
        {"n": 100, "dpd": 0.10},
        {"n": 200, "dpd": 0.20},
        {"n": 100, "dpd": 0.30},
    ]
    beta = 1.0

    # Numpy reference from equation:
    n = np.array([100.0, 200.0, 100.0])
    dpd = np.array([0.10, 0.20, 0.30])
    mean_dpd = np.mean(dpd) # 0.20
    w_base = n / np.sum(n)  # [0.25, 0.50, 0.25]
    w_raw = np.maximum(0.0, w_base - beta * (dpd - mean_dpd)) # [0.25 - (-0.10) = 0.35, 0.50 - 0 = 0.50, 0.25 - 0.10 = 0.15]
    w_ref = w_raw / np.sum(w_raw)
    stack_np = np.stack([u.numpy() for u in updates])
    agg_ref = np.sum(w_ref[:, None] * stack_np, axis=0)

    agg_lib, info_lib = aggregate("fairfed", updates, meta, fairfed_beta=beta)
    assert np.allclose(info_lib["weights"], w_ref, atol=1e-6)
    assert np.allclose(agg_lib.numpy(), agg_ref, atol=1e-6)


def test_qffl_fixture():
    """q-FedAvg heuristic power variant (Li et al., ICLR 2020):
    w_k = (loss_k)^q / sum(loss_j^q)
    """
    updates = [torch.tensor([1.0, 2.0, 0.0, -1.0]),
               torch.tensor([0.5, -0.5, 1.0, 2.0]),
               torch.tensor([-1.0, 0.0, 2.0, 1.0])]
    meta = [
        {"loss": 0.2},
        {"loss": 0.4},
        {"loss": 0.8},
    ]
    q = 2.0

    losses = np.array([0.2, 0.4, 0.8])
    w_raw = losses ** q
    w_ref = w_raw / np.sum(w_raw)
    stack_np = np.stack([u.numpy() for u in updates])
    agg_ref = np.sum(w_ref[:, None] * stack_np, axis=0)

    agg_lib, info_lib = aggregate("qffl", updates, meta, q_ffl=q)
    assert np.allclose(info_lib["weights"], w_ref, atol=1e-6)
    assert np.allclose(agg_lib.numpy(), agg_ref, atol=1e-6)


def test_qffl_eq_fixture():
    """q-FedAvg upstream exact formulation (litian96/fair_flearn flearn/trainers/qffedavg.py):
    Delta'_k = F_k^q * L * Delta_k
    h_k = q * F_k^{q-1} * ||L * Delta_k||^2 + L * F_k^q
    agg = sum(Delta'_k) / sum(h_k)
    c_k = F_k^q * L / sum(h_k), w = c / sum(c)
    """
    updates = [torch.tensor([1.0, 2.0, 0.0, -1.0]),
               torch.tensor([0.5, -0.5, 1.0, 2.0]),
               torch.tensor([-1.0, 0.0, 2.0, 1.0])]
    meta = [
        {"loss_pre": 0.25},
        {"loss_pre": 0.50},
        {"loss_pre": 0.75},
    ]
    q = 2.0
    L = 50.0 # e.g. 1 / lr with lr=0.02

    # Numpy reference of upstream equation:
    F = np.array([0.25, 0.50, 0.75]) + 1e-10
    deltas = np.stack([u.numpy() for u in updates])
    l_deltas = L * deltas
    norms_sq = np.sum(l_deltas ** 2, axis=1)
    f_q = F ** q
    f_qm1 = F ** (q - 1.0)
    deltas_prime = (f_q * L)[:, None] * deltas
    h = q * f_qm1 * norms_sq + L * f_q
    h_sum = np.sum(h)
    agg_ref = np.sum(deltas_prime, axis=0) / h_sum
    c_ref = (f_q * L) / h_sum
    w_ref = c_ref / np.sum(c_ref)

    agg_lib, info_lib = aggregate("qffl_eq", updates, meta, q_ffl=q, qffl_L=L)
    assert np.allclose(info_lib["weights"], w_ref, atol=1e-6)
    assert np.allclose(agg_lib.numpy(), agg_ref, atol=1e-6)
    assert np.allclose(info_lib["qffl_scale"], np.sum(c_ref), atol=1e-6)


def test_f2gnn_fixture():
    """F2GNN (Meng et al., ICDM 2023, Eq. 6-8):
    gamma_f = softmax(-dpd / 0.1)
    gamma_e = softmax(gbal / 0.1), gbal = 1 - |2*group1_rate - 1|
    w = softmax((0.5 * gamma_e + gamma_f) / 0.1)
    """
    updates = [torch.tensor([1.0, 0.0]),
               torch.tensor([0.0, 1.0]),
               torch.tensor([1.0, 1.0])]
    meta = [
        {"dpd": 0.1, "group1_rate": 0.5},
        {"dpd": 0.2, "group1_rate": 0.4},
        {"dpd": 0.3, "group1_rate": 0.2},
    ]

    dpd = np.array([0.1, 0.2, 0.3])
    r = np.array([0.5, 0.4, 0.2])
    gbal = 1.0 - np.abs(2.0 * r - 1.0)

    def sm(x):
        e = np.exp(x - np.max(x))
        return e / np.sum(e)

    gamma_f = sm(-dpd / 0.1)
    gamma_e = sm(gbal / 0.1)
    w_ref = sm((0.5 * gamma_e + gamma_f) / 0.1)
    stack_np = np.stack([u.numpy() for u in updates])
    agg_ref = np.sum(w_ref[:, None] * stack_np, axis=0)

    agg_lib, info_lib = aggregate("f2gnn", updates, meta)
    assert np.allclose(info_lib["weights"], w_ref, atol=1e-6)
    assert np.allclose(agg_lib.numpy(), agg_ref, atol=1e-6)


def test_cgsv_fixture():
    """CGSV (Xu et al., NeurIPS 2021, Eq. 4):
    mean_g = mean(Delta)
    cos_k = <Delta_k, mean_g> / (||Delta_k|| * ||mean_g||)
    w_k = ReLU(cos_k) / sum(ReLU(cos_k))
    """
    updates = [torch.tensor([1.0, 0.5]),
               torch.tensor([0.8, 0.6]),
               torch.tensor([-1.0, -0.5])] # negative cosine
    meta = [{"n": 10}, {"n": 10}, {"n": 10}]

    stack_np = np.stack([u.numpy() for u in updates])
    mean_g = np.mean(stack_np, axis=0)
    norm_mean = np.linalg.norm(mean_g)
    cos = np.array([np.dot(u, mean_g) / (np.linalg.norm(u) * norm_mean) for u in stack_np])
    ts = np.maximum(0.0, cos)
    w_ref = ts / np.sum(ts)
    agg_ref = np.sum(w_ref[:, None] * stack_np, axis=0)

    agg_lib, info_lib = aggregate("cgsv", updates, meta)
    assert np.allclose(info_lib["weights"], w_ref, atol=1e-6)
    assert np.allclose(agg_lib.numpy(), agg_ref, atol=1e-6)


def test_fltrust_root_fixture():
    """FLTrust (Cao et al., NDSS 2021, Alg. 1):
    TS_k = ReLU(cos(Delta_k, Delta_root))
    Delta_tilde_k = Delta_k * ||Delta_root|| / ||Delta_k||
    agg = sum(TS_k / sum(TS) * Delta_tilde_k)
    If sum(TS) == 0: agg = 0, fallback='all_zero_trust'.
    """
    delta_root = torch.tensor([1.0, 1.0, 1.0])
    updates = [
        torch.tensor([2.0, 2.0, 2.0]),   # aligned, norm=sqrt(12)
        torch.tensor([1.0, 0.0, -1.0]),  # orthogonal, cos=0
        torch.tensor([-1.0, -1.0, -1.0]) # opposite, cos < 0
    ]
    meta = [{}, {}, {}]

    root_np = delta_root.numpy()
    root_norm = np.linalg.norm(root_np)
    stack_np = np.stack([u.numpy() for u in updates])

    cos = np.array([np.dot(u, root_np) / (np.linalg.norm(u) * root_norm) for u in stack_np])
    ts = np.maximum(0.0, cos)
    w_ref = ts / np.sum(ts)
    scaled_ref = np.stack([u * (root_norm / np.linalg.norm(u)) for u in stack_np])
    agg_ref = np.sum(w_ref[:, None] * scaled_ref, axis=0)

    agg_lib, info_lib = aggregate("fltrust_root", updates, meta, delta_root=delta_root)
    assert np.allclose(info_lib["weights"], w_ref, atol=1e-6)
    assert np.allclose(agg_lib.numpy(), agg_ref, atol=1e-6)
    assert np.allclose(info_lib["fltrust_cos"], cos, atol=1e-6)

    # Check sum(TS) == 0 fallback
    opp_updates = [torch.tensor([-1.0, 0.0, 0.0]), torch.tensor([0.0, -1.0, 0.0]), torch.tensor([0.0, 0.0, -1.0])]
    agg_zero, info_zero = aggregate("fltrust_root", opp_updates, meta, delta_root=delta_root)
    assert torch.equal(agg_zero, torch.zeros_like(opp_updates[0]))
    assert info_zero.get("fltrust_fallback") == "all_zero_trust"


def test_fltrust_root_toy_model_training():
    """Test get_server_root_update on a toy PyG model."""
    from types import SimpleNamespace
    from torch_geometric.data import Data
    from src.trust.incentive import get_server_root_update

    class ToyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(2, 1)
        def forward(self, x, edge_index, s=None):
            return torch.sigmoid(self.lin(x)).flatten()

    model = ToyModel()
    # Dummy data
    x = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]], dtype=torch.float32)
    ei = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    y = torch.tensor([1.0, 0.0, 1.0, 0.0], dtype=torch.float32)
    val_mask = torch.ones(4, dtype=torch.bool)
    s = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    data = Data(x=x, edge_index=ei, y=y, val_mask=val_mask, sensitive_attr=s)
    data.num_nodes = 4

    cfg = SimpleNamespace(seed=42, local_lr=0.05, weight_decay=1e-4, local_epochs=2)
    root_update = get_server_root_update(model, data, cfg, t=0)
    assert root_update.shape[0] == sum(p.numel() for p in model.parameters())
    assert not torch.isnan(root_update).any()
    assert float(root_update.norm()) > 0.0, "Root update should be non-zero after 2 steps of SGD"
