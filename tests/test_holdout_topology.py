"""Unit tests for holdout topology modes (ADR-27, K8).
"""
import sys
from pathlib import Path
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.datasets import load_dataset
from src.data.partition import carve_server_holdout


def test_stratified_mode_is_default_and_bitexact():
    """Verify that calling without mode or with mode='stratified' is bit-exact identical."""
    data1 = load_dataset("german", root="data", seed=42)
    data2 = load_dataset("german", root="data", seed=42)

    h1, r1 = carve_server_holdout(data1, size=100, seed=42)
    h2, r2 = carve_server_holdout(data2, size=100, seed=42, mode="stratified")

    assert h1.num_nodes == h2.num_nodes
    assert torch.all(h1.x == h2.x)
    assert torch.all(h1.edge_index == h2.edge_index)
    assert torch.all(h1.y == h2.y)
    assert torch.all(h1.sensitive_attr == h2.sensitive_attr)
    assert h1.mode == "stratified"
    assert hasattr(h1, "mean_degree")


def test_connected_mode_properties_german():
    """Verify connected mode preserves connectivity on German Credit."""
    data = load_dataset("german", root="data", seed=42)
    val_nodes_orig = torch.nonzero(data.val_mask, as_tuple=False).flatten().tolist()
    val_set = set(val_nodes_orig)

    h_conn, r_conn = carve_server_holdout(data, size=100, seed=42, mode="connected")
    h_strat, r_strat = carve_server_holdout(data, size=100, seed=42, mode="stratified")

    assert h_conn.num_nodes == 100
    assert h_conn.mode == "connected"
    assert h_conn.mean_degree >= h_strat.mean_degree
    # Check that connected holdout has significantly more edges
    assert h_conn.edge_index.shape[1] > h_strat.edge_index.shape[1]

    # Verify disjointness: holdout nodes + rest nodes == original nodes
    assert h_conn.num_nodes + r_conn.num_nodes == data.num_nodes


def test_connected_mode_properties_bail():
    """Verify connected mode preserves connectivity on Bail Recidivism."""
    data = load_dataset("bail", root="data", seed=42)
    take = min(200, len(torch.nonzero(data.val_mask, as_tuple=False).flatten()) // 2)

    h_conn, r_conn = carve_server_holdout(data, size=200, seed=42, mode="connected")
    h_strat, r_strat = carve_server_holdout(data, size=200, seed=42, mode="stratified")

    assert h_conn.num_nodes == take
    assert h_conn.mode == "connected"
    # On Bail, stratified holdout is near-disconnected (degree ~0.27), while connected is ~9.2
    assert h_conn.mean_degree > 5.0
    assert h_conn.mean_degree > h_strat.mean_degree
    assert h_conn.edge_index.shape[1] > h_strat.edge_index.shape[1]
    assert h_conn.num_nodes + r_conn.num_nodes == data.num_nodes
