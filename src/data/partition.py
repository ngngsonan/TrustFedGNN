"""Partition a global graph into per-client subgraphs for federated learning.

Three regimes:
    uniform    : nodes split uniformly at random (IID baseline).
    dirichlet  : label- (or sensitive-) skewed non-IID split via a symmetric
                 Dirichlet(alpha) distribution -- the standard FL heterogeneity
                 model (Hsu et al., 2019). Smaller alpha => more skew.
    community  : Louvain/greedy-modularity communities assigned to clients,
                 producing topology-correlated heterogeneity.

Each client receives an *induced subgraph* with relabelled edges and inherits
the global train/val/test masks for its nodes. This mirrors the cross-silo FL
setting where each institution holds a disjoint portion of the entity graph.
"""
from __future__ import annotations

import warnings
from typing import List, Tuple

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import subgraph


def _induced(data: Data, node_idx: torch.Tensor) -> Data:
    ei, _ = subgraph(node_idx, data.edge_index, relabel_nodes=True,
                     num_nodes=data.num_nodes)
    sub = Data(
        x=data.x[node_idx],
        edge_index=ei,
        y=data.y[node_idx],
        sensitive_attr=data.sensitive_attr[node_idx],
        train_mask=data.train_mask[node_idx],
        val_mask=data.val_mask[node_idx],
        test_mask=data.test_mask[node_idx],
    )
    if hasattr(data, "label_known"):
        sub.label_known = data.label_known[node_idx]
    if hasattr(data, "pokec_label_protocol"):
        sub.pokec_label_protocol = data.pokec_label_protocol
    if hasattr(data, "meta"):
        sub.meta = data.meta
    return sub


def _uniform(n: int, k: int, gen) -> List[torch.Tensor]:
    perm = torch.randperm(n, generator=gen)
    return [perm[i::k] for i in range(k)]


def _dirichlet(values: np.ndarray, k: int, alpha: float, gen,
               floor: float = 0.05) -> List[torch.Tensor]:
    """Assign nodes to clients so each client's class mix ~ Dirichlet(alpha).

    A small per-client proportion floor guarantees every client receives some
    of every class (avoiding degenerate single-class silos that make local AUC
    undefined and destabilise metric-based aggregation), while preserving the
    Dirichlet heterogeneity that FL papers rely on.
    """
    rng = np.random.default_rng(int(torch.randint(0, 2**31 - 1, (1,), generator=gen).item()))
    client_idx: List[List[int]] = [[] for _ in range(k)]
    for cls in np.unique(values):
        idx = np.where(values == cls)[0]
        rng.shuffle(idx)
        props = rng.dirichlet(alpha * np.ones(k))
        props = (1 - floor * k) * props + floor          # mix in a uniform floor
        props = props / props.sum()
        cuts = (np.cumsum(props) * len(idx)).astype(int)[:-1]
        for c, part in enumerate(np.split(idx, cuts)):
            client_idx[c].extend(part.tolist())
    return [torch.tensor(sorted(c), dtype=torch.long) for c in client_idx]


def _community(data: Data, k: int, gen) -> List[torch.Tensor]:
    import networkx as nx

    g = nx.Graph()
    g.add_nodes_from(range(data.num_nodes))
    g.add_edges_from(data.edge_index.t().tolist())
    try:
        if hasattr(nx.community, "louvain_communities"):
            comms = list(nx.community.louvain_communities(g, seed=42))
        else:
            comms = list(nx.community.greedy_modularity_communities(g))
    except Exception:
        return _uniform(data.num_nodes, k, gen)
    # round-robin communities (largest first) into k buckets
    buckets: List[List[int]] = [[] for _ in range(k)]
    for i, com in enumerate(sorted(comms, key=len, reverse=True)):
        buckets[i % k].extend(com)
    return [torch.tensor(sorted(b), dtype=torch.long) for b in buckets if b]


def partition_graph(data: Data, num_clients: int, method: str = "dirichlet",
                    alpha: float = 0.5, by: str = "label", seed: int = 42) -> List[Data]:
    gen = torch.Generator().manual_seed(seed)
    if hasattr(data, "label_known") and not data.label_known.all():
        # P5-K10.4: Dirichlet partition on labeled nodes; unlabeled nodes split uniformly at random
        known_indices = torch.where(data.label_known)[0].cpu().numpy()
        unknown_indices = torch.where(~data.label_known)[0]
        if method == "dirichlet":
            vals = (data.y if by == "label" else data.sensitive_attr)[known_indices].cpu().numpy()
            parts_known = _dirichlet(vals, num_clients, alpha, gen)
        elif method == "uniform":
            parts_known = _uniform(len(known_indices), num_clients, gen)
        elif method == "community":
            parts_known = _uniform(len(known_indices), num_clients, gen)
        else:
            raise ValueError(f"Unknown partition method '{method}'")

        client_idx: List[List[int]] = [[] for _ in range(num_clients)]
        for c in range(num_clients):
            sub_part = parts_known[c].cpu().numpy()
            client_idx[c].extend(known_indices[sub_part].tolist())

        if len(unknown_indices) > 0:
            perm_un = unknown_indices[torch.randperm(len(unknown_indices), generator=gen)]
            for c in range(num_clients):
                c_un = perm_un[c::num_clients]
                client_idx[c].extend(c_un.tolist())

        parts = [torch.tensor(sorted(c), dtype=torch.long) for c in client_idx]
    else:
        if method == "uniform":
            parts = _uniform(data.num_nodes, num_clients, gen)
        elif method == "dirichlet":
            vals = (data.y if by == "label" else data.sensitive_attr).cpu().numpy()
            parts = _dirichlet(vals, num_clients, alpha, gen)
        elif method == "community":
            parts = _community(data, num_clients, gen)
        else:
            raise ValueError(f"Unknown partition method '{method}'")
    return [_induced(data, idx) for idx in parts if len(idx) > 0]


def select_connected_holdout_bfs(
    data: Data, val_idx: torch.Tensor, take: int, seed: int = 42
) -> torch.Tensor:
    """Select `take` validation nodes using BFS from random root nodes (ADR-27, K8),
    balancing label distribution to match `val_idx` as closely as possible.
    """
    from collections import deque
    gen = torch.Generator().manual_seed(seed)
    val_set = set(val_idx.cpu().tolist())
    val_y = data.y[val_idx].cpu().numpy()

    # Target label quotas
    classes, counts = np.unique(val_y, return_counts=True)
    total_val = len(val_idx)
    target_quotas = {}
    remaining_take = take
    for i, c in enumerate(classes):
        if i == len(classes) - 1:
            q = remaining_take
        else:
            q = int(round(take * (counts[i] / total_val)))
            q = min(q, remaining_take)
        target_quotas[int(c)] = q
        remaining_take -= q

    # Build adjacency list restricted to val_idx (vectorized)
    adj = {u: [] for u in val_set}
    ei = data.edge_index
    is_val = torch.zeros(data.num_nodes, dtype=torch.bool)
    is_val[val_idx] = True
    mask_val_edges = is_val[ei[0]] & is_val[ei[1]]
    val_ei = ei[:, mask_val_edges]
    for i in range(val_ei.shape[1]):
        u = int(val_ei[0, i].item())
        v = int(val_ei[1, i].item())
        adj[u].append(v)

    # Shuffle root candidates
    perm = torch.randperm(len(val_idx), generator=gen).tolist()
    root_candidates = [val_idx[p].item() for p in perm]

    selected = set()
    current_quotas = {int(c): 0 for c in classes}

    while len(selected) < take:
        # Find next root: only accept candidate belonging to an unfilled class quota
        root = None
        for cand in root_candidates:
            if cand in selected:
                continue
            cand_y = int(data.y[cand].item())
            if current_quotas[cand_y] < target_quotas[cand_y]:
                root = cand
                break
        if root is None:
            # Fall back to any unselected candidate only when no candidate with unfilled quota exists
            for cand in root_candidates:
                if cand not in selected:
                    root = cand
                    break
        if root is None:
            break

        # BFS from root
        q = deque([root])
        selected.add(root)
        current_quotas[int(data.y[root].item())] += 1

        while q and len(selected) < take:
            curr = q.popleft()
            neighbors = adj.get(curr, [])
            if neighbors:
                n_perm = torch.randperm(len(neighbors), generator=gen).tolist()
                shuffled_neighbors = [neighbors[p] for p in n_perm]
            else:
                shuffled_neighbors = []

            # First pass: neighbors that fit quotas
            for nxt in shuffled_neighbors:
                if nxt not in selected:
                    ny = int(data.y[nxt].item())
                    if current_quotas[ny] < target_quotas[ny]:
                        selected.add(nxt)
                        current_quotas[ny] += 1
                        q.append(nxt)
                        if len(selected) >= take:
                            break

            # If queue is empty and we still have unvisited neighbors, take any to keep connectivity
            if not q and len(selected) < take:
                for nxt in shuffled_neighbors:
                    if nxt not in selected:
                        selected.add(nxt)
                        current_quotas[int(data.y[nxt].item())] += 1
                        q.append(nxt)
                        if len(selected) >= take:
                            break

    # If disconnected components didn't reach take, fill remaining from root_candidates
    if len(selected) < take:
        for cand in root_candidates:
            if cand not in selected:
                selected.add(cand)
                if len(selected) >= take:
                    break

    return torch.tensor(sorted(list(selected)), dtype=torch.long)


def carve_server_holdout(data: Data, size: int, seed: int = 42,
                         mode: str = "stratified") -> Tuple[Data, Data]:
    """Split ``size`` validation nodes off for the server *before* partitioning.

    Motivation (A1/F10). The FU-Shapley target gradient is the server's only
    independent yardstick for scoring clients, so it must not be computed on
    data the clients -- including the Byzantine ones -- control. Building it
    from the pooled client validation nodes assumes away the very threat the
    robustness study measures, making the verifiability claim circular. Carving
    the holdout out here, ahead of ``partition_graph``, is what makes "the
    server has a small clean set" an explicit, auditable assumption instead of
    an accident of the plumbing.

    The returned graphs are node-disjoint, so edges crossing the cut are
    dropped from both sides -- an unavoidable cost of a genuinely disjoint
    holdout, and one the ablation over ``fu_holdout_size`` should report.
    Measured on german (1000 nodes, seed 0): mean degree falls 43.48 -> 4.12 at
    ``size=100``. The client subgraphs go through the *same* ``_induced`` call
    (see ``partition_graph``), and under Dirichlet alpha=0.3 they scatter from
    49 to 498 nodes with degrees 2.45 .. 20.54 -- i.e. two of five clients are
    *sparser* than the holdout. Neither side is "the dense one"; see ``1_1``
    section 1.1.6 and ``1_3`` section 3.9.7.

    Returns ``(holdout, rest)``. If there are not enough validation nodes to
    spare, returns ``(None, data)`` and the caller must fall back to pooled.

    The returned ``holdout`` carries ``requested_size`` / ``granted_size`` /
    ``truncated`` so callers and tests can tell an honoured request from a
    silently clipped one (C16).
    """
    gen = torch.Generator().manual_seed(seed)
    val_idx = torch.nonzero(data.val_mask, as_tuple=False).flatten()
    # Leave at least half the validation nodes with the clients: they still need
    # a local split for their own early stopping and reported metrics.
    #
    # C16: this cap used to clip *silently*. On german the val split is 250
    # nodes, so asking for fu_holdout_size=250 quietly yielded 125 -- and a
    # D7/D8 ablation reading "a bigger holdout does not help" would have been
    # measuring a holdout that never grew. The cap stays (clients genuinely
    # need their own val split); what changes is that it now says so.
    take = min(int(size), int(len(val_idx) // 2))
    if take <= 0:
        return None, data
    if take < int(size):
        warnings.warn(
            f"carve_server_holdout: requested {int(size)} server-holdout nodes "
            f"but only {take} could be spared ({len(val_idx)} validation nodes "
            f"exist and at most half may leave the clients). The holdout is "
            f"{take} nodes, NOT {int(size)} -- any ablation over "
            f"fu_holdout_size must read holdout.num_nodes, not the config.",
            RuntimeWarning, stacklevel=2)

    if mode == "connected":
        pick = select_connected_holdout_bfs(data, val_idx, take, seed=seed)
    else:
        pick = val_idx[torch.randperm(len(val_idx), generator=gen)[:take]]
    mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    mask[pick] = True

    holdout = _induced(data, torch.nonzero(mask, as_tuple=False).flatten())
    # Every holdout node is a server validation node, whatever its original split.
    holdout.val_mask = torch.ones(holdout.num_nodes, dtype=torch.bool)
    holdout.train_mask = torch.zeros(holdout.num_nodes, dtype=torch.bool)
    holdout.test_mask = torch.zeros(holdout.num_nodes, dtype=torch.bool)

    # C16: make the request/grant gap machine-readable, not just a warning a
    # batch run will swallow. `granted_size` is what the server actually scores
    # against; `fu_holdout_size` in the config is only a request.
    holdout.requested_size = int(size)
    holdout.granted_size = int(holdout.num_nodes)
    holdout.truncated = bool(take < int(size))
    holdout.mean_degree = float(holdout.edge_index.shape[1]) / float(holdout.num_nodes) if holdout.num_nodes > 0 else 0.0
    if hasattr(holdout, "label_known") and not holdout.label_known.all():
        y_h = holdout.y[holdout.label_known]
        holdout.pos_rate = float((y_h == 1).float().mean().item()) if len(y_h) > 0 else 0.0
    else:
        holdout.pos_rate = float((holdout.y == 1).float().mean().item()) if holdout.num_nodes > 0 else 0.0
    holdout.mode = mode

    rest = _induced(data, torch.nonzero(~mask, as_tuple=False).flatten())
    return holdout, rest


def partition_stats(clients: List[Data]) -> List[dict]:
    """Per-client heterogeneity summary (for logging / the paper's setup table)."""
    stats = []
    for i, c in enumerate(clients):
        if hasattr(c, "label_known") and not c.label_known.all():
            m = c.train_mask if c.train_mask.any() else c.label_known
            y_eval = c.y[m]
        else:
            y_eval = c.y
        s = c.sensitive_attr

        has_two_classes = bool(len(torch.unique(y_eval)) >= 2) if len(y_eval) > 0 else False
        has_two_groups = bool(len(torch.unique(s)) >= 2) if len(s) > 0 else False

        entry = {
            "client": i,
            "nodes": int(c.num_nodes),
            "edges": int(c.edge_index.shape[1]),
            "pos_rate": round(float(y_eval.float().mean()), 4) if len(y_eval) > 0 else 0.0,
            "group1_rate": round(float((s == 1).float().mean()), 4) if len(s) > 0 else 0.0,
            "train": int(c.train_mask.sum()),
        }
        if not has_two_classes:
            entry["missing_class"] = True
        if not has_two_groups:
            entry["missing_group"] = True
        stats.append(entry)
    return stats


def edge_retention(original_data: Data, client_data_list: List[Data]) -> float:
    """Fraction of the global graph's edges that survive partitioning.

    ``sum_k |E_k| / |E|`` over the induced client subgraphs. Every edge whose
    endpoints land on two different clients is dropped by ``_induced``, so this
    is the share of relational signal the federation still sees at all -- the
    quantity that separates "federated GNN training" from "K disconnected GNNs
    that happen to average their weights", and the one the partition-strategy
    comparisons need in order to be interpretable.

    Order of magnitude: for a partition into shares ``p_k`` of the nodes that
    ignores topology, an edge survives only if both endpoints fall in the same
    client, so retention ~= ``sum_k p_k^2`` -- about ``1/K`` for near-equal
    shares (20% at K=5, 10% at K=10). Community/METIS partitions do much better
    because they cut along sparse boundaries; Dirichlet partitions sit near the
    uniform figure. Returns 0.0 for an edgeless original graph.
    """
    total = int(original_data.edge_index.shape[1])
    if total <= 0:
        return 0.0
    kept = sum(int(c.edge_index.shape[1]) for c in client_data_list)
    return float(kept) / float(total)


def partition_summary(original_data: Data, client_data_list: List[Data]) -> dict:
    """``partition_stats`` plus the graph-level edge budget of the split."""
    kept = sum(int(c.edge_index.shape[1]) for c in client_data_list)
    total = int(original_data.edge_index.shape[1])
    nodes = sum(int(c.num_nodes) for c in client_data_list)
    shares = [c.num_nodes / max(nodes, 1) for c in client_data_list]
    return {
        "num_clients": len(client_data_list),
        "original_nodes": int(original_data.num_nodes),
        "original_edges": total,
        "client_nodes": nodes,
        "client_edges": kept,
        "edge_retention": edge_retention(original_data, client_data_list),
        # sum_k p_k^2: what a topology-blind split of the same node shares would
        # retain in expectation. Compare against edge_retention to see whether a
        # partition strategy actually respects the graph's community structure.
        "expected_retention_iid": float(sum(p * p for p in shares)),
        "per_client": partition_stats(client_data_list),
    }
