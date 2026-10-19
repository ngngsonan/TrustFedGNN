"""Real dataset loaders for fairness-aware graph learning.

Three canonical fair-GNN benchmarks with *genuine* sensitive attributes are
supported (the same benchmarks used by our fairness baselines FairINV and
DAB-GNN), plus an optional large-scale social graph and a fully documented
synthetic testbed for controlled bias studies.

    dataset   nodes    sensitive attr        target                framing
    -------   -----    --------------        ------                -------
    german    1,000    Gender (F=1/M=0)      good credit customer  credit risk
    credit    30,000   Age  (>=25 vs <25)    no default next month credit risk
    bail      18,876   WHITE (race)          recidivism (RECID)     pretrial risk
    pokec_z   ~67k     region                working field         social
    synthetic  cfg     injected group        injected fraud signal controlled

Files are downloaded on first use from the public NIFTY / FairGNN mirrors and
cached under ``<data_root>/raw``. Preprocessing follows the widely used NIFTY
convention (Agarwal et al., 2021): edges are symmetrised, features are
z-score standardised, and the sensitive attribute is excluded from the node
feature matrix (fairness best practice — the model must not see S directly).
"""
from __future__ import annotations

import hashlib
import os
import urllib.request
import zipfile
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data

# Public raw mirrors (verified reachable).
_NIFTY = "https://raw.githubusercontent.com/chirag126/nifty/main/dataset"
_FAIRGNN = "https://raw.githubusercontent.com/EnyanDai/FairGNN/main/dataset"

_FILES = {
    "german": [(f"{_NIFTY}/german/german.csv", "german.csv"),
               (f"{_NIFTY}/german/german_edges.txt", "german_edges.txt")],
    "credit": [(f"{_NIFTY}/credit/credit.csv", "credit.csv"),
               (f"{_NIFTY}/credit/credit_edges.txt.zip", "credit_edges.txt.zip")],
    "bail": [(f"{_NIFTY}/bail/bail.csv", "bail.csv"),
             (f"{_NIFTY}/bail/bail_edges.txt", "bail_edges.txt")],
}


@dataclass
class DatasetMeta:
    name: str
    sensitive_name: str
    label_name: str
    positive_meaning: str


def _download(url: str, dest: str) -> None:
    if os.path.exists(dest):
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"[data] downloading {url}")
    urllib.request.urlretrieve(url, dest)
    if dest.endswith(".zip"):
        with zipfile.ZipFile(dest) as zf:
            zf.extractall(os.path.dirname(dest))


def _fetch(name: str, root: str) -> str:
    raw = os.path.join(root, "raw", name)
    for url, fname in _FILES[name]:
        _download(url, os.path.join(raw, fname))
    return raw


def _edges_to_index(path: str, num_nodes: int) -> torch.Tensor:
    e = np.loadtxt(path).astype(np.int64)
    if e.ndim == 1:
        e = e.reshape(-1, 2)
    e = e[(e[:, 0] < num_nodes) & (e[:, 1] < num_nodes)]
    ei = torch.tensor(e.T, dtype=torch.long)
    # symmetrise (undirected) and drop duplicates
    ei = torch.cat([ei, ei.flip(0)], dim=1)
    ei = torch.unique(ei, dim=1)
    return ei


def _standardise(x: np.ndarray) -> torch.Tensor:
    return torch.tensor(StandardScaler().fit_transform(x), dtype=torch.float32)


def _make_splits(y: torch.Tensor, seed: int, ratios=(0.5, 0.25, 0.25)) -> tuple:
    """Stratified train/val/test masks over the whole graph."""
    g = torch.Generator().manual_seed(seed)
    n = y.shape[0]
    train = torch.zeros(n, dtype=torch.bool)
    val = torch.zeros(n, dtype=torch.bool)
    test = torch.zeros(n, dtype=torch.bool)
    for cls in torch.unique(y):
        idx = torch.where(y == cls)[0]
        idx = idx[torch.randperm(len(idx), generator=g)]
        a = int(ratios[0] * len(idx))
        b = int((ratios[0] + ratios[1]) * len(idx))
        train[idx[:a]] = True
        val[idx[a:b]] = True
        test[idx[b:]] = True
    return train, val, test


def _make_splits_known_only(y: torch.Tensor, label_known: torch.Tensor, seed: int,
                            ratios=(0.5, 0.25, 0.25)) -> tuple:
    """Stratified train/val/test masks ONLY over known-labeled nodes (P5-K10.4)."""
    g = torch.Generator().manual_seed(seed)
    n = y.shape[0]
    train = torch.zeros(n, dtype=torch.bool)
    val = torch.zeros(n, dtype=torch.bool)
    test = torch.zeros(n, dtype=torch.bool)
    known_idx = torch.where(label_known)[0]
    y_known = y[known_idx]
    for cls in torch.unique(y_known):
        idx_in_known = torch.where(y_known == cls)[0]
        perm = torch.randperm(len(idx_in_known), generator=g)
        idx = known_idx[idx_in_known[perm]]
        a = int(ratios[0] * len(idx))
        b = int((ratios[0] + ratios[1]) * len(idx))
        train[idx[:a]] = True
        val[idx[a:b]] = True
        test[idx[b:]] = True
    return train, val, test


def get_pokec_fingerprint(data: Data, protocol: str = "legacy_negative") -> str:
    """Compute deterministic fingerprint of dataset splits/labels including protocol version."""
    prot = getattr(data, "pokec_label_protocol", protocol)
    h = hashlib.sha256()
    h.update(str(prot).encode("utf-8"))
    h.update(np.ascontiguousarray(data.y.cpu().numpy()).tobytes())
    h.update(np.ascontiguousarray(data.sensitive_attr.cpu().numpy()).tobytes())
    h.update(np.ascontiguousarray(data.train_mask.cpu().numpy()).tobytes())
    h.update(np.ascontiguousarray(data.val_mask.cpu().numpy()).tobytes())
    h.update(np.ascontiguousarray(data.test_mask.cpu().numpy()).tobytes())
    if hasattr(data, "label_known"):
        h.update(np.ascontiguousarray(data.label_known.cpu().numpy()).tobytes())
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# NIFTY-style tabular graph loaders
# --------------------------------------------------------------------------- #
def _load_tabular(name: str, sens: str, label: str, drop_cols, neg_label,
                  root: str, seed: int, meta: DatasetMeta) -> Data:
    raw = _fetch(name, root)
    df = pd.read_csv(os.path.join(raw, f"{name}.csv"))

    # sensitive attribute (binary 0/1); string genders mapped explicitly
    s_raw = df[sens]
    if not pd.api.types.is_numeric_dtype(s_raw):
        s = (s_raw.astype(str).str.lower().isin(["female", "1", "true"])).astype(int).values
    else:
        s = (s_raw.values > 0).astype(int)

    # label -> {0,1}; ``neg_label`` maps to 0
    y_raw = df[label].values.astype(float)
    y = np.where(y_raw == neg_label, 0.0, 1.0) if neg_label is not None else y_raw.astype(float)

    feat_cols = [c for c in df.columns if c not in ([label, sens] + list(drop_cols))]
    x_df = df[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    x = _standardise(x_df.values.astype(np.float32))

    n = x.shape[0]
    edge_index = _edges_to_index(os.path.join(raw, f"{name}_edges.txt"), n)

    y_t = torch.tensor(y, dtype=torch.float32)
    train, val, test = _make_splits(y_t, seed)
    data = Data(
        x=x,
        edge_index=edge_index,
        y=y_t,
        sensitive_attr=torch.tensor(s, dtype=torch.long),
        train_mask=train, val_mask=val, test_mask=test,
    )
    data.meta = meta
    return data


def load_german(root="data", seed=42) -> Data:
    return _load_tabular(
        "german", sens="Gender", label="GoodCustomer",
        drop_cols=["OtherLoansAtStore", "PurposeOfLoan"], neg_label=-1,
        root=root, seed=seed,
        meta=DatasetMeta("german", "Gender", "GoodCustomer", "good credit customer"),
    )


def load_credit(root="data", seed=42) -> Data:
    return _load_tabular(
        "credit", sens="Age", label="NoDefaultNextMonth",
        drop_cols=["Single"], neg_label=0,
        root=root, seed=seed,
        meta=DatasetMeta("credit", "Age", "NoDefaultNextMonth", "no default next month"),
    )


def load_bail(root="data", seed=42) -> Data:
    return _load_tabular(
        "bail", sens="WHITE", label="RECID",
        drop_cols=["TIME"], neg_label=0,
        root=root, seed=seed,
        meta=DatasetMeta("bail", "WHITE", "RECID", "recidivism within follow-up"),
    )


# --------------------------------------------------------------------------- #
# Synthetic controlled testbed (clearly labelled; for bias-injection studies)
# --------------------------------------------------------------------------- #
def load_synthetic(root="data", seed=42, num_nodes=2000, d=32,
                   fraud_ratio=0.15, bias_strength=2.0, homophily=0.8) -> Data:
    """A *synthetic* graph with a controllable amount of structural bias.

    Group membership S is correlated with both a spurious feature block and the
    graph structure (homophily), so that a naive GNN will learn to associate S
    with the positive class. Used only for controlled ablations; never reported
    as a real-world result.
    """
    g = torch.Generator().manual_seed(seed)
    s = (torch.rand(num_nodes, generator=g) < 0.5).long()
    base = fraud_ratio
    p = torch.where(s == 1, torch.tensor(base + 0.10), torch.tensor(base))
    y = (torch.rand(num_nodes, generator=g) < p).float()

    x = torch.randn(num_nodes, d, generator=g)
    x[y == 1, :d // 4] += 1.5                       # genuine signal
    x[s == 1, d // 4:d // 2] += bias_strength       # spurious, group-correlated

    # homophilous edges: connect same-label with prob `homophily`
    src, dst = [], []
    deg = 6
    perm = torch.randperm(num_nodes, generator=g)
    for i in perm.tolist():
        same = torch.where(y == y[i])[0]
        diff = torch.where(y != y[i])[0]
        for _ in range(deg):
            pool = same if torch.rand(1, generator=g).item() < homophily else diff
            j = pool[torch.randint(len(pool), (1,), generator=g)].item()
            src += [i, j]
            dst += [j, i]
    edge_index = torch.tensor([src, dst], dtype=torch.long)

    train, val, test = _make_splits(y, seed)
    data = Data(x=x, edge_index=edge_index, y=y,
                sensitive_attr=s, train_mask=train, val_mask=val, test_mask=test)
    data.meta = DatasetMeta("synthetic", "injected_group", "injected_fraud", "fraud")
    return data


# --------------------------------------------------------------------------- #
# Elliptic Bitcoin (crypto AML, large scale ~204k nodes)
# --------------------------------------------------------------------------- #
def load_elliptic(root="data", seed=42) -> Data:
    """Elliptic Bitcoin transaction graph (Weber et al., 2019).

    203,769 transactions, 234,355 directed payment flows, 165 anonymised
    features. Labels: illicit (fraud, positive class y=1) vs licit (y=0); a
    large pool of unlabelled nodes is kept in the graph for message passing but
    excluded from the loss/metrics masks.

    Sensitive attribute (documented *operational* proxy, not a protected
    demographic): early- vs late-period transactions, split at the median of
    the 49 anonymised time steps. This lets us study whether an AML model flags
    transactions from one market period disproportionately -- a temporal-drift
    subgroup-disparity concern that is genuine in deployed crypto compliance
    systems.
    """
    from torch_geometric.datasets import EllipticBitcoinDataset

    ds = EllipticBitcoinDataset(root=os.path.join(root, "raw", "elliptic"))
    d = ds[0]

    # timestep aligned by CSV row order (PyG preserves it)
    feat_csv = os.path.join(root, "raw", "elliptic", "raw", "elliptic_txs_features.csv")
    ts = pd.read_csv(feat_csv, header=None).iloc[:, 1].values
    sensitive = torch.tensor((ts > np.median(ts)).astype(int), dtype=torch.long)

    labeled = d.y != 2  # 0 licit, 1 illicit, 2 unknown
    y = torch.where(d.y == 1, 1.0, 0.0).float()

    # stratified random split *within labelled nodes* (so both temporal
    # subgroups appear in every split -- a temporal split would confound S)
    train = torch.zeros(d.num_nodes, dtype=torch.bool)
    val = torch.zeros(d.num_nodes, dtype=torch.bool)
    test = torch.zeros(d.num_nodes, dtype=torch.bool)
    lab_idx = torch.where(labeled)[0]
    sub_train, sub_val, sub_test = _make_splits(y[lab_idx], seed)
    train[lab_idx[sub_train]] = True
    val[lab_idx[sub_val]] = True
    test[lab_idx[sub_test]] = True

    data = Data(x=d.x, edge_index=d.edge_index, y=y,
                sensitive_attr=sensitive,
                train_mask=train, val_mask=val, test_mask=test)
    data.meta = DatasetMeta("elliptic", "time_period(early/late)", "illicit", "illicit transaction")
    return data


# --------------------------------------------------------------------------- #
# Pokec social network (region-based fairness, large scale)
# --------------------------------------------------------------------------- #
def _load_pokec(variant: str, root="data", seed=42, protocol: str = "legacy_negative") -> Data:
    raw = os.path.join(root, "raw", "pokec")
    csv = "region_job.csv" if variant == "pokec_z" else "region_job_2.csv"
    rel = "region_job_relationship.txt" if variant == "pokec_z" else "region_job_2_relationship.txt"
    _download(f"{_FAIRGNN}/pokec/{csv}", os.path.join(raw, csv))
    _download(f"{_FAIRGNN}/pokec/{rel}", os.path.join(raw, rel))

    df = pd.read_csv(os.path.join(raw, csv))
    sens_attr, label_attr = "region", "I_am_working_in_field"
    drop = ["user_id", sens_attr, label_attr]
    feat_cols = [c for c in df.columns if c not in drop]
    x = _standardise(df[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).values.astype(np.float32))

    s = (df[sens_attr].values > df[sens_attr].median()).astype(int)
    assert len(np.unique(s)) >= 2, "Sensitive attribute S must contain at least two groups"
    y_raw = df[label_attr].values

    # id -> row index mapping for edges
    idmap = {uid: i for i, uid in enumerate(df["user_id"].values)}
    e = np.loadtxt(os.path.join(raw, rel)).astype(np.int64)
    e = np.array([[idmap[a], idmap[b]] for a, b in e if a in idmap and b in idmap], dtype=np.int64)
    ei = torch.tensor(e.T, dtype=torch.long)
    ei = torch.unique(torch.cat([ei, ei.flip(0)], dim=1), dim=1)

    if protocol == "known_only":
        label_known = torch.tensor(y_raw >= 0, dtype=torch.bool)
        assert label_known.any(), "Pokec dataset contains no labeled nodes"
        assert len(np.unique(s[label_known.cpu().numpy()])) >= 2, (
            "Sensitive attribute S must contain at least two groups on labeled nodes"
        )
        y = np.where(y_raw > 0, 1.0, 0.0).astype(np.float32)
        y_t = torch.tensor(y, dtype=torch.float32)
        train, val, test = _make_splits_known_only(y_t, label_known, seed)
        data = Data(x=x, edge_index=ei, y=y_t,
                    sensitive_attr=torch.tensor(s, dtype=torch.long),
                    train_mask=train, val_mask=val, test_mask=test)
        data.label_known = label_known
    else:  # "legacy_negative"
        y = np.where(y_raw > 0, 1.0, 0.0).astype(np.float32)  # -1/unknown -> 0
        y_t = torch.tensor(y, dtype=torch.float32)
        train, val, test = _make_splits(y_t, seed)
        data = Data(x=x, edge_index=ei, y=y_t,
                    sensitive_attr=torch.tensor(s, dtype=torch.long),
                    train_mask=train, val_mask=val, test_mask=test)
        data.label_known = torch.ones(len(y_t), dtype=torch.bool)

    data.pokec_label_protocol = protocol
    data.fingerprint = get_pokec_fingerprint(data, protocol)
    data.meta = DatasetMeta(variant, "region", "working_in_field", "works in target field")
    return data


def load_pokec_z(root="data", seed=42, protocol: str = "legacy_negative") -> Data:
    return _load_pokec("pokec_z", root, seed, protocol=protocol)


def load_pokec_n(root="data", seed=42, protocol: str = "legacy_negative") -> Data:
    return _load_pokec("pokec_n", root, seed, protocol=protocol)


def load_ogbn_products(root="data", seed=42) -> Data:
    """ogbn-products (Hu et al., 2020): a 2.4M-node, 61M-edge Amazon co-purchase
    graph -- used here as a pure *scalability* stress-test (requires neighbor
    sampling; no full-batch training is possible). It is not a fairness
    benchmark: we construct clearly-documented proxies so the framework can be
    exercised at scale.

      target (proxy):    positive = product belongs to a rare category (bottom
                         20% of the 47 categories by frequency) -- a rare-class
                         detection task in the spirit of fraud.
      sensitive (proxy): high- vs low-degree node (split at the median degree)
                         -- a structural connectivity subgroup.

    Results on this dataset should be read as evidence that TrustFedGNN *scales*,
    not as a demographic-fairness claim.
    """
    from ogb.nodeproppred import PygNodePropPredDataset
    from torch_geometric.utils import degree
    # auto-confirm the download prompt (patch every namespace that bound it)
    import ogb.utils.url as _u
    import ogb.nodeproppred.dataset_pyg as _dp
    _u.decide_download = lambda url: True
    _dp.decide_download = lambda url: True
    # torch >=2.6 defaults weights_only=True, which breaks loading OGB's
    # processed data; OGB is a trusted source, so force full unpickling.
    _orig_load = torch.load
    def _load(*a, **k):
        k.setdefault("weights_only", False)
        return _orig_load(*a, **k)
    torch.load = _load

    ds = PygNodePropPredDataset(name="ogbn-products", root=os.path.join(root, "raw", "ogb"))
    torch.load = _orig_load
    d = ds[0]
    split = ds.get_idx_split()
    y_class = d.y.view(-1)

    # Balanced, LEARNABLE binary target: assign whole categories to the positive
    # class greedily until ~50% of nodes are covered. Because the label is a
    # (coarsened) function of the true product category, a GNN can actually
    # learn it (unlike an arbitrary rare-class split), giving a meaningful AUC.
    counts = torch.bincount(y_class)
    order = torch.argsort(counts, descending=True)
    pos_classes, covered, half = set(), 0, d.num_nodes / 2.0
    for c in order.tolist():
        pos_classes.add(c); covered += int(counts[c])
        if covered >= half:
            break
    y = torch.zeros(d.num_nodes, dtype=torch.float32)
    mask_pos = torch.zeros(len(counts), dtype=torch.bool)
    mask_pos[list(pos_classes)] = True
    y = mask_pos[y_class].float()

    # Structural sensitive proxy: high- vs low-degree node (median split).
    deg = degree(d.edge_index[0], num_nodes=d.num_nodes)
    sensitive = (deg > deg.median()).long()

    n = d.num_nodes
    train = torch.zeros(n, dtype=torch.bool); val = torch.zeros(n, dtype=torch.bool)
    test = torch.zeros(n, dtype=torch.bool)
    train[split["train"]] = True; val[split["valid"]] = True; test[split["test"]] = True

    data = Data(x=d.x, edge_index=d.edge_index, y=y, sensitive_attr=sensitive,
                train_mask=train, val_mask=val, test_mask=test)
    data.meta = DatasetMeta("ogbn_products", "degree(high/low)", "rare_category",
                            "rare-category product")
    return data


_LOADERS = {
    "german": load_german,
    "credit": load_credit,
    "bail": load_bail,
    "elliptic": load_elliptic,
    "pokec_z": load_pokec_z,
    "pokec_n": load_pokec_n,
    "ogbn_products": load_ogbn_products,
    "synthetic": load_synthetic,
}


def load_dataset(name: str, root="data", seed=42, protocol: str = "legacy_negative", **kwargs) -> Data:
    name = name.lower()
    if name not in _LOADERS:
        raise ValueError(f"Unknown dataset '{name}'. Available: {list(_LOADERS)}")
    if name in ("pokec_z", "pokec_n"):
        return _LOADERS[name](root=root, seed=seed, protocol=protocol)
    return _LOADERS[name](root=root, seed=seed)


def dataset_summary(data: Data) -> dict:
    if hasattr(data, "label_known") and not data.label_known.all():
        m = data.label_known
        s = data.sensitive_attr[m]
        y = data.y[m]
    else:
        s = data.sensitive_attr
        y = data.y
    return {
        "nodes": int(data.num_nodes),
        "edges": int(data.edge_index.shape[1]),
        "features": int(data.x.shape[1]),
        "pos_rate": float(y.float().mean()) if len(y) else 0.0,
        "group1_rate": float((s == 1).float().mean()) if len(s) else 0.0,
        "base_rate_gap": float(y[s == 0].float().mean() - y[s == 1].float().mean()) if (s == 0).any() and (s == 1).any() else 0.0,
    }
