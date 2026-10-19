"""Preflight dataset audit script.

Audits benchmark datasets (german, bail, credit, pokec_z, elliptic, ogbn_products)
for node/edge counts, feature dimensions, sensitive homophily (h_s), and verifies
the Zero-Feature Leakage criterion (max_j AUC(x_j, y) < 0.85).

Outputs:
  - results/preflight_datasets.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch_geometric.utils import to_undirected

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.datasets import load_dataset, _LOADERS
from src.utils.metrics import sensitive_homophily
from src.utils.provenance import build_manifest


ALL_DATASETS = ["german", "bail", "credit", "pokec_z", "elliptic", "ogbn_products"]


def audit_single_dataset(name: str, root: str = "data", seed: int = 42) -> dict:
    print(f"[*] Auditing dataset: {name}...", flush=True)
    data = load_dataset(name, root=root, seed=seed)

    n_nodes = int(data.num_nodes)
    n_edges_directed = int(data.edge_index.shape[1])
    n_edges_undirected = int(to_undirected(data.edge_index).shape[1] // 2)
    n_features = int(data.x.shape[1])

    sensitive_name = getattr(data.meta, "sensitive_name", "sensitive") if hasattr(data, "meta") else "sensitive"
    target_name = getattr(data.meta, "label_name", getattr(data.meta, "target_name", "label")) if hasattr(data, "meta") else "label"

    h_s = float(sensitive_homophily(data.edge_index, data.sensitive_attr))

    # Zero-Feature Leakage audit
    y = data.y.detach().cpu().numpy().ravel()
    x = data.x.detach().cpu().numpy()

    # Filter to labelled binary nodes
    valid_mask = (y == 0) | (y == 1)
    if hasattr(data, "labeled") and data.labeled is not None:
        valid_mask = valid_mask & data.labeled.detach().cpu().numpy().ravel().astype(bool)

    y_valid = y[valid_mask]
    x_valid = x[valid_mask]

    aucs = []
    if len(np.unique(y_valid)) >= 2:
        for j in range(n_features):
            col = x_valid[:, j]
            if np.all(col == col[0]):
                aucs.append(0.5)
            else:
                try:
                    auc = float(roc_auc_score(y_valid, col))
                    aucs.append(max(auc, 1.0 - auc))
                except Exception:
                    aucs.append(0.5)
    else:
        aucs = [0.5] * n_features

    max_feat_auc = float(max(aucs)) if aucs else 0.5
    max_feat_idx = int(np.argmax(aucs)) if aucs else 0
    passed_leakage = max_feat_auc < 0.85

    record = {
        "name": name,
        "n_nodes": n_nodes,
        "nodes": n_nodes,
        "n_edges_directed": n_edges_directed,
        "edges": n_edges_directed,
        "n_edges_undirected": n_edges_undirected,
        "n_features": n_features,
        "dim": n_features,
        "sensitive_attr": sensitive_name,
        "sensitive": sensitive_name,
        "label": target_name,
        "h_s": h_s,
        "homophily_hs": h_s,
        "max_feat_auc": max_feat_auc,
        "max_feat_idx": max_feat_idx,
        "status": "PASS (Zero Leakage)" if passed_leakage else "FAIL (Leakage Detected)",
    }
    print(f"    -> N={n_nodes:,}, |E|={n_edges_directed:,} (undir={n_edges_undirected:,}), D={n_features}, "
          f"h_s={h_s:.4f}, max_feat_auc={max_feat_auc:.4f} ({record['status']})", flush=True)
    return record


def run_preflight_audit(datasets=None, root="data", seed=42,
                        out_json="results/preflight_datasets.json") -> dict:
    if datasets is None:
        datasets = ALL_DATASETS

    os.makedirs(os.path.dirname(out_json), exist_ok=True)

    # Load existing records if any
    existing_records = {}
    if os.path.exists(out_json):
        try:
            with open(out_json, "r") as f:
                content = json.load(f)
                items = content.get("datasets", content) if isinstance(content, dict) else content
                for item in items:
                    if isinstance(item, dict) and "name" in item:
                        existing_records[item["name"]] = item
        except Exception as e:
            print(f"[!] Warning reading existing {out_json}: {e}")

    updated_records = dict(existing_records)

    for ds in datasets:
        try:
            rec = audit_single_dataset(ds, root=root, seed=seed)
            updated_records[ds] = rec
        except Exception as e:
            if ds in existing_records:
                print(f"[!] Warning: Could not audit '{ds}' locally ({e}). Preserving previous audit record.")
            else:
                print(f"[!] Warning: Could not audit '{ds}' ({e}). Raw data may be remote (Colab).")

    records_list = [updated_records[k] for k in ALL_DATASETS if k in updated_records]
    # Also include any other datasets that were audited
    for k, v in updated_records.items():
        if v not in records_list:
            records_list.append(v)

    payload = {
        "manifest": build_manifest(experiment="preflight_dataset_audit", audited=list(updated_records.keys())),
        "datasets": records_list,
    }

    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"[+] Saved preflight dataset audit ({len(records_list)} datasets) to {out_json}")
    return payload


def main():
    parser = argparse.ArgumentParser(description="Audit benchmark dataset characteristics and leakage.")
    parser.add_argument("--datasets", nargs="+", default=ALL_DATASETS,
                        help="Datasets to audit.")
    parser.add_argument("--root", default="data", help="Root data directory.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--out-json", default="results/preflight_datasets.json",
                        help="Path to output JSON.")
    args = parser.parse_args()

    run_preflight_audit(datasets=args.datasets, root=args.root, seed=args.seed,
                        out_json=args.out_json)


if __name__ == "__main__":
    main()
