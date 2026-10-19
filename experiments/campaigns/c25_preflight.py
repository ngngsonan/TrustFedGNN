"""Preflight statistics for Pokec-z under legacy_negative vs known_only protocols (K14 / C25).
"""
import json
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)

import numpy as np
import torch
from src.data.datasets import load_dataset
from src.utils.provenance import build_manifest


def main():
    print("Generating Pokec-z preflight statistics...")
    data_kn = load_dataset("pokec_z", protocol="known_only")
    data_leg = load_dataset("pokec_z", protocol="legacy_negative")

    known_mask = data_kn.label_known.bool()
    num_total_nodes = int(data_kn.num_nodes)
    num_known_nodes = int(known_mask.sum().item())
    known_frac = float(num_known_nodes / num_total_nodes)

    # Known only statistics
    y_known = data_kn.y[known_mask].float()
    s_known = data_kn.sensitive_attr[known_mask].float()
    pos_rate_known = float(y_known.mean().item())
    group1_rate_known = float(s_known.mean().item())

    # Legacy negative statistics
    y_leg = data_leg.y.float()
    s_leg = data_leg.sensitive_attr.float()
    pos_rate_leg = float(y_leg.mean().item())
    group1_rate_leg = float(s_leg.mean().item())

    # Split sizes in known_only
    train_known = int(data_kn.train_mask.sum().item())
    val_known = int(data_kn.val_mask.sum().item())
    test_known = int(data_kn.test_mask.sum().item())

    preflight = {
        "manifest": build_manifest(
            experiment="C25_preflight_pokec_known",
            args={"dataset": "pokec_z", "protocols": ["legacy_negative", "known_only"]},
        ),
        "total_nodes": num_total_nodes,
        "known_nodes": num_known_nodes,
        "known_fraction": known_frac,
        "known_splits": {
            "train": train_known,
            "val": val_known,
            "test": test_known,
            "sum": train_known + val_known + test_known,
        },
        "known_only": {
            "positive_rate": pos_rate_known,
            "negative_rate": 1.0 - pos_rate_known,
            "group1_rate": group1_rate_known,
            "group0_rate": 1.0 - group1_rate_known,
            "fingerprint": getattr(data_kn, "fingerprint", None),
        },
        "legacy_negative": {
            "positive_rate": pos_rate_leg,
            "negative_rate": 1.0 - pos_rate_leg,
            "group1_rate": group1_rate_leg,
            "group0_rate": 1.0 - group1_rate_leg,
            "fingerprint": getattr(data_leg, "fingerprint", None),
        },
    }

    out_path = os.path.join(REPO_ROOT, "results/campaigns/C25_preflight_pokec_known.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(preflight, f, indent=2)

    print(f"Preflight saved to {out_path}:")
    print(f"  Total nodes: {num_total_nodes}, Known: {num_known_nodes} ({known_frac*100:.2f}%)")
    print(f"  Known-only positive rate: {pos_rate_known:.4f}, Group 1 rate: {group1_rate_known:.4f}")
    print(f"  Legacy positive rate: {pos_rate_leg:.4f}, Group 1 rate: {group1_rate_leg:.4f}")


if __name__ == "__main__":
    main()
