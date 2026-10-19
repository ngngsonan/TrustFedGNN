"""Heterogeneous minimax testing error contract for privatised disparity reports (C21).

Evaluates the heterogeneous lower bound R*_het from Theorem 1 / Le Cam:
    S = sum_{k: n0_k > 0, n1_k > 0} sigma_tilde_k^{-2}
    R*_het = max(0, 0.5 * (1 - (tau / 2) * sqrt(S)))

and compares it with the equal-noise approximation:
    R*_equal = max(0, 0.5 * (1 - (tau * sqrt(K)) / (2 * sigma_w)))
where sigma_w = sum_k w_k sigma_tilde_k.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

# Ensure repo root on path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.utils.provenance import build_manifest


def compute_privacy_contract(in_json: str, out_json: str, tau: float = 0.05) -> dict:
    with open(in_json) as f:
        data = json.load(f)

    eps_keys = sorted([k for k in data.keys() if k not in ("manifest", "_manifest")], key=float)
    summary_by_eps = {}

    for eps_str in eps_keys:
        block = data[eps_str]
        tau_val = float(block.get("tau", tau))
        per_round = block.get("per_round", [])

        r_het_list = []
        r_equal_list = []
        s_list = []
        n_excluded_list = []
        all_sigmas = []
        round_records = []

        for r in per_round:
            s_idx = r["seed"]
            t_idx = r["round"]
            sigma_w = float(r.get("sigma_w", 0.0))
            cvs = r.get("client_vectors", [])
            K = len(cvs) if cvs else 10

            valid_sigmas = []
            excluded = 0
            for cv in cvs:
                st = cv.get("sigma_tilde")
                n0 = cv.get("n0", 0)
                n1 = cv.get("n1", 0)
                if st is not None and st > 0 and n0 > 0 and n1 > 0:
                    valid_sigmas.append(float(st))
                else:
                    excluded += 1

            if valid_sigmas:
                S = sum(1.0 / (sig ** 2) for sig in valid_sigmas)
                r_het = max(0.0, 0.5 * (1.0 - (tau_val / 2.0) * math.sqrt(S)))
                all_sigmas.extend(valid_sigmas)
            else:
                S = 0.0
                r_het = 0.0

            if sigma_w > 0:
                r_equal = max(0.0, 0.5 * (1.0 - (tau_val * math.sqrt(K)) / (2.0 * sigma_w)))
            else:
                r_equal = 0.0

            r_het_list.append(r_het)
            r_equal_list.append(r_equal)
            s_list.append(S)
            n_excluded_list.append(excluded)

            round_records.append({
                "seed": s_idx,
                "round": t_idx,
                "S": S,
                "n_excluded": excluded,
                "R_het": r_het,
                "R_equal": r_equal,
                "sigma_w": sigma_w,
            })

        summary_by_eps[eps_str] = {
            "tau": tau_val,
            "R_het_mean": float(np.mean(r_het_list)) if r_het_list else 0.0,
            "R_het_median": float(np.median(r_het_list)) if r_het_list else 0.0,
            "R_het_min": float(np.min(r_het_list)) if r_het_list else 0.0,
            "R_het_max": float(np.max(r_het_list)) if r_het_list else 0.0,
            "R_equal_mean": float(np.mean(r_equal_list)) if r_equal_list else 0.0,
            "R_equal_median": float(np.median(r_equal_list)) if r_equal_list else 0.0,
            "R_equal_min": float(np.min(r_equal_list)) if r_equal_list else 0.0,
            "R_equal_max": float(np.max(r_equal_list)) if r_equal_list else 0.0,
            "sigma_min": float(np.min(all_sigmas)) if all_sigmas else 0.0,
            "sigma_med": float(np.median(all_sigmas)) if all_sigmas else 0.0,
            "sigma_max": float(np.max(all_sigmas)) if all_sigmas else 0.0,
            "n_excluded_mean": float(np.mean(n_excluded_list)) if n_excluded_list else 0.0,
            "rounds_evaluated": len(round_records),
            "per_round": round_records,
        }

    payload = {
        "manifest": build_manifest(experiment="c21_privacy_contract", in_json=in_json),
        "epsilons": eps_keys,
        "results": summary_by_eps,
    }

    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[+] Privacy contract written to {out_json}")
    return payload


def main():
    parser = argparse.ArgumentParser(description="Evaluate heterogeneous privacy contract (C21)")
    parser.add_argument("--in-json", default="results/campaigns/C21a_bfwa_slack_vector.json")
    parser.add_argument("--out-json", default="results/campaigns/C21_privacy_contract.json")
    parser.add_argument("--tau", type=float, default=0.05)
    args = parser.parse_args()
    compute_privacy_contract(args.in_json, args.out_json, tau=args.tau)


if __name__ == "__main__":
    main()
