"""C26: isolate the fairness reference in FU-Alignment.

Run canonical FU-Alignment with ``fu_alpha=0`` under the C23
fairness-poisoning protocol. The paired ``fu_alpha=0.1`` records remain in
the frozen C23 artifact, so this is the smallest run that closes the
fairness-reference attribution gap.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from typing import Any, Dict

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from experiments.campaigns.c23_mechanism_factorial import run_one
from src.utils.provenance import build_manifest


DATASETS = ["german", "bail"]
SEEDS_30 = list(range(42, 72))
ARM = dict(aggregator="fu_shapley", fu_alpha=0.0, dp_enabled=False)
ATTACK = dict(attack="fairness_poison", num_byzantine=1, attack_intensity=10.0)


def _worker(task):
    return run_one(*task)


def run_campaign(out_json: str, smoke: bool = False, workers: int = 6) -> dict:
    seeds = [42] if smoke else SEEDS_30
    datasets = ["german"] if smoke else DATASETS
    tasks = [
        (dataset, "FU_alpha0", ARM, "fairness_poison", ATTACK, seed, "cpu")
        for dataset in datasets for seed in seeds
    ]
    started = time.perf_counter()
    with mp.Pool(processes=workers) as pool:
        records = pool.map(_worker, tasks)

    results: Dict[str, Dict[str, Any]] = {}
    for dataset in datasets:
        per_seed = sorted(
            (r for r in records if r["dataset"] == dataset),
            key=lambda r: r["seed"],
        )
        results[dataset] = {
            "per_seed": per_seed,
            **{
                f"{metric}_mean": float(np.mean([r[metric] for r in per_seed]))
                for metric in ("auc", "dpd_hard", "eod", "w_adv")
            },
        }

    payload = {
        "manifest": build_manifest(
            experiment="C26_fu_alpha0",
            args={
                "smoke": smoke,
                "workers": workers,
                "datasets": datasets,
                "seeds": seeds,
                "arm": ARM,
                "scenario": ATTACK,
                "paired_reference_artifact": "results/campaigns/C23_mechanism_factorial.json",
            },
        ),
        "results": results,
    }
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=1)
        f.write("\n")
    print(f"[+] C26 wrote {out_json} in {time.perf_counter() - started:.2f}s")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-json", default="results/campaigns/C26_fu_alpha0.json")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    run_campaign(args.out_json, smoke=args.smoke, workers=args.workers)


if __name__ == "__main__":
    main()
