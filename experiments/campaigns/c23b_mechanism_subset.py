"""C23b Mechanism Subset Runner (K13 / C23b).

Runs 4 key arms under fairness_poison on German and Bail across 30 seeds {42..71}:
  1. A4_fu_shapley: canonical bi-objective FU-Shapley (dot product alignment)
  2. R1_fltrust_root: faithful FLTrust root-update formulation (Cao et al. NDSS'21)
  3. A1_fltrust: FLTrust gradient reference
  4. A5_fltrust_ema: FLTrust with EMA momentum smoothing (beta_ema = 0.9)

CRITICAL PROTOCOL REQUIREMENT (Claude Review C23 Item 3):
1. Saves JSON to disk BEFORE gating.
2. Gating checks auc, dpd_hard, w_adv per-seed against C23 (raises error if any seed is missing).
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)

from src.config import ExperimentConfig
from src.federated.trainer import FederatedTrainer
from src.utils.provenance import build_manifest

SEEDS_30 = list(range(42, 72))
DATASETS = ["german", "bail"]
NUM_CLIENTS = 5

ARMS: Dict[str, Dict[str, Any]] = {
    "A4_fu_shapley": dict(
        aggregator="fu_shapley",
        fu_alpha=0.1,
        dp_enabled=False,
    ),
    "R1_fltrust_root": dict(
        aggregator="fltrust_root",
        fu_alpha=0.0,
        fu_val_source="server_holdout",
        fu_holdout_size=200,
        dp_enabled=False,
    ),
    "A1_fltrust": dict(
        aggregator="fltrust",
        fu_alpha=0.0,
        dp_enabled=False,
    ),
    "A5_fltrust_ema": dict(
        aggregator="fltrust_ema",
        fu_alpha=0.0,
        fu_ema_beta=0.9,
        dp_enabled=False,
    ),
}

SCENARIOS: Dict[str, Dict[str, Any]] = {
    "fairness_poison": dict(
        attack="fairness_poison",
        num_byzantine=1,
        attack_intensity=10.0,
    ),
}


def run_one(
    dataset: str,
    arm_name: str,
    arm_kwargs: dict,
    scenario_name: str,
    attack_kwargs: dict,
    seed: int,
    device: str = "cpu",
) -> dict:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)

    cfg = ExperimentConfig.canonical(
        dataset=dataset,
        seed=seed,
        num_clients=NUM_CLIENTS,
        device=device,
        log_mechanism=True,
        **arm_kwargs,
        **attack_kwargs,
    )
    t0 = time.perf_counter()
    trainer = FederatedTrainer(cfg)
    res = trainer.run(verbose=False)
    wall = time.perf_counter() - t0
    fin = res.get("final", res)

    num_byz = cfg.num_byzantine
    adv_weights = []
    for rec in getattr(trainer, "history", []):
        w = rec.get("agg_weights")
        if w is not None and len(w) > 0 and num_byz > 0:
            adv_weights.append(float(sum(w[:num_byz])))
    w_adv = float(np.mean(adv_weights)) if adv_weights else 0.0

    per_round = []
    for rec in getattr(trainer, "history", []):
        pr = {
            "round": rec.get("round"),
            "w": rec.get("agg_weights"),
            "phi_now": rec.get("phi_now"),
            "phi_ema": rec.get("phi_ema"),
            "fallback": rec.get("fu_fallback", rec.get("fltrust_fallback")),
            "w_adv_round": rec.get("w_adv_round"),
            "I_adv": rec.get("I_adv"),
            "agg_norm": rec.get("agg_norm"),
            "cos_agg_g_task": rec.get("cos_agg_g_task"),
            "cos_agg_g_fair": rec.get("cos_agg_g_fair"),
            "t_server_s": rec.get("t_server_s"),
            "t_clients_s": rec.get("t_clients_s"),
        }
        per_round.append(pr)

    return {
        "dataset": dataset,
        "scenario": scenario_name,
        "arm": arm_name,
        "seed": seed,
        "auc": float(fin["auc"]),
        "dpd_hard": float(fin["dpd_hard"]),
        "eod": float(fin.get("eod", 0.0)),
        "w_adv": w_adv,
        "wall_s": float(wall),
        "per_round": per_round,
    }


def _worker_task(args: Tuple) -> dict:
    fn, a = args
    return fn(*a)


def check_c23b_gating(results: dict):
    """Strict per-seed bit-exact gating against C23 baseline artifact.

    Must fail loudly if any seed is missing or metric differs.
    """
    c23_path = os.path.join(REPO_ROOT, "results/campaigns/C23_mechanism_factorial.json")
    if not os.path.exists(c23_path):
        raise FileNotFoundError(f"C23 baseline artifact not found for gating: {c23_path}")

    with open(c23_path) as f:
        c23_data = json.load(f)

    c23_res = c23_data.get("results", {})
    expected_seeds = set(SEEDS_30)

    for ds in DATASETS:
        if ds not in c23_res or ds not in results:
            raise RuntimeError(f"C23b gate error: missing dataset {ds}")

        for arm in ARMS:
            c23_arm_data = c23_res[ds].get("fairness_poison", {}).get(arm, {})
            c23b_arm_data = results[ds].get("fairness_poison", {}).get(arm, {})

            exp_seeds_dict = {r["seed"]: r for r in c23_arm_data.get("per_seed", [])}
            act_seeds_dict = {r["seed"]: r for r in c23b_arm_data.get("per_seed", [])}

            # Must contain all 30 seeds
            if set(act_seeds_dict.keys()) != expected_seeds:
                missing = expected_seeds - set(act_seeds_dict.keys())
                raise RuntimeError(f"C23b gate FAIL: {ds} {arm} missing seeds: {sorted(missing)}")
            if set(exp_seeds_dict.keys()) != expected_seeds:
                missing = expected_seeds - set(exp_seeds_dict.keys())
                raise RuntimeError(f"C23 baseline missing seeds for {ds} {arm}: {sorted(missing)}")

            # Check per-seed bit-exact match
            for s in sorted(expected_seeds):
                exp = exp_seeds_dict[s]
                act = act_seeds_dict[s]
                for m in ("auc", "dpd_hard", "w_adv"):
                    if exp.get(m) != act.get(m):
                        raise RuntimeError(
                            f"C23b gate FAIL: {ds} {arm} seed {s} metric '{m}' mismatch: "
                            f"exp={exp.get(m)} vs act={act.get(m)}"
                        )

    print("[+] C23b Gating Checks Passed 100% Bit-Exact against C23 (0 deviations, 30/30 seeds per arm)!")


def run_campaign(out_json: str, smoke: bool = False, workers: int = 6):
    t_start = time.perf_counter()
    seeds = [42] if smoke else SEEDS_30
    datasets = ["german"] if smoke else DATASETS

    print(f"=== Starting C23b Mechanism Subset Campaign (smoke={smoke}, workers={workers}) ===")

    tasks = []
    for ds in datasets:
        scen_name = "fairness_poison"
        scen_kw = SCENARIOS[scen_name]
        arms_to_run = list(ARMS.keys())[:2] if smoke else list(ARMS.keys())
        for arm_name in arms_to_run:
            arm_kw = ARMS[arm_name]
            for s in seeds:
                tasks.append((run_one, (ds, arm_name, arm_kw, scen_name, scen_kw, s, "cpu")))

    print(f"[*] Dispatching {len(tasks)} tasks across {workers} workers...")
    pool = mp.Pool(processes=workers)
    results_flat = pool.map(_worker_task, tasks)
    pool.close()
    pool.join()

    # Structure results hierarchy
    res_dict: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for ds in datasets:
        res_dict[ds] = {"fairness_poison": {}}
        arms_to_run = list(ARMS.keys())[:2] if smoke else list(ARMS.keys())
        for arm_name in arms_to_run:
            res_dict[ds]["fairness_poison"][arm_name] = {"per_seed": []}

    for r in results_flat:
        ds = r["dataset"]
        scen = r["scenario"]
        arm = r["arm"]
        res_dict[ds][scen][arm]["per_seed"].append(r)

    # Compute summaries
    for ds in datasets:
        for scen in res_dict[ds]:
            for arm in res_dict[ds][scen]:
                recs = res_dict[ds][scen][arm]["per_seed"]
                res_dict[ds][scen][arm]["auc_mean"] = float(np.mean([x["auc"] for x in recs]))
                res_dict[ds][scen][arm]["dpd_hard_mean"] = float(np.mean([x["dpd_hard"] for x in recs]))
                res_dict[ds][scen][arm]["eod_mean"] = float(np.mean([x["eod"] for x in recs]))
                res_dict[ds][scen][arm]["w_adv_mean"] = float(np.mean([x["w_adv"] for x in recs]))

    manifest = build_manifest(
        experiment="C23b_mechanism_subset",
        args={
            "smoke": smoke,
            "workers": workers,
            "datasets": datasets,
            "seeds": seeds,
            "arms": list(ARMS.keys()),
            "scenario": "fairness_poison",
            "log_mechanism": True,
        },
    )

    payload = {
        "manifest": manifest,
        "results": res_dict,
    }

    # CRITICAL: Ghi JSON TRƯỚC khi gate!
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=1)
        f.write("\n")
    print(f"[+] Output written to {out_json} BEFORE gating.")

    # Gate checks (only on full run)
    if not smoke:
        check_c23b_gating(res_dict)

    wall_total = time.perf_counter() - t_start
    print(f"[+] Campaign C23b complete ({wall_total:.2f}s)")
    return payload


def main():
    ap = argparse.ArgumentParser(description="C23b Mechanism Subset Campaign Runner")
    ap.add_argument("--out-json", default="results/campaigns/C23b_mechanism_subset.json")
    ap.add_argument("--smoke", action="store_true", default=False)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    run_campaign(args.out_json, smoke=args.smoke, workers=args.workers)


if __name__ == "__main__":
    main()
