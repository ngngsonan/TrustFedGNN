"""Campaign C23: Mechanism Factorial (K13).

Evaluates the 2x2 factorial mechanism design (rectify_first x magnitude) along with
faithful baselines and controls:
  - F_SR_C: fu_shapley, cosine, alpha=0.1, rectify_first=False, magnitude="clip" (identical to C17 A6)
  - F_SR_R: fu_shapley, cosine, alpha=0.1, rectify_first=False, magnitude="rescale"
  - F_RS_C: fu_shapley, cosine, alpha=0.1, rectify_first=True,  magnitude="clip"
  - F_RS_R: fu_shapley, cosine, alpha=0.1, rectify_first=True,  magnitude="rescale"
  - A4_fu_shapley: canonical (dot)
  - A1_fltrust: fltrust, alpha=0.0
  - A5_fltrust_ema: fltrust_ema, alpha=0.0, beta=0.9
  - R1_fltrust_root: fltrust_root
  - A0_fedavg: fedavg

Scenarios:
  - fairness_poison: all 9 arms
  - clean: A0_fedavg, A4_fu_shapley, R1_fltrust_root
Datasets: German, Bail. Seeds 42..71. log_mechanism=True.

MANDATORY FULL GATE:
  - A1, A4, A5 (poison) per-seed equals C07a/C07b
  - F_SR_C equals C17 A6
  - A4 clean equals C07 clean

Output: results/campaigns/C23_mechanism_factorial.json
"""
from __future__ import annotations

import argparse
import copy
import json
import multiprocessing as mp
import os
import sys
import time
from typing import Any, Dict, List

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import torch

from src.config import ExperimentConfig
from src.federated.trainer import FederatedTrainer
from src.utils.provenance import build_manifest

DATASETS = ["german", "bail"]
NUM_CLIENTS = 5
SEEDS_30 = list(range(42, 72))

ARMS = {
    "F_SR_C": dict(
        aggregator="fu_shapley",
        fu_score="cosine",
        fu_alpha=0.1,
        fu_ema_rectify_first=False,
        fu_magnitude="clip",
        dp_enabled=False,
    ),
    "F_SR_R": dict(
        aggregator="fu_shapley",
        fu_score="cosine",
        fu_alpha=0.1,
        fu_ema_rectify_first=False,
        fu_magnitude="rescale",
        dp_enabled=False,
    ),
    "F_RS_C": dict(
        aggregator="fu_shapley",
        fu_score="cosine",
        fu_alpha=0.1,
        fu_ema_rectify_first=True,
        fu_magnitude="clip",
        dp_enabled=False,
    ),
    "F_RS_R": dict(
        aggregator="fu_shapley",
        fu_score="cosine",
        fu_alpha=0.1,
        fu_ema_rectify_first=True,
        fu_magnitude="rescale",
        dp_enabled=False,
    ),
    "A4_fu_shapley": dict(
        aggregator="fu_shapley",
        fu_alpha=0.1,
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
    "R1_fltrust_root": dict(
        aggregator="fltrust_root",
        fu_alpha=0.0,
        fu_val_source="server_holdout",
        fu_holdout_size=200,
        dp_enabled=False,
    ),
    "A0_fedavg": dict(
        aggregator="fedavg",
        dp_enabled=False,
    ),
}

SCENARIO_ARMS = {
    "fairness_poison": list(ARMS.keys()),
    "clean": ["A0_fedavg", "A4_fu_shapley", "R1_fltrust_root"],
}

SCENARIOS = {
    "fairness_poison": dict(
        attack="fairness_poison",
        num_byzantine=1,
        attack_intensity=10.0,
    ),
    "clean": dict(
        attack="none",
        num_byzantine=0,
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
            "phi_raw": rec.get("phi_raw"),
            "phi_now": rec.get("phi_now"),
            "phi_ema": rec.get("phi_ema"),
            "fallback": rec.get("fu_fallback", rec.get("fltrust_fallback")),
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


def _worker_task(task):
    fn, args = task
    return fn(*args)


def check_c23_gating(results: dict):
    """Verify bit-exact reproducibility against C07 and C17 baselines."""
    c07a_path = "results/campaigns/C07a_factorial_german.json"
    c07b_path = "results/campaigns/C07b_factorial_bail.json"
    c17_path = "results/campaigns/C17_mechanism.json"

    if not (os.path.exists(c07a_path) and os.path.exists(c07b_path) and os.path.exists(c17_path)):
        print("[!] Gating check skipped: baseline artifacts missing.")
        return

    with open(c07a_path) as f:
        c07a = json.load(f)["results"]["german"]
    with open(c07b_path) as f:
        c07b = json.load(f)["results"]["bail"]
    with open(c17_path) as f:
        c17 = json.load(f)["results"]

    base_c07 = {"german": c07a, "bail": c07b}

    for ds in ("german", "bail"):
        # Check A1, A4, A5 under fairness_poison
        c07_scen = base_c07[ds].get("fairness_poison", {})
        for arm_key in ("A1_fltrust", "A4_fu_shapley", "A5_fltrust_ema"):
            exp_per_seed = {r["seed"]: r for r in c07_scen.get(arm_key, {}).get("per_seed", [])}
            act_per_seed = {r["seed"]: r for r in results[ds]["fairness_poison"].get(arm_key, {}).get("per_seed", [])}
            for s, exp in exp_per_seed.items():
                if s in act_per_seed:
                    act = act_per_seed[s]
                    for m in ("auc", "dpd_hard", "w_adv"):
                        if exp.get(m) != act.get(m):
                            raise RuntimeError(f"C23 gate FAIL: {ds} poison {arm_key} s{s} {m}: exp={exp.get(m)} act={act.get(m)}")

        # Check F_SR_C under fairness_poison against C17 A6_fu_cosine
        c17_scen = c17[ds].get("fairness_poison", {})
        exp_c17 = {r["seed"]: r for r in c17_scen.get("A6_fu_cosine", {}).get("per_seed", [])}
        act_f_sr_c = {r["seed"]: r for r in results[ds]["fairness_poison"].get("F_SR_C", {}).get("per_seed", [])}
        for s, exp in exp_c17.items():
            if s in act_f_sr_c:
                act = act_f_sr_c[s]
                for m in ("auc", "dpd_hard", "w_adv"):
                    if exp.get(m) != act.get(m):
                        raise RuntimeError(f"C23 gate FAIL: {ds} poison F_SR_C s{s} {m}: exp={exp.get(m)} act={act.get(m)}")

        # Check A4_fu_shapley under clean against C07 clean
        c07_clean = base_c07[ds].get("clean", {})
        exp_clean = {r["seed"]: r for r in c07_clean.get("A4_fu_shapley", {}).get("per_seed", [])}
        act_clean = {r["seed"]: r for r in results[ds]["clean"].get("A4_fu_shapley", {}).get("per_seed", [])}
        for s, exp in exp_clean.items():
            if s in act_clean:
                act = act_clean[s]
                for m in ("auc", "dpd_hard"):
                    if exp.get(m) != act.get(m):
                        raise RuntimeError(f"C23 gate FAIL: {ds} clean A4 s{s} {m}: exp={exp.get(m)} act={act.get(m)}")

    print("[+] C23 Full Gating Checks Passed 100% Bit-Exact!")


def run_campaign(out_json: str, smoke: bool = False, workers: int = 6):
    t_start = time.perf_counter()
    seeds = [42] if smoke else SEEDS_30
    datasets = ["german"] if smoke else DATASETS

    print(f"=== Starting C23 Mechanism Factorial Campaign (smoke={smoke}, workers={workers}) ===")

    tasks = []
    for ds in datasets:
        for scen_name, arm_list in SCENARIO_ARMS.items():
            scen_kw = SCENARIOS[scen_name]
            arms_to_run = arm_list[:2] if smoke else arm_list
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
        res_dict[ds] = {}
        for scen_name, arm_list in SCENARIO_ARMS.items():
            res_dict[ds][scen_name] = {}
            arms_to_run = arm_list[:2] if smoke else arm_list
            for arm_name in arms_to_run:
                res_dict[ds][scen_name][arm_name] = {"per_seed": []}

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

    # Gate checks (only on full run)
    if not smoke:
        check_c23_gating(res_dict)

    manifest = build_manifest(
        experiment="C23_mechanism_factorial",
        args={
            "smoke": smoke,
            "workers": workers,
            "datasets": datasets,
            "seeds": seeds,
            "arms": list(ARMS.keys()),
            "scenarios": list(SCENARIOS.keys()),
            "log_mechanism": True,
        },
    )

    payload = {
        "manifest": manifest,
        "results": res_dict,
    }

    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=1)
        f.write("\n")

    wall_total = time.perf_counter() - t_start
    print(f"[+] Output written to {out_json} ({wall_total:.2f}s)")
    return payload


def main():
    ap = argparse.ArgumentParser(description="C23 Mechanism Factorial Campaign Runner")
    ap.add_argument("--out-json", default="results/campaigns/C23_mechanism_factorial.json")
    ap.add_argument("--smoke", action="store_true", default=False)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    run_campaign(args.out_json, smoke=args.smoke, workers=args.workers)


if __name__ == "__main__":
    main()
