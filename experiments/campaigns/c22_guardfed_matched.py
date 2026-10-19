"""C22 Campaign Runner: GuardFed Parameter-Matched Baseline (K13).

Evaluates the GuardFed-style trust-score aggregator across a 14-configuration
grid with parameter tuning and confirmatory evaluation:
  (1) Tune: seeds 1000..1004 across 14 configs (gamma in {0, 0.001, 0.003, 0.01, 0.03, 0.06, 0.1}
      x tau in {1, 10}) on three scenarios:
      - A-German (Part A design, German, canonical C07)
      - A-Bail (Part A design, Bail, canonical C07)
      - B-Bail (Part B design, Bail stealth C05 across 4 corruption ratios)
      Selection criterion: max mean(val_auc - val_dpd_hard) on union of client val masks.
      Tie-breaker: lower mean w_adv.
  (2) Confirm: run selected config on:
      - Part A German: seeds 42..71 (30 seeds)
      - Part A Bail: seeds 42..71 (30 seeds)
      - Part B Bail: seeds 42..51 (10 seeds, 4 ratios)
  (3) fltrust_root: run on Part B Bail stealth: seeds 42..51 (10 seeds, 4 ratios).

Outputs:
  - results/campaigns/C22_guardfed_matched.json
"""
from __future__ import annotations

import argparse
import copy
import json
import multiprocessing as mp
import os
import sys
import time
from typing import Any, Dict, List, Tuple

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import torch

from src.config import ExperimentConfig
from src.federated.trainer import FederatedTrainer
from src.federated.client import load_flat_state
from src.federated.aggregation import compute_guardfed_holdout
from src.utils.metrics import all_metrics
from src.utils.provenance import build_manifest
from experiments.revision.adaptive_poisoner import evaluate_adaptive_run

GAMMA_VALUES = [0.0, 0.001, 0.003, 0.01, 0.03, 0.06, 0.1]
TAU_VALUES = [1.0, 10.0]

GRID: List[Dict[str, Any]] = [
    {"name": f"gs_tau{int(tau)}_gam{gamma}", "gs_tau": tau, "gs_gamma": gamma}
    for tau in TAU_VALUES
    for gamma in GAMMA_VALUES
]

SEEDS_TUNE = [1000, 1001, 1002, 1003, 1004]
SEEDS_CONFIRM_A = list(range(42, 72))
SEEDS_CONFIRM_B = list(range(42, 52))
PART_B_RATIOS = [0.1, 0.2, 0.3, 0.4]


def evaluate_global_val(trainer: FederatedTrainer) -> Dict[str, float]:
    """Evaluate global model on the union of client validation masks (B.2)."""
    load_flat_state(trainer.ref_model, trainer.global_flat.to(trainer.device))
    trainer.ref_model.eval()
    ys, ps, ss = [], [], []
    for d in trainer.clients_data:
        d = d.to(trainer.device)
        mask = d.val_mask
        if mask.sum() == 0:
            continue
        pred = trainer.ref_model(d.x, d.edge_index, d.sensitive_attr)[mask]
        sm = d.sensitive_attr[mask]
        ys.append(d.y[mask].cpu())
        ps.append(pred.cpu())
        ss.append(sm.cpu())
    if not ys:
        return {"auc": float("nan"), "dpd_hard": float("nan"), "eod": float("nan")}
    return all_metrics(torch.cat(ys), torch.cat(ps), torch.cat(ss))


def run_part_a_one(dataset: str, arm_name: str, arm_kwargs: dict, seed: int, device: str = "cpu") -> dict:
    """Run Part (a): C07 design cell with GuardFed aggregator and validation scoring."""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)

    cfg = ExperimentConfig.canonical(
        dataset=dataset,
        seed=seed,
        num_clients=5,
        rounds=20,
        local_epochs=3,
        dirichlet_alpha=0.3,
        device=device,
        model="trustfedgnn",
        attack="fairness_poison",
        num_byzantine=1,
        attack_intensity=10.0,
        dp_enabled=False,
        fu_val_source="server_holdout",
        fu_holdout_size=200,
        aggregator="guardfed_style",
        log_mechanism=True,
        **arm_kwargs,
    )
    t0 = time.perf_counter()
    trainer = FederatedTrainer(cfg)
    res = trainer.run(verbose=False)
    wall = time.perf_counter() - t0
    fin = res.get("final", res)
    val = evaluate_global_val(trainer)

    adv_weights = []
    empty_s_count = 0
    total_rounds = 0
    for rec in getattr(trainer, "history", []):
        total_rounds += 1
        w = rec.get("agg_weights")
        if w is not None and len(w) > 0 and cfg.num_byzantine > 0:
            adv_weights.append(float(sum(w[:cfg.num_byzantine])))
            if all(v == 0.0 for v in w):
                empty_s_count += 1
    w_adv = float(np.mean(adv_weights)) if adv_weights else 0.0
    empty_s_frac = float(empty_s_count / total_rounds) if total_rounds > 0 else 0.0

    return {
        "dataset": dataset,
        "arm": arm_name,
        "seed": seed,
        "auc": float(fin["auc"]),
        "dpd_hard": float(fin["dpd_hard"]),
        "eod": float(fin.get("eod", 0.0)),
        "val_auc": float(val.get("auc", float("nan"))),
        "val_dpd_hard": float(val.get("dpd_hard", float("nan"))),
        "val_eod": float(val.get("eod", float("nan"))),
        "w_adv": w_adv,
        "empty_s_frac": empty_s_frac,
        "wall_s": float(wall),
        "config": {
            "model": cfg.model,
            "aggregator": cfg.aggregator,
            "gs_tau": getattr(cfg, "gs_tau", 1.0),
            "gs_gamma": getattr(cfg, "gs_gamma", 0.0),
            "dp_enabled": cfg.dp_enabled,
            "local_epochs": cfg.local_epochs,
            "rounds": cfg.rounds,
            "num_clients": cfg.num_clients,
            "num_byzantine": cfg.num_byzantine,
            "attack": cfg.attack,
            "attack_intensity": cfg.attack_intensity,
            "dirichlet_alpha": cfg.dirichlet_alpha,
            "fu_val_source": cfg.fu_val_source,
        },
        "manifest": build_manifest(dataset=dataset, rounds=cfg.rounds, device=device),
    }


def run_part_b_one(arm_name: str, arm_kwargs: dict, ratio: float, seed: int, device: str = "cpu") -> dict:
    """Run Part (b): C05 design stealth cell using evaluate_adaptive_run harness."""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    rec = evaluate_adaptive_run(
        aggregator="guardfed_style",
        byz_ratio=ratio,
        seed=seed,
        rounds=15,
        dataset="bail",
        num_clients=10,
        device=device,
        extra_cfg=dict(log_mechanism=True, **arm_kwargs),
    )
    rec["arm"] = arm_name
    rec["dataset"] = "bail"
    rec["manifest"] = build_manifest(dataset="bail", rounds=15, device=device)
    return rec


def run_fltrust_root_part_b_one(ratio: float, seed: int, device: str = "cpu") -> dict:
    """Run fltrust_root on Part (b) Bail stealth (seeds 42..51, 4 ratios)."""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    rec = evaluate_adaptive_run(
        aggregator="fltrust_root",
        byz_ratio=ratio,
        seed=seed,
        rounds=15,
        dataset="bail",
        num_clients=10,
        device=device,
        extra_cfg=dict(log_mechanism=True),
    )
    rec["arm"] = "fltrust_root"
    rec["dataset"] = "bail"
    rec["manifest"] = build_manifest(dataset="bail", rounds=15, device=device)
    return rec


def check_c05_gating():
    """Verify bit-exact reproduction of C05 fu_shapley and fedavg before running Part B."""
    c05_path = "results/campaigns/C05_adaptive_poisoner.json"
    if not os.path.exists(c05_path):
        raise FileNotFoundError(f"Missing {c05_path}")
    with open(c05_path) as f:
        c05 = json.load(f)
    torch.set_num_threads(2)
    for agg in ("fedavg", "fu_shapley"):
        for br in (0.2, 0.4):
            for seed in (42, 43):
                matches = [r for r in c05["records"] if r["aggregator"] == agg and abs(r["byz_ratio"] - br) < 1e-5 and r["seed"] == seed]
                assert len(matches) == 1, f"Missing C05 record for {agg} br={br} s={seed}"
                exp = matches[0]
                act = evaluate_adaptive_run(agg, br, seed=seed, rounds=15, dataset="bail", num_clients=10, device="cpu")
                for m in ("auc", "dpd_hard", "w_adv"):
                    if exp.get(m) != act.get(m):
                        raise RuntimeError(
                            f"C05 gate mismatch for {agg} br={br} s={seed} {m}: exp={exp.get(m)} act={act.get(m)}"
                        )
    print("[+] C05 Part B gating passed bit-exact!")


def pick_best_config(records_by_cfg: Dict[str, List[dict]]) -> Tuple[str, Dict[str, Any]]:
    """Pick configuration maximizing mean(val_auc - val_dpd_hard). Tie-breaker: lower mean w_adv."""
    best_name = None
    best_score = -float("inf")
    best_w = float("inf")
    summary = {}
    for name, recs in records_by_cfg.items():
        s = float(np.mean([r["val_auc"] - r["val_dpd_hard"] for r in recs]))
        w_vals = [r["w_adv"] for r in recs if np.isfinite(r["w_adv"])]
        w = float(np.mean(w_vals)) if w_vals else 0.0
        empty_s = float(np.mean([r.get("empty_s_frac", 0.0) for r in recs]))
        summary[name] = {
            "score": s,
            "mean_val_auc": float(np.mean([r["val_auc"] for r in recs])),
            "mean_val_dpd": float(np.mean([r["val_dpd_hard"] for r in recs])),
            "mean_w_adv": w,
            "empty_s_frac": empty_s,
            "n_recs": len(recs),
        }
        if s > best_score or (abs(s - best_score) < 1e-9 and w < best_w):
            best_score = s
            best_w = w
            best_name = name
    assert best_name is not None
    return best_name, summary


def _worker_task(task):
    fn, args = task
    return fn(*args)


def run_campaign(out_json: str, smoke: bool = False, workers: int = 6):
    t_start = time.perf_counter()
    print(f"=== Starting C22 GuardFed Matched Campaign (smoke={smoke}, workers={workers}) ===")

    # 0. Gate C05 Part B harness
    check_c05_gating()

    tune_seeds = [1000] if smoke else SEEDS_TUNE
    conf_seeds_a = [42] if smoke else SEEDS_CONFIRM_A
    conf_seeds_b = [42] if smoke else SEEDS_CONFIRM_B
    part_b_ratios = [0.2] if smoke else PART_B_RATIOS
    grid = GRID[:2] if smoke else GRID

    pool = mp.Pool(processes=workers)

    # 1. TUNE
    print(f"[*] Running Tune Phase across {len(grid)} configs...")
    tune_tasks_a_german = []
    tune_tasks_a_bail = []
    tune_tasks_b_bail = []

    for cfg_item in grid:
        name = cfg_item["name"]
        kw = {"gs_tau": cfg_item["gs_tau"], "gs_gamma": cfg_item["gs_gamma"]}
        for s in tune_seeds:
            tune_tasks_a_german.append((run_part_a_one, ("german", name, kw, s, "cpu")))
            tune_tasks_a_bail.append((run_part_a_one, ("bail", name, kw, s, "cpu")))
            for r in part_b_ratios:
                tune_tasks_b_bail.append((run_part_b_one, (name, kw, r, s, "cpu")))

    print(f"    Dispatching {len(tune_tasks_a_german)} A-German, {len(tune_tasks_a_bail)} A-Bail, {len(tune_tasks_b_bail)} B-Bail runs...")
    res_tune_a_ger = pool.map(_worker_task, tune_tasks_a_german)
    res_tune_a_bail = pool.map(_worker_task, tune_tasks_a_bail)
    res_tune_b_bail = pool.map(_worker_task, tune_tasks_b_bail)

    # Group tune results by config name
    by_cfg_a_ger: Dict[str, List[dict]] = {c["name"]: [] for c in grid}
    for r in res_tune_a_ger:
        by_cfg_a_ger[r["arm"]].append(r)

    by_cfg_a_bail: Dict[str, List[dict]] = {c["name"]: [] for c in grid}
    for r in res_tune_a_bail:
        by_cfg_a_bail[r["arm"]].append(r)

    by_cfg_b_bail: Dict[str, List[dict]] = {c["name"]: [] for c in grid}
    for r in res_tune_b_bail:
        by_cfg_b_bail[r["arm"]].append(r)

    best_a_ger, summ_a_ger = pick_best_config(by_cfg_a_ger)
    best_a_bail, summ_a_bail = pick_best_config(by_cfg_a_bail)
    best_b_bail, summ_b_bail = pick_best_config(by_cfg_b_bail)

    cfg_dict = {c["name"]: c for c in grid}

    selection = {
        "A-German": {
            "selected": best_a_ger,
            "config": cfg_dict[best_a_ger],
            "grid_summary": summ_a_ger,
        },
        "A-Bail": {
            "selected": best_a_bail,
            "config": cfg_dict[best_a_bail],
            "grid_summary": summ_a_bail,
        },
        "B-Bail": {
            "selected": best_b_bail,
            "config": cfg_dict[best_b_bail],
            "grid_summary": summ_b_bail,
        },
    }
    print(f"[+] Selection Phase complete:")
    print(f"    A-German: {best_a_ger} (score={summ_a_ger[best_a_ger]['score']:.4f})")
    print(f"    A-Bail:   {best_a_bail} (score={summ_a_bail[best_a_bail]['score']:.4f})")
    print(f"    B-Bail:   {best_b_bail} (score={summ_b_bail[best_b_bail]['score']:.4f})")

    # 2. CONFIRM
    print("[*] Running Confirm Phase...")
    confirm_tasks_a_ger = []
    sel_ger_kw = {"gs_tau": cfg_dict[best_a_ger]["gs_tau"], "gs_gamma": cfg_dict[best_a_ger]["gs_gamma"]}
    for s in conf_seeds_a:
        confirm_tasks_a_ger.append((run_part_a_one, ("german", best_a_ger, sel_ger_kw, s, "cpu")))

    confirm_tasks_a_bail = []
    sel_bail_kw = {"gs_tau": cfg_dict[best_a_bail]["gs_tau"], "gs_gamma": cfg_dict[best_a_bail]["gs_gamma"]}
    for s in conf_seeds_a:
        confirm_tasks_a_bail.append((run_part_a_one, ("bail", best_a_bail, sel_bail_kw, s, "cpu")))

    confirm_tasks_b_bail = []
    sel_b_kw = {"gs_tau": cfg_dict[best_b_bail]["gs_tau"], "gs_gamma": cfg_dict[best_b_bail]["gs_gamma"]}
    for s in conf_seeds_b:
        for r in part_b_ratios:
            confirm_tasks_b_bail.append((run_part_b_one, (best_b_bail, sel_b_kw, r, s, "cpu")))

    # 3. fltrust_root on Part B stealth
    print("[*] Running fltrust_root stealth on Part B...")
    tasks_fltrust_root_b = []
    for s in conf_seeds_b:
        for r in part_b_ratios:
            tasks_fltrust_root_b.append((run_fltrust_root_part_b_one, (r, s, "cpu")))

    res_confirm_a_ger = pool.map(_worker_task, confirm_tasks_a_ger)
    res_confirm_a_bail = pool.map(_worker_task, confirm_tasks_a_bail)
    res_confirm_b_bail = pool.map(_worker_task, confirm_tasks_b_bail)
    res_fltrust_root_b = pool.map(_worker_task, tasks_fltrust_root_b)

    pool.close()
    pool.join()

    manifest = build_manifest(
        experiment="C22_guardfed_matched",
        args={
            "smoke": smoke,
            "workers": workers,
            "tune_seeds": tune_seeds,
            "confirm_seeds_a": conf_seeds_a,
            "confirm_seeds_b": conf_seeds_b,
            "part_b_ratios": part_b_ratios,
            "grid_size": len(grid),
        },
    )

    payload = {
        "manifest": manifest,
        "selection": selection,
        "tune": {
            "A-German": res_tune_a_ger,
            "A-Bail": res_tune_a_bail,
            "B-Bail": res_tune_b_bail,
        },
        "confirm": {
            "part_a": {
                "german": res_confirm_a_ger,
                "bail": res_confirm_a_bail,
            },
            "part_b": res_confirm_b_bail,
        },
        "fltrust_root_stealth": res_fltrust_root_b,
    }

    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=1)
        f.write("\n")

    wall_total = time.perf_counter() - t_start
    print(f"[+] Output written to {out_json} ({wall_total:.2f}s)")
    return payload


def main():
    ap = argparse.ArgumentParser(description="C22 GuardFed Matched Campaign Runner")
    ap.add_argument("--out-json", default="results/campaigns/C22_guardfed_matched.json")
    ap.add_argument("--smoke", action="store_true", default=False)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    run_campaign(args.out_json, smoke=args.smoke, workers=args.workers)


if __name__ == "__main__":
    main()
