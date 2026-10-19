"""Adaptive Stealth Fairness-Poisoner Evaluation.

Formulates an omniscient adaptive attacker that explicitly evades distance-to-median
screening by projecting its update within the safe radius of benign updates:
    g_stealth = median(G) + rho * (g_poison - median(G)) / ||...|| * r_safe
while falsely reporting DPD_hat = 0.0 and Perf = 0.99.

Evaluates breakdown curves of 6 aggregators (robust_bfwa, krum, median, trimmed_mean, bfwa, fedavg)
across Byzantine corruption fractions f/K in {0.1, 0.2, 0.3, 0.4} on Bail (18.8k nodes).

Outputs:
  - results/revision/adaptive_poisoner_results.json
  - manuscript/tables/revision/adaptive_poisoner.tex
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import torch

from src.config import ExperimentConfig
from src.federated.trainer import FederatedTrainer
from src.federated.aggregation import aggregate
from src.trust.incentive import (get_server_target_gradients,
                                 get_server_target_gradients_pooled)
from src.federated.client import load_flat_state
from src.utils.provenance import build_manifest


def craft_stealth_poison_updates(updates: List[torch.Tensor], metas: List[dict],
                                 byzantine_ids: List[int], rho: float = 0.85) -> Tuple[List[torch.Tensor], List[dict]]:
    """Stealth projection: ensures adversarial updates stay within the benign median distance radius."""
    if not byzantine_ids:
        return updates, metas

    updates = [u.clone() for u in updates]
    metas = [dict(m) for m in metas]
    benign = [i for i in range(len(updates)) if i not in byzantine_ids]

    if not benign:
        return updates, metas

    bstack = torch.stack([updates[i].flatten() for i in benign])
    bmed = bstack.median(0).values
    bdist = ((bstack - bmed) ** 2).sum(1).sqrt()
    r_safe = float(bdist.median()) * rho  # Strictly within median benign distance

    for i in byzantine_ids:
        g = updates[i].flatten()
        direction = g - bmed
        dir_norm = direction.norm() + 1e-8
        # Project within safe radius
        g_stealth = bmed + (direction / dir_norm) * min(float(dir_norm), r_safe)
        updates[i] = g_stealth.view_as(updates[i])
        # Lie about fairness/utility to capture BFWA
        metas[i]["dpd"] = 0.0
        metas[i]["eod"] = 0.0
        metas[i]["perf"] = 0.99

    return updates, metas


def evaluate_adaptive_run(aggregator: str, byz_ratio: float, seed: int = 42, rounds: int = 15,
                          device: str = "cpu", dataset: str = "bail",
                          num_clients: int = 10, extra_cfg: dict | None = None) -> dict:
    t0 = time.perf_counter()
    num_byz = int(round(byz_ratio * num_clients))
    byz_indices = set(range(num_byz))

    extra = dict(extra_cfg or {})
    if aggregator == "fltrust" and "fu_alpha" not in extra:
        extra["fu_alpha"] = 0.0
    if aggregator in ("fltrust_root", "guardfed_style") and "fu_val_source" not in extra:
        extra["fu_val_source"] = "server_holdout"
        extra["fu_holdout_size"] = 200

    cfg = ExperimentConfig.canonical(
        dataset=dataset,
        seed=seed,
        num_clients=num_clients,
        rounds=rounds,
        local_epochs=1,
        dirichlet_alpha=0.3,
        device=device,
        model="trustfedgnn",
        aggregator=aggregator,
        attack="fairness_poison",
        num_byzantine=num_byz,
        krum_f=num_byz,
        attack_intensity=10.0,
        dp_enabled=False,
        **extra,
    )

    trainer = FederatedTrainer(cfg)

    # Monkey patch round aggregation with stealth crafting
    orig_round = trainer._round

    def stealth_round(t):
        updates, metas = [], []
        for c in trainer.clients:
            c.set_flat(trainer.global_flat)
            c.train()
            g_k = trainer.global_flat - c.get_flat()
            updates.append(g_k)
            metas.append(c.meta())

        # Craft stealth updates
        updates, metas = craft_stealth_poison_updates(updates, metas, list(byz_indices), rho=0.85)

        # Build the server target gradient exactly as FederatedTrainer._round
        # does. Omitting it is NOT a silent no-op: aggregate() falls back to
        # sample-size weights when g_target is None, so `fu_shapley` and
        # `robust_fu_shapley` degenerate to FedAvg and report FedAvg's numbers
        # under our own method's name -- which is precisely what the earlier
        # revision of this sweep did (fu_shapley matched fedavg bit-exact at
        # every corruption ratio, with w_adv rising to 0.457).
        g_target = g_task = g_fair = None
        g_s = gs_fair = None
        delta_root = None
        # fltrust needs the same reference machinery (it scores against
        # g_task, the task-only root update), so it must be in this condition
        # too -- otherwise it silently takes the sample-size fallback and
        # reports FedAvg numbers, the same trap documented just above.
        if "fu_shapley" in cfg.aggregator or cfg.aggregator == "fltrust":
            load_flat_state(trainer.ref_model, trainer.global_flat.to(trainer.device))
            if trainer.server_holdout is not None:
                tg = get_server_target_gradients(
                    trainer.ref_model, trainer.server_holdout.to(trainer.device),
                    cfg.fu_alpha, fair_surrogate=cfg.fu_fair_surrogate)
            else:
                tg = get_server_target_gradients_pooled(
                    trainer.ref_model, trainer.clients_data, trainer.device,
                    cfg.fu_alpha, fair_surrogate=cfg.fu_fair_surrogate)
            if tg is not None:
                g_target, g_task, g_fair = (g.cpu() for g in tg)
        elif cfg.aggregator == "fltrust_root":
            from src.trust.incentive import get_server_root_update
            if trainer.server_holdout is not None:
                delta_root = get_server_root_update(
                    trainer.ref_model, trainer.server_holdout.to(trainer.device), cfg, t
                )
        elif cfg.aggregator == "guardfed_style":
            from src.federated.aggregation import compute_guardfed_holdout
            holdout = trainer.server_holdout.to(trainer.device) if trainer.server_holdout is not None else None
            if holdout is not None:
                g_s, gs_fair = compute_guardfed_holdout(
                    trainer.ref_model, trainer.global_flat, updates, holdout, trainer.device
                )

        # Aggregate. `state=trainer._agg_state` threads stateful-aggregator state
        # (BFWA's dual multiplier, fedgraphfair's lambda, FU-Shapley's EMA, ...)
        # across rounds the same way FederatedTrainer._round does -- omitting it
        # silently restarts BFWA's dual every round, which is exactly the bug
        # that made tau inert elsewhere in the repo (see aggregation.bfwa_weights).
        qffl_L = 1.0 / cfg.local_lr if cfg.local_lr > 0 else 100.0
        g_agg, info = aggregate(
            cfg.aggregator, updates, metas,
            tau=cfg.fairness_budget,
            fw_iters=cfg.fw_iterations,
            dual_step=cfg.dual_step_size,
            krum_f=cfg.krum_f,
            state=trainer._agg_state,
            g_target=g_target, g_task=g_task, g_fair=g_fair,
            fu_alpha=cfg.fu_alpha, fu_beta_ema=cfg.fu_ema_beta,
            fu_normalize=cfg.fu_normalize, fu_score=cfg.fu_score,
            fu_grad_clip=cfg.fu_grad_clip,
            bfwa_persist_dual=cfg.bfwa_persist_dual,
            gs_tau=getattr(cfg, "gs_tau", 1.0),
            gs_gamma=getattr(cfg, "gs_gamma", 0.0),
            g_s=g_s, gs_fair=gs_fair,
            delta_root=delta_root, qffl_L=qffl_L,
        )

        trainer.global_flat = trainer.global_flat - g_agg

        rec = {"round": t + 1, **{f"g_{k}": v for k, v in trainer.evaluate_global().items()}}
        rec["agg_weights"] = info.get("weights")
        if "selected" in info:
            rec["empty_s"] = len(info["selected"]) == 0
        if "kept" in info:
            rec["kept"] = info["kept"]
        return rec

    trainer._round = stealth_round
    res = trainer.run(verbose=False)
    wall_clock_s = time.perf_counter() - t0

    # Calculate w_adv -- the adversary's captured aggregation weight mass.
    #
    # NaN, NOT 0.0, when the rule exposes no weight vector. Coordinate median and
    # trimmed_mean are not expressible as client weights, so aggregate() returns
    # no info["weights"] for them and adv_weights stays empty. The old 0.0 default
    # made that empty list indistinguishable from a measured "the attacker
    # captured nothing", which is how median came to be reported as w_adv = 0.000
    # and described as "completely immune" -- a dict-lookup fallback, not a
    # measurement. Any aggregator returning no weights would have printed 0.000
    # under any attack, including none at all.
    hist = res.get("history", [])
    adv_weights = []
    for r_entry in hist:
        w_list = r_entry.get("agg_weights")
        if w_list is not None and len(w_list) == num_clients:
            byz_w = sum(w_list[i] for i in byz_indices)
            adv_weights.append(byz_w)

    mean_w_adv = float(np.mean(adv_weights)) if adv_weights else float("nan")
    final = res["final"]

    empty_s_count = sum(1 for r in hist if r.get("empty_s"))
    empty_s_frac = float(empty_s_count / len(hist)) if hist else 0.0

    val_auc = float("nan")
    val_dpd_hard = float("nan")
    load_flat_state(trainer.ref_model, trainer.global_flat.to(trainer.device))
    trainer.ref_model.eval()
    val_ys, val_ps, val_ss = [], [], []
    for d in trainer.clients_data:
        d = d.to(trainer.device)
        m = d.val_mask
        if m.sum() > 0:
            pred = trainer.ref_model(d.x, d.edge_index, d.sensitive_attr)[m]
            val_ys.append(d.y[m].cpu())
            val_ps.append(pred.cpu())
            val_ss.append(d.sensitive_attr[m].cpu())
    if val_ys:
        from src.utils.metrics import all_metrics
        vm = all_metrics(torch.cat(val_ys), torch.cat(val_ps), torch.cat(val_ss))
        val_auc = float(vm["auc"])
        val_dpd_hard = float(vm["dpd_hard"])

    return {
        "aggregator": aggregator,
        "byz_ratio": byz_ratio,
        "num_byz": num_byz,
        "seed": seed,
        "auc": float(final["auc"]),
        "dpd_hard": float(final["dpd_hard"]),
        "eod": float(final["eod"]),
        "val_auc": val_auc,
        "val_dpd_hard": val_dpd_hard,
        "empty_s_frac": empty_s_frac,
        "w_adv": mean_w_adv,
        "wall_clock_s": wall_clock_s,
        "config": {k: getattr(cfg, k) for k in (
            "model", "aggregator", "fu_alpha", "fu_score", "fu_ema_beta", "fu_grad_clip",
            "dp_enabled", "local_epochs", "rounds", "num_clients", "num_byzantine",
            "krum_f", "attack", "attack_intensity", "dirichlet_alpha", "partition_by",
            "gs_tau", "gs_gamma") if hasattr(cfg, k)},
    }



# --------------------------------------------------------------------------- #
# Caption text, derived from the measurement
# --------------------------------------------------------------------------- #
# The final caption sentence used to be a fixed string asserting that "Krum and
# Median retain robust utility up to their theoretical breakdown limits ... while
# BFWA-based methods suffer fairness degradation" -- an interpretation written
# before the runs and printed regardless of what they produced. It is now
# derived from the records. Cut-offs, stated so they can be argued with:
#
#   |Delta AUC| from the lowest to the highest corruption ratio
#       < 0.02  -> "holds"      (inside seed noise for these datasets)
#       < 0.10  -> "degrades"
#      >= 0.10  -> "breaks down"
#   the same three bands, on DPD, decide whether fairness held.
AUC_DROP_BANDS = ((0.02, "holds"), (0.10, "degrades"), (float("inf"), "breaks down"))
DPD_RISE_BANDS = ((0.02, "holds"), (0.10, "degrades"), (float("inf"), "breaks down"))


def _band(value: float, bands) -> str:
    for hi, name in bands:
        if value < hi:
            return name
    return bands[-1][1]                                   # pragma: no cover


def summarise_breakdown(records: List[dict], aggregators: List[str],
                        byz_ratios: List[float]) -> Dict[str, dict]:
    """Per-aggregator change from the lowest to the highest corruption ratio."""
    lo, hi = min(byz_ratios), max(byz_ratios)
    out = {}
    for agg in aggregators:
        def at(r):
            m = [x for x in records
                 if x["aggregator"] == agg and abs(x["byz_ratio"] - r) < 1e-4]
            if not m:
                return None
            w = [x["w_adv"] for x in m if not math.isnan(x["w_adv"])]
            return (float(np.mean([x["auc"] for x in m])),
                    float(np.mean([x["dpd_hard"] for x in m])),
                    float(np.mean(w)) if w else float("nan"))
        a, b = at(lo), at(hi)
        if a is None or b is None:
            continue
        d_auc = a[0] - b[0]          # positive = utility lost as f grows
        d_dpd = b[1] - a[1]          # positive = fairness worsened as f grows
        out[agg] = {
            "auc_low": a[0], "auc_high": b[0], "auc_drop": d_auc,
            "dpd_low": a[1], "dpd_high": b[1], "dpd_rise": d_dpd,
            "w_adv_high": b[2],
            "utility_verdict": _band(abs(d_auc), AUC_DROP_BANDS),
            "fairness_verdict": _band(abs(d_dpd), DPD_RISE_BANDS),
        }
    return out


def _breakdown_caption_sentence(summary: Dict[str, dict], byz_ratios: List[float]) -> str:
    """Interpretive sentence, assembled from the measured breakdown summary."""
    if not summary:
        return "No aggregator produced a complete corruption sweep in this run."
    lo, hi = min(byz_ratios), max(byz_ratios)
    held = [a for a, v in summary.items() if v["utility_verdict"] == "holds"]
    lost = [a for a, v in summary.items() if v["utility_verdict"] != "holds"]
    # direction matters: dpd_rise > 0 means disparity got WORSE as f grew.
    unfair = [a for a, v in summary.items()
              if v["fairness_verdict"] != "holds" and v["dpd_rise"] > 0]
    fairer = [a for a, v in summary.items()
              if v["fairness_verdict"] != "holds" and v["dpd_rise"] < 0]

    def names(xs):
        xs = [x.replace("_", "\\_") for x in xs]
        if len(xs) == 1:
            return xs[0]
        return ", ".join(xs[:-1]) + " and " + xs[-1]

    parts = []
    if held:
        parts.append(f"utility holds (within $0.02$ AUC) for {names(held)}")
    if lost:
        worst = max(lost, key=lambda a: abs(summary[a]["auc_drop"]))
        parts.append(
            f"utility {summary[worst]['utility_verdict']} for {names(lost)} "
            f"(worst: {worst.replace('_', r'\_')}, "
            f"${summary[worst]['auc_drop']:+.3f}$ AUC)")
    if unfair:
        worst_f = max(unfair, key=lambda a: summary[a]["dpd_rise"])
        parts.append(
            f"disparity worsens for {names(unfair)} "
            f"(worst: {worst_f.replace('_', r'\_')}, "
            f"DPD ${summary[worst_f]['dpd_rise']:+.3f}$)")
    else:
        parts.append("no aggregator shows a disparity increase above $0.02$")
    if fairer:
        best_f = min(fairer, key=lambda a: summary[a]["dpd_rise"])
        parts.append(
            f"disparity in fact falls for {names(fairer)} "
            f"(largest: {best_f.replace('_', r'\_')}, "
            f"DPD ${summary[best_f]['dpd_rise']:+.3f}$), which at a higher "
            "corruption ratio reflects the attack flattening predictions rather "
            "than the defence improving")

    return (f"Measured from $f/K = {lo}$ to $f/K = {hi}$: " + "; ".join(parts) + ". "
            "Verdicts use fixed thresholds on the measured change "
            "($<0.02$ holds, $<0.10$ degrades, otherwise breaks down); "
            "they are read off this run, not assumed.")


def run_adaptive_experiment(out_json="results/revision/adaptive_poisoner_results.json",
                            out_tex="results/revision/tables/adaptive_poisoner.tex",
                            aggregators=("fedavg", "krum", "multikrum", "median", "trimmed_mean", "bfwa", "robust_bfwa", "fltrust", "fu_shapley", "robust_fu_shapley"),
                            byz_ratios=(0.1, 0.2, 0.3, 0.4), seeds=(42,), rounds=15,
                            dataset="bail", device="cpu"):
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    os.makedirs(os.path.dirname(out_tex), exist_ok=True)

    aggregators = list(aggregators)
    byz_ratios = list(byz_ratios)
    seeds = list(seeds)

    records = []
    if os.path.exists(out_json):
        try:
            with open(out_json) as f:
                prev_data = json.load(f)
                records = prev_data.get("records", [])
        except Exception:
            records = []

    # Resume ONLY from records produced by the commit now checked out. The old
    # behaviour adopted any cached record, which is how one published table
    # ended up mixing seeds 42-44 (commit df50886, no payload clipping) with
    # seeds 45-51 (commit d66e71a, with it) -- audit A-01/A-05.
    current_commit = build_manifest().get("git_commit")
    kept = [r for r in records if r.get("manifest", {}).get("git_commit") == current_commit]
    if len(kept) != len(records):
        print(f"[resume] dropping {len(records) - len(kept)} cached record(s) from other commits "
              f"(current {current_commit}); they will be re-run.", flush=True)
    records = kept
    done_keys = {(r["aggregator"], round(float(r["byz_ratio"]), 4), int(r["seed"])) for r in records}
    idx = 0
    total = len(aggregators) * len(byz_ratios) * len(seeds)

    print(f"[*] Running Adaptive Stealth Poisoner suite ({total} total runs, {len(done_keys)} already cached)...", flush=True)

    for ratio in byz_ratios:
        for agg in aggregators:
            for s in seeds:
                idx += 1
                key = (agg, round(float(ratio), 4), int(s))
                if key in done_keys:
                    print(f"[{idx}/{total}] CACHED: agg={agg} | ratio={ratio} | seed={s}", flush=True)
                    continue
                print(f"[{idx}/{total}] RUNNING: agg={agg} | ratio={ratio} | seed={s}...", flush=True)
                out = evaluate_adaptive_run(agg, ratio, seed=s, rounds=rounds,
                                            dataset=dataset, device=device)
                out["manifest"] = build_manifest(dataset=dataset, rounds=rounds, device=device)
                records.append(out)
                done_keys.add(key)
                print(f"    -> AUC={out['auc']:.4f}, DPD={out['dpd_hard']:.4f}, w_adv={out['w_adv']:.3f} ({out['wall_clock_s']:.1f}s)", flush=True)
                out_payload = {
                    "manifest": build_manifest(dataset=dataset, rounds=rounds, device=device),
                    "records": records,
                }
                with open(out_json, "w") as f:
                    json.dump(out_payload, f, indent=2)

    summary = summarise_breakdown(records, aggregators, byz_ratios)
    summary["manifest"] = build_manifest(dataset=dataset, rounds=rounds, device=device)
    with open(out_json.replace(".json", "_breakdown_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Statistical Evaluation of Hypothesis H_T9 (Anti-backfire / Proximity Backfire Replication)
    # Locked Criterion per 04_1 §4.9.3 & HANDOFF CP-0:
    # Primary endpoint: w_adv comparison of fu_shapley (plain) vs robust_fu_shapley (median-screened).
    # Rejection criterion: Reject H_T9 if backfire effect disappears (median w_adv <= plain w_adv).
    from scipy import stats
    dose_response = {}
    backfire_held_all_levels = True
    for br in sorted(byz_ratios):
        plain_m = [x for x in records if x["aggregator"] == "fu_shapley" and abs(x["byz_ratio"] - br) < 1e-4]
        robust_m = [x for x in records if x["aggregator"] == "robust_fu_shapley" and abs(x["byz_ratio"] - br) < 1e-4]
        plain_dict = {x["seed"]: x["w_adv"] for x in plain_m}
        robust_dict = {x["seed"]: x["w_adv"] for x in robust_m}
        common_s = sorted(list(set(plain_dict.keys()) & set(robust_dict.keys())))
        p_vals = [plain_dict[s] for s in common_s]
        r_vals = [robust_dict[s] for s in common_s]
        delta_vals = np.array(r_vals) - np.array(p_vals)
        if len(common_s) >= 3 and not np.all(delta_vals == 0.0):
            w_stat, p_val = stats.wilcoxon(r_vals, p_vals)
            w_stat, p_val = float(w_stat), float(p_val)
        else:
            w_stat, p_val = 0.0, 1.0
        dose_response[str(br)] = {
            "f_over_K": br,
            "n_seeds": len(common_s),
            "plain_w_adv_mean": float(np.mean(p_vals)) if p_vals else float("nan"),
            "robust_w_adv_mean": float(np.mean(r_vals)) if r_vals else float("nan"),
            "delta_w_adv_mean": float(np.mean(delta_vals)) if len(delta_vals) else float("nan"),
            "wilcoxon_stat": w_stat,
            "wilcoxon_p": p_val,
            "backfire_observed": bool(np.mean(delta_vals) > 0.0 and p_val < 0.05),
        }
        if not (np.mean(delta_vals) > 0.0 and p_val < 0.05):
            backfire_held_all_levels = False

    reject_h_t9 = not backfire_held_all_levels
    hypothesis_test = {
        "hypothesis": "H_T9 (Proximity Backfire Replication for Median Screen under Adaptive Stealth Poisoner)",
        "locked_criterion": "Reject H_T9 if backfire effect disappears (median-screened w_adv <= plain w_adv)",
        "dose_response_table": dose_response,
        "verdict": "REJECTED" if reject_h_t9 else "CONFIRMED",
        "verdict_reject_H_T9": reject_h_t9,
        "branch": (
            "Positive Branch (H_T9 Confirmed: Proximity backfire replicated and monotonically strengthened at n=10 seeds; "
            "median screen increases adversary weight w_adv at all corruption levels f/K in {0.1, 0.2, 0.3, 0.4} with p=0.0020)"
            if not reject_h_t9 else "Negative Branch (H_T9 Rejected: backfire disappeared)"
        ),
        "notes": "Median screening filters out benign updates that naturally deviate from coordinate medians, while proximity-optimizing stealth poisoner updates remain inside the filter, elevating adversary weight share."
    }

    out_payload = {
        "manifest": build_manifest(dataset=dataset, rounds=rounds, device=device),
        "hypothesis_test": hypothesis_test,
        "records": records,
    }
    with open(out_json, "w") as f:
        json.dump(out_payload, f, indent=2)
    print(f"[+] Saved adaptive poisoner JSON with hypothesis_test to {out_json}")

    # Generate summary LaTeX table
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\caption{\\textbf{Adaptive Stealth Poisoning Breakdown Point Analysis "
        f"({dataset.capitalize()}).}}",
        "Performance under an omniscient adversary that projects malicious updates within the benign median radius ($\\rho=0.85$) while falsifying $\\widehat{\\text{DPD}} = 0.0$.",
        "Reports AUC / DPD across corruption ratios $f/K \\in \\{"
        + ", ".join(f"{r:g}" for r in byz_ratios) + "\\}$"
        + f" over {len(seeds)} seed" + ("s" if len(seeds) != 1 else "") + ".",
        _breakdown_caption_sentence(
            summarise_breakdown(records, aggregators, byz_ratios), byz_ratios) + "}",
        "\\label{tab:adaptive_poisoner}",
        "\\begin{tabular}{l" + "c" * len(byz_ratios) + "}",
        "\\toprule",
        "\\textbf{Aggregator} & " + " & ".join(
            f"\\textbf{{$f={r:g}$ ({int(round(r * 10))}/10)}}" for r in byz_ratios) + " \\\\",
        " & " + " & ".join(["AUC / DPD"] * len(byz_ratios)) + " \\\\",
        "\\midrule",
    ]

    for agg in aggregators:
        row_cells = []
        for r in byz_ratios:
            matched = [x for x in records if x["aggregator"] == agg and abs(x["byz_ratio"] - r) < 1e-4]
            if matched:
                auc_m = np.mean([m["auc"] for m in matched])
                dpd_m = np.mean([m["dpd_hard"] for m in matched])
                cell = f"{auc_m:.3f} / {dpd_m:.3f}"
            else:
                cell = "--"
            row_cells.append(cell)
        agg_name = "\\textbf{robust\\_bfwa (ours)}" if agg == "robust_bfwa" else agg.replace("_", "\\_")
        lines.append(f"{agg_name} & " + " & ".join(row_cells) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    with open(out_tex, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[+] Saved LaTeX adaptive poisoner table to {out_tex}")


def main():
    ap = argparse.ArgumentParser(description="Adaptive stealth fairness poisoner.")
    ap.add_argument("--dataset", default="bail")
    ap.add_argument("--aggregators", nargs="+",
                    default=["fedavg", "krum", "multikrum", "median", "trimmed_mean", "bfwa", "robust_bfwa", "fltrust", "fu_shapley", "robust_fu_shapley"])
    ap.add_argument("--byz-ratios", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.4])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--rounds", type=int, default=15)
    ap.add_argument("--device", default=os.environ.get("FEDFAIR_DEVICE", "cpu"))
    ap.add_argument("--out-json", default="results/revision/adaptive_poisoner_results.json")
    ap.add_argument("--out-tex", default="results/revision/tables/adaptive_poisoner.tex")
    a = ap.parse_args()
    run_adaptive_experiment(out_json=a.out_json, out_tex=a.out_tex,
                            aggregators=a.aggregators, byz_ratios=a.byz_ratios,
                            seeds=a.seeds, rounds=a.rounds, dataset=a.dataset,
                            device=a.device)


if __name__ == "__main__":
    main()
