"""Fixed-state replay runner for evaluating single-step metadata manipulation.

At each communication round t of an honest trajectory:
1. Snapshot state (theta, updates, references, benign telemetry, aggregator state).
2. Compute input_sha256 to ensure exact identical inputs.
3. Call aggregate() twice:
   - Branch H: honest client metadata
   - Branch L: adversary reports best-response false metadata
4. Record w_h, w_l, w_adv_h, w_adv_l, agg_diff_norm, and 1-step val loss/DPD.
5. Trajectory continues along the honest branch.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

# Ensure repo root is on sys.path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.config import ExperimentConfig, set_seed
from src.federated.aggregation import aggregate
from src.federated.attacks import BEST_RESPONSE_LIE, poison_updates
from src.federated.client import Client, flatten_state, load_flat_state, _weighted_bce, _soft_dpd
from src.data.datasets import load_dataset
from src.data.partition import partition_graph, carve_server_holdout
from src.models import build_model
from src.trust.incentive import get_server_target_gradients, get_server_root_update
from src.utils.provenance import build_manifest

RULES = [
    "fairfed",
    "qffl",
    "qffl_eq",
    "f2gnn",
    "fedgraphfair",
    "popets_fairfed",
    "bfwa",
    "cgsv",
    "fltrust",
    "fltrust_root",
    "fu_shapley",
    "fu_shapley_alpha0",
]


def compute_input_sha256(theta: torch.Tensor, updates: List[torch.Tensor],
                         ref: torch.Tensor | None, benign_metas: List[dict],
                         state: dict | None) -> str:
    h = hashlib.sha256()
    h.update(theta.detach().cpu().contiguous().numpy().tobytes())
    for u in updates:
        h.update(u.detach().cpu().contiguous().numpy().tobytes())
    if ref is not None:
        h.update(ref.detach().cpu().contiguous().numpy().tobytes())
    h.update(json.dumps(benign_metas, sort_keys=True).encode("utf-8"))
    if state:
        st_parts = []
        for k, v in sorted(state.items()):
            if isinstance(v, torch.Tensor):
                st_parts.append((k, v.detach().cpu().numpy().tolist()))
            elif isinstance(v, (int, float, str, bool)):
                st_parts.append((k, v))
        h.update(json.dumps(st_parts, sort_keys=True).encode("utf-8"))
    return h.hexdigest()


@torch.no_grad()
def eval_val_metrics(model, clients_data, flat: torch.Tensor, device: str) -> Dict[str, float]:
    load_flat_state(model, flat.to(device))
    model.eval()
    ys, ps, ss = [], [], []
    for d in clients_data:
        d = d.to(device)
        m = d.val_mask
        if m.sum() == 0:
            continue
        p = model(d.x, d.edge_index, d.sensitive_attr)[m]
        ys.append(d.y[m].cpu())
        ps.append(p.cpu())
        ss.append(d.sensitive_attr[m].cpu())
    if not ys:
        return {"loss": 0.0, "dpd": 0.0}
    y = torch.cat(ys)
    p = torch.cat(ps)
    s = torch.cat(ss)
    bce = float(_weighted_bce(p, y.float()).item())
    dpd = float(_soft_dpd(p, s).item())
    return {"loss": bce, "dpd": dpd}


def _supersedes() -> Dict:
    """Link to the artifact this replay replaces (C20a), by path and sha256."""
    old = os.path.join(os.getcwd(), "results", "campaigns", "C20a_fixed_state_replay.json")
    if not os.path.exists(old):
        return {}
    import hashlib
    with open(old, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    return {"artifact": "results/campaigns/C20a_fixed_state_replay.json", "sha256": digest,
            "reason": "C20a injected no falsified report for rules outside BEST_RESPONSE_LIE "
                      "and did not seed each run"}


def union_lie(metas: List[Dict], byzantine_ids) -> Dict:
    """Falsified report for rules without a best-response lie (CGSV, FLTrust,
    root-FLTrust, FU-Alignment). Every report field a rule could read is moved to
    its most favourable value; the sample count ``n`` is left honest."""
    benign = [m for i, m in enumerate(metas) if i not in byzantine_ids]
    lie = {"dpd": 0.0, "eod": 0.0, "eo": 0.0, "perf": 0.99, "loss": 0.01, "group1_rate": 0.5}
    if any("loss_pre" in m for m in benign):
        lie["loss_pre"] = 2.0 * max(m.get("loss_pre", 0.0) for m in benign)
    return lie


def run_rule_replay(rule: str, seed: int, rounds: int, dataset: str = "german",
                    device: str = "cpu") -> List[Dict]:
    # Seed every RNG inside the worker, before any data, model or client exists,
    # so each (rule, seed) run is reproducible and initialisations are paired.
    set_seed(seed)
    # Configure rule
    aggregator = "fu_shapley" if rule.startswith("fu_shapley") else rule
    fu_alpha = 0.0 if rule in ("fu_shapley_alpha0", "fltrust", "fltrust_root") else 0.1
    report_loss_pre = (rule == "qffl_eq")

    cfg = ExperimentConfig.canonical(
        dataset=dataset,
        seed=seed,
        num_clients=5,
        rounds=rounds,
        model="gcn",
        aggregator=aggregator,
        attack="fairness_poison_honest_report",
        attack_intensity=10.0,
        num_byzantine=1,
        dp_enabled=False,
        local_fairness=True,
        report_group_rate=True,
        report_loss_pre=report_loss_pre,
        fu_alpha=fu_alpha,
        fu_val_source="server_holdout",
        fu_holdout_size=100,
        bfwa_persist_dual=True,
    )

    # Initialize data & partitions
    data_full = load_dataset(cfg.dataset, root=cfg.data_root, seed=cfg.seed)
    server_holdout, client_raw = carve_server_holdout(
        data_full, cfg.fu_holdout_size, seed=cfg.seed, mode=cfg.fu_holdout_mode
    )
    clients_data = partition_graph(
        client_raw, cfg.num_clients, method=cfg.partition,
        alpha=cfg.dirichlet_alpha, by=cfg.partition_by, seed=cfg.seed
    )

    # Models & clients
    ref_model = build_model(cfg.model, data_full.x.shape[1], cfg).to(device)
    global_flat = flatten_state(ref_model.state_dict()).cpu()

    clients = [
        Client(i, clients_data[i], cfg, device, byzantine=(i == 0))
        for i in range(cfg.num_clients)
    ]
    byzantine_ids = [0]
    agg_state: Dict = {}
    round_records = []

    for t in range(rounds):
        # 1. Client local training under honest report trajectory
        updates, metas_honest = [], []
        for c in clients:
            c.set_flat(global_flat)
            c.train()
            g_k = global_flat - c.get_flat()
            updates.append(g_k)
            metas_honest.append(c.meta())

        # 2. Server references
        g_target = g_task = g_fair = None
        delta_root = None
        fu_warmup = t < cfg.fu_warmup_rounds

        if aggregator == "fltrust_root":
            load_flat_state(ref_model, global_flat.to(device))
            delta_root = get_server_root_update(ref_model, server_holdout.to(device), cfg, t).cpu()
        elif "fu_shapley" in aggregator or aggregator in ("fltrust", "fltrust_ema"):
            load_flat_state(ref_model, global_flat.to(device))
            tg = get_server_target_gradients(
                ref_model, server_holdout.to(device), cfg.fu_alpha,
                fair_surrogate=cfg.fu_fair_surrogate
            )
            if tg is not None:
                g_target, g_task, g_fair = (g.cpu() for g in tg)

        # 3. Snapshot and input hash verification
        benign_metas = [metas_honest[i] for i in range(len(metas_honest)) if i not in byzantine_ids]
        ref_tensor = delta_root if delta_root is not None else (g_target if g_target is not None else g_task)
        input_hash_h = compute_input_sha256(global_flat, updates, ref_tensor, benign_metas, agg_state)

        # 4. Honest aggregation branch
        qffl_L = 1.0 / cfg.local_lr if cfg.local_lr > 0 else 100.0
        state_h = copy.deepcopy(agg_state)
        agg_kw = dict(
            tau=cfg.fairness_budget, fw_iters=cfg.fw_iterations,
            dual_step=cfg.dual_step_size, trimmed_beta=cfg.trimmed_beta,
            krum_f=1, q_ffl=cfg.q_ffl, fairfed_beta=cfg.fairfed_beta,
            g_target=g_target, g_task=g_task, g_fair=g_fair,
            fu_alpha=cfg.fu_alpha, fu_beta_ema=cfg.fu_ema_beta,
            fu_normalize=cfg.fu_normalize, fu_score=cfg.fu_score,
            fu_warmup=fu_warmup, fu_grad_clip=cfg.fu_grad_clip,
            fu_warmup_agg=cfg.fu_warmup_agg,
            bfwa_persist_dual=cfg.bfwa_persist_dual,
            gs_tau=cfg.gs_tau, gs_gamma=cfg.gs_gamma,
            delta_root=delta_root, qffl_L=qffl_L,
        )

        agg_h, info_h = aggregate(aggregator, updates, copy.deepcopy(metas_honest),
                                  state=state_h, **agg_kw)

        # 5. Lying aggregation branch (adversary lies via BEST_RESPONSE_LIE)
        metas_lying = copy.deepcopy(metas_honest)
        if rule in BEST_RESPONSE_LIE:
            lie_vals = BEST_RESPONSE_LIE[rule](metas_lying, set(byzantine_ids))
        else:
            lie_vals = union_lie(metas_lying, set(byzantine_ids))
        metas_lying[0].update(lie_vals)
        # Gate: the falsified report must actually differ from the honest one,
        # while the sample count stays honest.
        assert metas_lying[0] != metas_honest[0], f"{rule} t={t}: falsified report equals honest report"
        assert metas_lying[0]["n"] == metas_honest[0]["n"], f"{rule} t={t}: sample count changed"

        input_hash_l = compute_input_sha256(global_flat, updates, ref_tensor, benign_metas, agg_state)
        assert input_hash_h == input_hash_l, f"Input sha256 mismatch at round {t}: {input_hash_h} vs {input_hash_l}"

        state_l = copy.deepcopy(agg_state)
        agg_l, info_l = aggregate(aggregator, updates, copy.deepcopy(metas_lying),
                                  state=state_l, **agg_kw)

        # 6. Step measurements
        w_h = info_h.get("weights", [1.0 / cfg.num_clients] * cfg.num_clients)
        w_l = info_l.get("weights", [1.0 / cfg.num_clients] * cfg.num_clients)
        w_adv_h = float(sum(w_h[i] for i in byzantine_ids))
        w_adv_l = float(sum(w_l[i] for i in byzantine_ids))
        agg_diff_norm = float(torch.norm(agg_h - agg_l).item())

        flat_next_h = global_flat - agg_h
        flat_next_l = global_flat - agg_l
        val_h = eval_val_metrics(ref_model, clients_data, flat_next_h, device)
        val_l = eval_val_metrics(ref_model, clients_data, flat_next_l, device)

        rec = {
            "round": t,
            "input_sha256": input_hash_h,
            "w_h": [float(x) for x in w_h],
            "w_l": [float(x) for x in w_l],
            "w_adv_h": w_adv_h,
            "w_adv_l": w_adv_l,
            "d_adv": w_adv_l - w_adv_h,
            "agg_diff_norm": agg_diff_norm,
            "val_h": val_h,
            "val_l": val_l,
        }
        round_records.append(rec)

        # 7. Advance honest trajectory
        global_flat = flat_next_h
        agg_state = state_h

    return round_records


def _eval_seed_task(task_args):
    rule, s, rounds, dataset, device = task_args
    recs = run_rule_replay(rule, s, rounds, dataset=dataset, device=device)
    d_s = float(np.mean([r["d_adv"] for r in recs]))
    w_h_mean = float(np.mean([r["w_adv_h"] for r in recs]))
    w_l_mean = float(np.mean([r["w_adv_l"] for r in recs]))
    return s, d_s, w_h_mean, w_l_mean, recs


def main():
    parser = argparse.ArgumentParser(description="Fixed-state replay runner (C20a)")
    parser.add_argument("--smoke", action="store_true", help="Run 1 seed (42), 2 rounds")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="Specific seeds")
    parser.add_argument("--rules", type=str, nargs="+", default=RULES, help="Rules to run")
    parser.add_argument("--rounds", type=int, default=20, help="Rounds per run")
    parser.add_argument("--workers", type=int, default=1, help="Parallel worker processes")
    parser.add_argument("--dataset", type=str, default="german", help="Dataset")
    parser.add_argument("--device", type=str, default="cpu", help="Compute device")
    parser.add_argument("--out-json", type=str, default=None, help="Output path")
    args = parser.parse_args()

    seeds = [42] if args.smoke else (args.seeds if args.seeds else list(range(42, 72)))
    rounds = 2 if args.smoke else args.rounds
    out_path = args.out_json or ("results/campaigns/smoke/C20_replay_smoke.json" if args.smoke
                                else "results/campaigns/C20a_fixed_state_replay.json")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    t0 = time.time()
    results: Dict[str, dict] = {}

    import concurrent.futures

    for rule in args.rules:
        t_rule = time.time()
        results[rule] = {"seeds": {}}
        if args.workers > 1 and len(seeds) > 1:
            tasks = [(rule, s, rounds, args.dataset, args.device) for s in seeds]
            with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as ex:
                for s, d_s, w_h_mean, w_l_mean, recs in ex.map(_eval_seed_task, tasks):
                    results[rule]["seeds"][str(s)] = {
                        "d_s": d_s,
                        "w_adv_h": w_h_mean,
                        "w_adv_l": w_l_mean,
                        "rounds": recs,
                    }
        else:
            for s in seeds:
                _, d_s, w_h_mean, w_l_mean, recs = _eval_seed_task(
                    (rule, s, rounds, args.dataset, args.device)
                )
                results[rule]["seeds"][str(s)] = {
                    "d_s": d_s,
                    "w_adv_h": w_h_mean,
                    "w_adv_l": w_l_mean,
                    "rounds": recs,
                }
        # summary across seeds
        d_seeds = [results[rule]["seeds"][str(s)]["d_s"] for s in seeds]
        results[rule]["summary"] = {
            "mean_d": float(np.mean(d_seeds)),
            "n_nonzero": int(sum(abs(d) > 1e-5 for d in d_seeds)),
            "n_positive": int(sum(d > 1e-5 for d in d_seeds)),
            "elapsed_s": round(time.time() - t_rule, 2),
        }
        print(f"[{rule:18s}] mean_d={results[rule]['summary']['mean_d']:+.4f} "
              f"nonzero={results[rule]['summary']['n_nonzero']}/{len(seeds)} "
              f"({results[rule]['summary']['elapsed_s']}s)")

    payload = {
        "manifest": build_manifest(device=args.device),
        "config": {
            "dataset": args.dataset,
            "seeds": seeds,
            "rounds": rounds,
            "rules": args.rules,
            "smoke": args.smoke,
            "lie_protocol": "BEST_RESPONSE_LIE for report-reading rules; union_lie (n kept honest) for all others",
            "seeding": "set_seed(seed) per run; per-seed data split",
        },
        "supersedes": _supersedes(),
        "elapsed_total_s": round(time.time() - t0, 2),
        "results": results,
    }

    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[+] Output written to {out_path} ({payload['elapsed_total_s']}s)")


if __name__ == "__main__":
    main()
