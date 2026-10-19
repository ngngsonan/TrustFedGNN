"""GPU Revision Runner for Colab (T4).

Executes GPU workloads on Colab T4:
1. Pokec-z Aggregator Control (Pha 2.1):
   Isolates fu_shapley vs fedavg on identical TrustFedGNN GAT backbone + FTGD/DP
   across 10 seeds {42..51} on Pokec-z (K=10, R=50, alpha=0.3).
2. FLTrust on Flagship SOTA Benchmarks (Pha 2.2):
   Evaluates fltrust baseline across 10 seeds {42..51} on Pokec-z and Credit Default.
3. Pokec-z Adversarial (P5-X3, Tier D+):
   Clean/Scaling/Fairness-Poisoning on Pokec-z (K=10, R=50, n=10 seeds).

Outputs:
  - /content/results/aggregator_control_pokecz.json
  - /content/results/fltrust_sota_results.json
  - /content/results/pokecz_adversarial.json
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import statistics as st
import sys
import time

if os.path.exists("/content/FedFairGNN"):
    os.chdir("/content/FedFairGNN")
    sys.path.insert(0, "/content/FedFairGNN")
sys.path.insert(0, os.path.abspath("."))

if os.path.exists("/content/manifest_local.json"):
    try:
        with open("/content/manifest_local.json") as f:
            _man = json.load(f)
        _c = _man.get("commit", "")
        _d = "1" if _man.get("dirty", False) else "0"
        os.environ["FEDFAIR_GIT_COMMIT"] = _c
        os.environ["FEDFAIR_GIT_DIRTY"] = _d
        os.environ["GIT_COMMIT"] = _c
        os.environ["GIT_DIRTY"] = _d
    except Exception as _e:
        print(f"Warning loading manifest_local.json: {_e}")

from scipy.stats import wilcoxon
import torch

from src.config import ExperimentConfig
from src.federated.trainer import FederatedTrainer
from src.utils.metrics import weight_oscillation
from src.utils.provenance import build_manifest

SEEDS = list(range(42, 52))


def run_pokecz_control(out_file: str, device: str = "cuda", pokec_label_protocol: str = "legacy_negative"):
    print("=" * 80, flush=True)
    print("  PHASE 2.1: POKEC-Z AGGREGATOR ISOLATION CONTROL (GPU)", flush=True)
    print(f"  Device: {device.upper()} | Seeds: {SEEDS} | Clients: K=10 | Rounds: 50 | Protocol: {pokec_label_protocol}", flush=True)
    print("=" * 80, flush=True)

    arms = {
        "M1_fu_shapley": dict(model="trustfedgnn", aggregator="fu_shapley", dp_enabled=True, dp_mode="ftgd"),
        "A0_fedavg": dict(model="trustfedgnn", aggregator="fedavg", dp_enabled=True, dp_mode="ftgd"),
    }

    raw_per_seed = {}
    arms_summary = {}

    for name, overrides in arms.items():
        print(f"\n--- Running Arm: {name} ---", flush=True)
        per = []
        for s in SEEDS:
            t0 = time.perf_counter()
            cfg = ExperimentConfig.canonical(
                dataset="pokec_z",
                seed=s,
                num_clients=10,
                rounds=50,
                local_epochs=3,
                dirichlet_alpha=0.3,
                device=device,
                pokec_label_protocol=pokec_label_protocol,
                **overrides,
            )
            trainer = FederatedTrainer(cfg)
            res = trainer.run(verbose=False)
            dt = time.perf_counter() - t0
            f = res.get("final", res)
            per.append({
                "seed": s,
                "auc": float(f["auc"]),
                "dpd_hard": float(f["dpd_hard"]),
                "eod": float(f["eod"]),
                "wall_clock_s": float(dt),
            })
            print(f"  [Seed {s}] AUC={f['auc']:.4f} DPD={f['dpd_hard']:.4f} EOD={f['eod']:.4f} ({dt:.1f}s)", flush=True)

            del trainer, res
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        raw_per_seed[name] = per
        arms_summary[name] = {
            "auc": {"mean": float(st.mean(x["auc"] for x in per)), "std": float(st.stdev(x["auc"] for x in per))},
            "dpd_hard": {"mean": float(st.mean(x["dpd_hard"] for x in per)), "std": float(st.stdev(x["dpd_hard"] for x in per))},
            "eod": {"mean": float(st.mean(x["eod"] for x in per)), "std": float(st.stdev(x["eod"] for x in per))},
            "wall_clock_s": {"mean": float(st.mean(x["wall_clock_s"] for x in per))},
        }

    a = raw_per_seed["M1_fu_shapley"]
    b = raw_per_seed["A0_fedavg"]
    paired = {}
    for k in ("auc", "dpd_hard", "eod"):
        d = [x[k] - y[k] for x, y in zip(a, b)]
        w_res = wilcoxon(d)
        paired[k] = {
            "mean_delta": float(st.mean(d)),
            "wilcoxon_p": float(w_res.pvalue),
            "wins_m1": int(sum(1 for v in d if v > 0)),
            "wins_a0": int(sum(1 for v in d if v < 0)),
            "ties": int(sum(1 for v in d if v == 0)),
            "per_seed_deltas": [float(v) for v in d],
        }
        print(f"  Delta {k:<9} = {paired[k]['mean_delta']:+.4f} (p={paired[k]['wilcoxon_p']:.4f}, wins={paired[k]['wins_m1']}/10)", flush=True)

    manifest = build_manifest(
        experiment="pokecz_aggregator_control",
        args={
            "dataset": "pokec_z",
            "num_clients": 10,
            "rounds": 50,
            "local_epochs": 3,
            "dirichlet_alpha": 0.3,
            "seeds": SEEDS,
            "note": "Pokec-z aggregator isolation control on identical GAT backbone and FTGD/DP",
        },
    )

    payload = {
        "manifest": manifest,
        "arms": arms_summary,
        "paired_M1_minus_A0": paired,
        "per_seed_raw": raw_per_seed,
    }

    os.makedirs(os.path.dirname(out_file) or ".", exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[+] Successfully saved Pokec-z control results to {out_file}", flush=True)
    return payload


def run_fltrust_sota(out_file: str, device: str = "cuda"):
    print("=" * 80, flush=True)
    print("  PHASE 2.2: FLTRUST BASELINE ON FLAGSHIP SOTA BENCHMARKS (GPU)", flush=True)
    print(f"  Device: {device.upper()} | Seeds: {SEEDS} | Clients: K=10 | Rounds: 50", flush=True)
    print("=" * 80, flush=True)

    datasets = ["pokec_z", "credit"]
    results = {}

    for dset in datasets:
        print(f"\n--- Running FLTrust on {dset.upper()} ---", flush=True)
        per = []
        for s in SEEDS:
            t0 = time.perf_counter()
            cfg = ExperimentConfig.canonical(
                dataset=dset,
                seed=s,
                num_clients=10,
                rounds=50,
                local_epochs=3,
                dirichlet_alpha=0.3,
                device=device,
                model="trustfedgnn",
                aggregator="fltrust",
                # Task-only reference, stated explicitly: canonical() defaults
                # fu_alpha=0.1 and the fltrust branch now honours fu_alpha>0
                # (aggregation.py), so omitting this would silently run FLTrust
                # with a fairness term. The published FLTrust rows were produced
                # before that branch existed, i.e. task-only.
                fu_alpha=0.0,
                dp_enabled=False,
            )
            trainer = FederatedTrainer(cfg)
            res = trainer.run(verbose=False)
            dt = time.perf_counter() - t0
            f = res.get("final", res)
            weights_hist = [r.get("agg_weights") for r in res.get("history", [])]
            omega_w = weight_oscillation(weights_hist) if weights_hist else 0.0

            entry = {
                "seed": s,
                "auc": float(f["auc"]),
                "dpd_soft": float(f.get("dpd_soft", 0.0)),
                "dpd_hard": float(f["dpd_hard"]),
                "eod": float(f["eod"]),
                "omega_w": float(omega_w),
                "wall_clock_s": float(dt),
            }
            per.append(entry)
            print(f"  [{dset} Seed {s}] AUC={f['auc']:.4f} DPD={f['dpd_hard']:.4f} EOD={f['eod']:.4f} ({dt:.1f}s)", flush=True)

            del trainer, res
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        results[dset] = {
            "summary": {
                "auc": {"mean": float(st.mean(x["auc"] for x in per)), "std": float(st.stdev(x["auc"] for x in per))},
                "dpd_hard": {"mean": float(st.mean(x["dpd_hard"] for x in per)), "std": float(st.stdev(x["dpd_hard"] for x in per))},
                "eod": {"mean": float(st.mean(x["eod"] for x in per)), "std": float(st.stdev(x["eod"] for x in per))},
                "omega_w": {"mean": float(st.mean(x["omega_w"] for x in per))},
                "wall_clock_s": {"mean": float(st.mean(x["wall_clock_s"] for x in per))},
            },
            "per_seed": per,
        }

    manifest = build_manifest(
        experiment="fltrust_flagship_sota",
        args={
            "datasets": datasets,
            "num_clients": 10,
            "rounds": 50,
            "local_epochs": 3,
            "dirichlet_alpha": 0.3,
            "seeds": SEEDS,
            "aggregator": "fltrust",
        },
    )

    payload = {
        "manifest": manifest,
        "results": results,
    }

    os.makedirs(os.path.dirname(out_file) or ".", exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[+] Successfully saved FLTrust SOTA results to {out_file}", flush=True)
    return payload


def run_pokecz_adversarial(out_file: str, device: str = "cuda"):
    """P5-X3: Adversarial robustness on Pokec-z (N=67796, K=10, R=50).

    Pre-registered decision rule (fixed before the run):
    H_P5X3: w_adv <= 0.0500 under Scaling(c=100) and AUC >= 0.7500.
    """
    print("=" * 80, flush=True)
    print("  P5-X3: POKEC-Z ADVERSARIAL (K=10, R=50, n=10 seeds)", flush=True)
    print(f"  Device: {device.upper()} | Seeds: {SEEDS}", flush=True)
    print("=" * 80, flush=True)

    scenarios = {
        "clean": dict(attack="none", num_byzantine=0, attack_intensity=0.0),
        "scaling_c100": dict(attack="scaling", num_byzantine=3, attack_intensity=100.0),
        "fairness_poison": dict(attack="fairness_poison", num_byzantine=3, attack_intensity=10.0),
    }

    results = {}
    total_runs = len(SEEDS) * len(scenarios)
    done = 0

    for scenario_name, attack_kwargs in scenarios.items():
        print(f"\n--- Scenario: {scenario_name} ---", flush=True)
        per = []
        for s in SEEDS:
            t0 = time.perf_counter()
            cfg = ExperimentConfig.canonical(
                dataset="pokec_z",
                seed=s,
                num_clients=10,
                rounds=50,
                local_epochs=3,
                dirichlet_alpha=0.3,
                device=device,
                model="trustfedgnn",
                aggregator="fu_shapley",
                dp_enabled=False,
                **attack_kwargs,
            )
            trainer = FederatedTrainer(cfg)
            res = trainer.run(verbose=False)
            dt = time.perf_counter() - t0
            f = res.get("final", res)

            # Collect adversary weights if applicable
            adv_weights = []
            for rec in getattr(trainer, "history", []):
                w = rec.get("agg_weights")
                if w and attack_kwargs["num_byzantine"] > 0:
                    adv_weights.append(float(sum(w[:attack_kwargs["num_byzantine"]])))
            w_adv = float(sum(adv_weights) / len(adv_weights)) if adv_weights else 0.0

            row = {
                "seed": s,
                "auc": float(f["auc"]),
                "dpd_hard": float(f["dpd_hard"]),
                "w_adv": w_adv,
                "wall_clock_s": float(dt),
            }
            per.append(row)
            done += 1
            print(f"  [{done}/{total_runs}] {scenario_name} seed={s} AUC={f['auc']:.4f} "
                  f"DPD={f['dpd_hard']:.4f} w_adv={w_adv:.4f} ({dt:.1f}s)", flush=True)

            del trainer, res
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        aucs = [r["auc"] for r in per]
        dpds = [r["dpd_hard"] for r in per]
        w_advs = [r["w_adv"] for r in per]
        results[scenario_name] = {
            "summary": {
                "auc": {"mean": float(st.mean(aucs)), "std": float(st.stdev(aucs))},
                "dpd_hard": {"mean": float(st.mean(dpds)), "std": float(st.stdev(dpds))},
                "w_adv": {"mean": float(st.mean(w_advs)), "std": float(st.stdev(w_advs))},
            },
            "per_seed": per,
        }

    manifest = build_manifest(
        experiment="p5_x3_pokecz_adversarial",
        args={
            "dataset": "pokec_z",
            "num_clients": 10,
            "rounds": 50,
            "local_epochs": 3,
            "dirichlet_alpha": 0.3,
            "seeds": SEEDS,
            "scenarios": list(scenarios.keys()),
            "aggregator": "fu_shapley",
            "note": "P5-X3: adversarial robustness on Pokec-z large-scale graph",
        },
    )
    sc = results.get("scaling_c100", {})
    w_adv_sc = sc.get("summary", {}).get("w_adv", {}).get("mean", 999.0)
    auc_cl = results.get("clean", {}).get("summary", {}).get("auc", {}).get("mean", 0.0)
    rejected = bool(w_adv_sc > 0.0500 or auc_cl < 0.7500)
    verdict = "REJECTED (H_P5X3 breached)" if rejected else "CONFIRMED (H_P5X3 holds)"

    hypothesis_test = {
        "pre_registered_thresholds": {
            "w_adv_scaling_max": 0.0500,
            "auc_min": 0.7500,
            "p_alpha": 0.05,
        },
        "measured": {
            "w_adv_scaling_mean": float(w_adv_sc),
            "auc_clean_mean": float(auc_cl),
        },
        "verdict": verdict,
        "rejected": rejected,
    }

    payload = {
        "manifest": manifest,
        "results": results,
        "rejection_criterion": {
            "w_adv_scaling_threshold": 0.0500,
            "auc_threshold": 0.7500,
            "alpha": 0.05,
            "n": len(SEEDS),
        },
        "hypothesis_test": hypothesis_test,
    }
    os.makedirs(os.path.dirname(out_file) or ".", exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[+] Successfully saved P5-X3 results to {out_file}", flush=True)

    print(f"  H_P5X3: w_adv(scaling)={w_adv_sc:.4f} auc(clean)={auc_cl:.4f} => {verdict}", flush=True)
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", choices=["all", "pokecz_control", "fltrust", "pokecz_adversarial"], default="all")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--pokecz-out", default="/content/results/aggregator_control_pokecz.json")
    parser.add_argument("--fltrust-out", default="/content/results/fltrust_sota_results.json")
    parser.add_argument("--pokecz-adv-out", default="/content/results/pokecz_adversarial.json")
    parser.add_argument("--pokec-label-protocol", choices=["legacy_negative", "known_only"], default="legacy_negative",
                        help="Pokec label protocol: 'legacy_negative' or 'known_only' (P5-K10.4)")
    args, _ = parser.parse_known_args()

    # build_manifest() reads the device from FEDFAIR_DEVICE and defaults to "cpu".
    os.environ["FEDFAIR_DEVICE"] = args.device

    if args.job in ("all", "pokecz_control"):
        run_pokecz_control(args.pokecz_out, device=args.device, pokec_label_protocol=args.pokec_label_protocol)

    if args.job in ("all", "fltrust"):
        run_fltrust_sota(args.fltrust_out, device=args.device)

    if args.job in ("all", "pokecz_adversarial"):
        run_pokecz_adversarial(args.pokecz_adv_out, device=args.device)
