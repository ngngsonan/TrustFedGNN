"""End-to-end capture of published fairness-aware aggregators via a false report.

The paper's central claim is that steering aggregation on a statistic each client
reports about *itself* is an attack surface. Until now that was demonstrated only
against BFWA, which is our own construction -- a fair objection is that
a baseline built to consume the falsified fields cannot substantiate a claim about
the literature. This runs the attack end-to-end against rules taken from published
methods, and against two that read no metadata at all as controls.

Three arms, so the damage can be attributed to the channel rather than to the
update:

    clean          no adversary
    poison_honest  adversary trains to maximise disparity, reports truthfully
    poison_lie     same update, best-response false report

Without ``poison_honest`` a drop under ``poison_lie`` could be the poisoned
gradient rather than the lie, and the claim would not follow.

Backbone is held at GCN for every arm so the contrast is between aggregation
rules and nothing else. Two registry entries differ from their METHODS defaults
under this constraint (f2gnn is GAT there, fltrust is trustfedgnn); the caption
must say so.

The adversary lies only about fairness/utility statistics, never about ``n``.
Sample count is a different channel, and an adversary permitted to inflate it
would also capture plain FedAvg -- collapsing the distinction this experiment
exists to draw. See attacks.BEST_RESPONSE_LIE.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath("."))

import numpy as np
from scipy.stats import wilcoxon

from src.config import ExperimentConfig
from src.federated.trainer import FederatedTrainer
from src.utils.provenance import build_manifest

SEEDS = list(range(42, 72))

# Rules under test. `reads` documents the client-reported field each one consumes
# to form weights; it is what makes the taxonomy a prediction rather than a label.
RULES = {
    "fairfed":           dict(reads="dpd",               published=True),
    "qffl":              dict(reads="loss",              published=True),
    "qffl_eq":           dict(reads="loss (pre-training)", published=True),
    "f2gnn":             dict(reads="dpd, group1_rate",  published=True),
    "fedgraphfair":      dict(reads="loss (persistent dual)", published=True),
    "popets_fairfed":    dict(reads="dpd",               published=True),
    "bfwa":              dict(reads="dpd, perf",         published=True),   # ours, PMLR v319; positive control
    "cgsv":              dict(reads="-- gradients only", published=True),   # negative control
    "fltrust":           dict(reads="-- gradients only", published=True),   # negative control
    "fltrust_root":      dict(reads="-- gradients only", published=True),   # faithful FLTrust control
    "fu_shapley":        dict(reads="-- gradients only", published=False),  # ours, server holdout fairness-utility
    "fu_shapley_alpha0": dict(reads="-- gradients only", published=False),  # ours ablation, alpha=0.0 task-only
}

ARMS = {
    "clean":         dict(attack="none",                          num_byzantine=0),
    "poison_honest": dict(attack="fairness_poison_honest_report", num_byzantine=1),
    "poison_lie":    dict(attack="fairness_poison",               num_byzantine=1),
}


def _cfg(rule: str, arm: str, seed: int, dataset: str, num_clients: int, rounds: int):
    if rule not in RULES:
        raise ValueError(f"unknown rule {rule!r}; add it to RULES explicitly")
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; add it to ARMS explicitly")
    over = dict(ARMS[arm])
    aggregator = "fu_shapley" if rule.startswith("fu_shapley") else rule
    extra = {}
    if rule == "fu_shapley_alpha0":
        extra["fu_alpha"] = 0.0
    elif rule == "fu_shapley":
        extra["fu_alpha"] = 0.1
    elif rule == "fltrust":
        # Task-only FLTrust, stated explicitly: canonical() defaults fu_alpha=0.1
        # and the fltrust branch honours fu_alpha>0, so without this the negative
        # control would silently carry a fairness term (audit A-06).
        extra["fu_alpha"] = 0.0
    elif rule == "fltrust_root":
        extra["fu_alpha"] = 0.0
        extra["fu_val_source"] = "server_holdout"
    elif rule == "qffl_eq":
        extra["report_loss_pre"] = True
    return ExperimentConfig.canonical(
        seed=seed, dataset=dataset, num_clients=num_clients, rounds=rounds,
        model="gcn",                      # held fixed across every rule
        aggregator=aggregator,
        local_fairness=True,
        report_group_rate=True,           # f2gnn needs it; harmless elsewhere
        dp_enabled=False,                 # isolate the metadata channel from DP
        meta_lie=rule if over["attack"] == "fairness_poison" else None,
        **extra,
        **over,
    )


def _w_adv(trainer, num_byzantine: int):
    """Mean weight the adversary held across rounds; None if the rule hides weights."""
    if not num_byzantine:
        return None
    vals = []
    for rec in getattr(trainer, "history", []):
        w = rec.get("agg_weights")
        if w:
            vals.append(float(np.sum([w[i] for i in range(num_byzantine)])))
    return float(np.mean(vals)) if vals else None



CONTRAST_METRICS = ("dpd_hard", "eod", "auc", "w_adv")


def paired_contrasts(lie_per_seed, hon_per_seed, metrics=CONTRAST_METRICS):
    """Paired lie-minus-honest contrast, seed by seed.

    Pairs on the INTERSECTION of seeds that are finite in BOTH arms. The earlier
    version dropped only None and let NaN through, so a diverged run produced a
    NaN p-value; a single NaN then silently corrupts an entire Holm family
    downstream, because comparisons against NaN are all False and Holm's reject
    flag is cumulative (guard: experiments/stats.py :: holm_bonferroni).
    Observed for real on q-FedAvg in an earlier run.

    n_nonzero_pairs is reported because the signed-rank test DROPS tied pairs,
    so the attainable p-floor is 2**(1 - n_nonzero_pairs), not 2**(1 - n_pairs).
    """
    lie = {r["seed"]: r for r in lie_per_seed}
    hon = {r["seed"]: r for r in hon_per_seed}
    pair = {}
    for k in metrics:
        def _ok(r, _k=k):
            return r.get(_k) is not None and np.isfinite(r[_k])
        seeds_ok = [s for s in sorted(hon) if s in lie and _ok(hon[s]) and _ok(lie[s])]
        if not seeds_ok:
            pair[k] = {"mean_delta": None, "wilcoxon_p": None, "n_pairs": 0,
                       "seeds_used": [], "seeds_dropped": sorted(hon)}
            continue
        x = [lie[s][k] for s in seeds_ok]
        y = [hon[s][k] for s in seeds_ok]
        d = np.asarray(x, float) - np.asarray(y, float)
        p = 1.0 if np.all(d == 0) else float(wilcoxon(x, y)[1])
        pair[k] = {
            "mean_delta": float(d.mean()), "wilcoxon_p": p,
            "n_pairs": len(seeds_ok),
            "n_nonzero_pairs": int(np.count_nonzero(d)),
            "n_positive": int((d > 0).sum()), "n_negative": int((d < 0).sum()),
            "seeds_used": seeds_ok,
            "seeds_dropped": [s for s in sorted(hon) if s not in seeds_ok],
            "per_seed_deltas": [float(v) for v in d],
        }
    return pair


def run(out_json: str, dataset="german", num_clients=5, rounds=20, seeds=SEEDS, rules=None):
    rules_to_run = rules if rules is not None else list(RULES.keys())
    results = {}
    if os.path.exists(out_json):
        try:
            with open(out_json) as f:
                prev = json.load(f).get("results", {})
                for r_name, r_val in prev.items():
                    if r_name in RULES and r_name not in rules_to_run:
                        results[r_name] = r_val
        except Exception:
            pass

    total = len(rules_to_run) * len(ARMS) * len(seeds)
    done = 0
    for rule in rules_to_run:
        results[rule] = {"reads": RULES[rule]["reads"],
                         "published": RULES[rule]["published"], "arms": {}}
        for arm in ARMS:
            per_seed = []
            for s in seeds:
                cfg = _cfg(rule, arm, s, dataset, num_clients, rounds)
                tr = FederatedTrainer(cfg)
                res = tr.run(verbose=False)
                fin = res["final"]          # metrics live here, not at top level
                per_seed.append({
                    "seed": s,
                    "auc": float(fin.get("auc", float("nan"))),
                    "dpd_hard": float(fin.get("dpd_hard", float("nan"))),
                    "eod": float(fin.get("eod", float("nan"))),
                    "diverged": float(fin.get("diverged", 0.0)),
                    "w_adv": _w_adv(tr, ARMS[arm]["num_byzantine"]),
                })
                done += 1
                print(f"[{done}/{total}] {rule:<15} {arm:<14} seed={s} "
                      f"auc={per_seed[-1]['auc']:.4f} dpd={per_seed[-1]['dpd_hard']:.4f} "
                      f"w_adv={per_seed[-1]['w_adv']}", flush=True)
            def _m(k):
                """Mean over finite seeds, AND how many there were.

                Reporting the mean alone hides dropped runs: q-FedAvg diverges on
                German for 3/10 seeds in one arm and 2/10 in another, so the two
                arm means were taken over DIFFERENT seed sets while the artifact
                looked complete. n_valid makes that visible.
                """
                v = [r[k] for r in per_seed if r[k] is not None and np.isfinite(r[k])]
                return (float(np.mean(v)) if v else None), len(v)

            summary = {"per_seed": per_seed}
            for key, out in (("auc", "auc_mean"), ("dpd_hard", "dpd_hard_mean"),
                             ("eod", "eod_mean"), ("w_adv", "w_adv_mean")):
                summary[out], summary[out.replace("_mean", "_n_valid")] = _m(key)
            summary["n_diverged"] = sum(1 for r in per_seed if r.get("diverged"))
            results[rule]["arms"][arm] = summary
            # written after every arm: a killed session leaves usable partials
            _dump(out_json, results, dataset, num_clients, rounds, seeds)

    # paired lie-vs-honest contrast, the quantity the claim rests on
    for rule in results:
        a = results[rule]["arms"]
        if "poison_lie" not in a or "poison_honest" not in a:
            continue
        results[rule]["paired_lie_minus_honest"] = paired_contrasts(
            a["poison_lie"]["per_seed"], a["poison_honest"]["per_seed"])
    _dump(out_json, results, dataset, num_clients, rounds, seeds)
    return results


def _dump(path, results, dataset, num_clients, rounds, seeds):
    payload = {
        "manifest": build_manifest(
            experiment="metadata_capture_endtoend",
            args={"dataset": dataset, "num_clients": num_clients, "rounds": rounds,
                  "seeds": list(seeds), "rules": list(results.keys()), "arms": list(ARMS),
                  "backbone": "gcn (held fixed across all rules)",
                  "dp_enabled": False,
                  "lie_scope": "fairness/utility statistics only; never n"},
        ),
        "results": results,
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=1)
        f.write("\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-json", default="results/revision/metadata_capture_endtoend.json")
    ap.add_argument("--dataset", default="german")
    ap.add_argument("--num-clients", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--rules", nargs="+", default=None, help="Specific rules to run")
    a = ap.parse_args()
    run(a.out_json, a.dataset, a.num_clients, a.rounds, tuple(a.seeds), rules=a.rules)


if __name__ == "__main__":
    main()
