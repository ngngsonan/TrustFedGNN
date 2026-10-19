"""Deterministic paired analysis for the Discover AI manuscript."""
from __future__ import annotations

import json
import os
from typing import Dict

import numpy as np

from experiments.campaigns.stats_p5 import (
    factorial_2x2,
    holm,
    paired_bootstrap_ci,
    wilcoxon_p,
)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULTS = os.path.join(ROOT, "results", "campaigns")
METRICS = ("w_adv", "auc", "dpd_hard", "eod")


def load(name: str) -> dict:
    with open(os.path.join(RESULTS, name)) as f:
        return json.load(f)


# C24 schedules are defined on the 0-indexed round t used by the trainer, while
# the artifact stores the 1-indexed round number t + 1.
def active_poison_round(stored_round: int, schedule: str) -> bool:
    t = stored_round - 1
    if schedule == "S1":
        return t >= 10
    if schedule == "S2":
        return t >= 10 and ((t - 10) // 5) % 2 == 0
    if schedule == "S3":
        return t >= 1
    return False


# The S3 report omitted the sample count `n`, which FedAvg reads (aggregate()
# defaults it to 1). Those 20 cells are excluded; the other methods do not read
# `n` (see tests/test_campaigns.py).
C24_INVALID_CELLS = {("S3", "fedavg")}


def c24_active_round_summary(c24: dict) -> Dict:
    """Mean w_adv and I_adv over the active poisoning rounds of each run, then over seeds."""
    out: Dict = {}
    for run in c24["raw_runs"]:
        sch, meth = run["schedule"], run["method"]
        if sch == "benign" or abs(run["byzantine_fraction"] - 0.2) > 1e-9 or (sch, meth) in C24_INVALID_CELLS:
            continue
        rounds = [pr for pr in run["per_round"] if active_poison_round(pr["round"], sch)]
        cell = out.setdefault(run["dataset"], {}).setdefault(sch, {}).setdefault(
            meth, {"w_adv": [], "I_adv": [], "seeds": []})
        cell["w_adv"].append(float(np.mean([pr["w_adv_round"] for pr in rounds])))
        cell["I_adv"].append(float(np.mean([pr["I_adv"] for pr in rounds])))
        cell["seeds"].append(run["seed"])
    for by_sch in out.values():
        for by_meth in by_sch.values():
            for meth, c in by_meth.items():
                by_meth[meth] = {"w_adv_active_mean": float(np.mean(c["w_adv"])),
                                 "I_adv_active_mean": float(np.mean(c["I_adv"])),
                                 "n_seeds": len(c["seeds"])}
    return out


def paired(left: list, right: list, metric: str, seed: int = 0) -> dict:
    a = {x["seed"]: x for x in left}
    b = {x["seed"]: x for x in right}
    seeds = sorted(set(a) & set(b))
    delta = np.asarray([a[s][metric] - b[s][metric] for s in seeds], dtype=float)
    return {
        "n": len(seeds),
        "left_mean": float(np.mean([a[s][metric] for s in seeds])),
        "right_mean": float(np.mean([b[s][metric] for s in seeds])),
        "mean_delta": float(np.mean(delta)),
        "ci95": list(paired_bootstrap_ci(delta, seed=seed)),
        "wilcoxon_p": wilcoxon_p(delta),
        "left_better_seeds": int(np.sum(delta < 0 if metric in ("w_adv", "dpd_hard", "eod") else delta > 0)),
    }


def main() -> None:
    c23 = load("C23_mechanism_factorial.json")["results"]
    c26 = load("C26_fu_alpha0.json")["results"]
    output: Dict[str, dict] = {"controlled": {}, "factorial": {}, "clean": {}, "fairness_reference": {}}

    root_tests = []
    root_refs = []
    for dataset in ("german", "bail"):
        poison = c23[dataset]["fairness_poison"]
        fu = poison["A4_fu_shapley"]["per_seed"]
        output["controlled"][dataset] = {}
        for comparator, arm in (("root_fltrust", "R1_fltrust_root"), ("fedavg", "A0_fedavg")):
            output["controlled"][dataset][comparator] = {}
            for metric in METRICS:
                result = paired(fu, poison[arm]["per_seed"], metric)
                output["controlled"][dataset][comparator][metric] = result
                if comparator == "root_fltrust":
                    root_tests.append(result["wilcoxon_p"])
                    root_refs.append(result)

        output["factorial"][dataset] = {}
        for metric in METRICS:
            values = {}
            means = {}
            for key, arm in (("RS_R", "F_RS_R"), ("RS_C", "F_RS_C"),
                             ("SR_R", "F_SR_R"), ("SR_C", "F_SR_C")):
                records = sorted(poison[arm]["per_seed"], key=lambda x: x["seed"])
                values[key] = [x[metric] for x in records]
                means[key] = float(np.mean(values[key]))
            effects = factorial_2x2(values)
            for value in effects.values():
                value["wilcoxon_p"] = wilcoxon_p(value["per_seed"])
                del value["per_seed"]
            output["factorial"][dataset][metric] = {"cell_means": means, "effects": effects}

        clean = c23[dataset]["clean"]
        output["clean"][dataset] = {
            comparator: {metric: paired(clean["A4_fu_shapley"]["per_seed"], clean[arm]["per_seed"], metric)
                         for metric in ("auc", "dpd_hard", "eod")}
            for comparator, arm in (("fedavg", "A0_fedavg"), ("root_fltrust", "R1_fltrust_root"))
        }
        output["fairness_reference"][dataset] = {
            metric: paired(c23[dataset]["fairness_poison"]["A4_fu_shapley"]["per_seed"],
                           c26[dataset]["per_seed"], metric)
            for metric in METRICS
        }

    for result, adjusted in zip(root_refs, holm(root_tests)):
        result["holm_p_across_8"] = adjusted
    alpha_results = [output["fairness_reference"][d][m] for d in ("german", "bail") for m in METRICS]
    for result, adjusted in zip(alpha_results, holm([x["wilcoxon_p"] for x in alpha_results])):
        result["holm_p_across_8"] = adjusted

    output["c24_active_rounds"] = c24_active_round_summary(load("C24_adaptive_access.json"))

    out_path = os.path.join(RESULTS, "analysis", "discover_ai.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"[+] wrote {out_path}")


if __name__ == "__main__":
    main()
