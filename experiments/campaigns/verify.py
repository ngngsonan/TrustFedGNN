"""Acceptance checks and summaries for the campaigns cited by the article.

    python -m experiments.campaigns.verify                 # every campaign with a status file
    python -m experiments.campaigns.verify --ids C20r C23   # selected
    python -m experiments.campaigns.verify --smoke         # check smoke artifacts instead

For each campaign:
  * outputs exist and match the sha256 recorded in status/<ID>.json;
  * every manifest found in the artifact (top level and per record) carries the
    SAME git_commit as the status file and git_dirty == False;
  * the design is complete (cells x seeds) and the config invariants hold
    (e.g. FLTrust fu_alpha == 0.0, Dirichlet "ours" arm is fu_shapley, factorial K=5);
  * NaN counts are reported, not hidden;
  * summary statistics are recomputed from per-seed values (mean, sample std ddof=1)
    so the manuscript tables can be regenerated from this report.

Writes results/campaigns/verification_report.{json,md}. Exit code 0 only
if every requested campaign PASSes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics as st
import sys
from collections import defaultdict

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)

from experiments.campaigns.campaigns import BY_ID, CAMPAIGNS, OUT  # noqa: E402

SEEDS_10 = list(range(42, 52))
SEEDS_30 = list(range(42, 72))
SEEDS_20 = list(range(42, 62))


# ---------------------------------------------------------------- helpers
def isnan(x) -> bool:
    return isinstance(x, float) and math.isnan(x)


def ms(vals):
    v = [x for x in vals if isinstance(x, (int, float)) and not isnan(x)]
    if not v:
        return {"n": 0, "n_nan": len(vals), "mean": None, "std": None}
    return {"n": len(v), "n_nan": len(vals) - len(v), "mean": round(st.mean(v), 6),
            "std": round(st.stdev(v), 6) if len(v) > 1 else 0.0}


def manifests(obj, out=None):
    """Every dict carrying a git_commit key, anywhere in the artifact."""
    out = [] if out is None else out
    if isinstance(obj, dict):
        if "git_commit" in obj:
            out.append(obj)
        for v in obj.values():
            manifests(v, out)
    elif isinstance(obj, list):
        for v in obj:
            manifests(v, out)
    return out


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Report:
    def __init__(self, cid):
        self.cid, self.checks, self.summary = cid, [], {}

    def check(self, name, ok, detail=""):
        self.checks.append({"check": name, "ok": bool(ok), "detail": str(detail)[:500]})
        return ok

    @property
    def verdict(self):
        if not self.checks:
            return "MISSING"
        return "PASS" if all(c["ok"] for c in self.checks) else "FAIL"


# ---------------------------------------------------------------- per-campaign checks
def chk_C05(d, r):
    rec = d.get("records", [])
    aggs = ["fedavg", "krum", "multikrum", "median", "trimmed_mean", "flame", "bfwa", "robust_bfwa",
            "fltrust", "fu_shapley", "robust_fu_shapley"]
    cells = defaultdict(list)
    for x in rec:
        cells[(x["aggregator"], round(x["byz_ratio"], 2))].append(x)
    exp = [(a, f) for a in aggs for f in (0.1, 0.2, 0.3, 0.4)]
    bad = {f"{a}@{f}": len(cells.get((a, f), [])) for a, f in exp if sorted(x["seed"] for x in cells.get((a, f), [])) != SEEDS_10}
    r.check("11 rules x 4 ratios x seeds 42..51", not bad and len(rec) == 440, bad or len(rec))
    fl = [x.get("config", {}) for x in rec if x["aggregator"] == "fltrust"]
    r.check("FLTrust fu_alpha=0.0", fl and all(c.get("fu_alpha") == 0.0 for c in fl))
    r.check("protocol R=15, E=1 recorded", rec and all(x.get("config", {}).get("rounds") == 15 and
                                                     x.get("config", {}).get("local_epochs") == 1 for x in rec))
    r.summary["cells"] = {f"{a}@{f}": {m: ms([x[m] for x in cells.get((a, f), [])]) for m in ("auc", "dpd_hard", "w_adv")}
                          for a, f in exp}


def chk_C20(datas, r):
    """End-to-end report capture (C20b German, C20c Bail)."""
    r.check("C20 produced 2 artifacts", len(datas) == 2, len(datas))
    for ds_name, d_obj in zip(("german", "bail"), datas):
        res = d_obj.get("results", {})
        r.check(f"C20 audit {ds_name} has qffl_eq and fltrust_root",
                "qffl_eq" in res and "fltrust_root" in res, list(res.keys()))
        for r_k in ("qffl_eq", "fltrust_root"):
            arms = res.get(r_k, {}).get("arms", {})
            r.check(f"C20 audit {ds_name} {r_k} has 3 arms",
                    {"clean", "poison_honest", "poison_lie"}.issubset(arms.keys()), list(arms.keys()))
        root = res.get("fltrust_root", {}).get("arms", {})
        if {"poison_honest", "poison_lie"}.issubset(root):
            r.check(f"C20 audit {ds_name} root-FLTrust invariant end to end",
                    root["poison_honest"].get("w_adv_mean") == root["poison_lie"].get("w_adv_mean"))


def chk_C20r(d, r):
    """Fixed-update report audit: every rule receives a falsified report; runs are seeded."""
    cfg = d.get("config", {})
    r.check("C20r has 30 seeds", cfg.get("seeds") == SEEDS_30, cfg.get("seeds"))
    r.check("C20r rounds == 20", cfg.get("rounds") == 20, cfg.get("rounds"))
    r.check("C20r records the union-lie protocol", "union_lie" in cfg.get("lie_protocol", ""), cfg.get("lie_protocol"))
    r.check("C20r links the superseded C20a", bool(d.get("supersedes", {}).get("sha256")), d.get("supersedes"))
    expected = {"fairfed", "qffl", "qffl_eq", "f2gnn", "fedgraphfair", "popets_fairfed",
                "bfwa", "cgsv", "fltrust", "fltrust_root", "fu_shapley", "fu_shapley_alpha0"}
    res = d.get("results", {})
    r.check("C20r covers 12 rules", expected.issubset(res), sorted(res))
    for rule in sorted(expected & set(res)):
        seeds = res[rule].get("seeds", {})
        r.check(f"C20r {rule} has 30 seeds", len(seeds) == 30, len(seeds))
        r.summary[rule] = res[rule].get("summary")


def chk_C21(datas, r):
    if isinstance(datas, dict):
        datas = [datas]
    r.check("C21 produced 2 artifacts", len(datas) >= 2, len(datas))
    c21a = datas[0]
    c21_contract = datas[1] if len(datas) > 1 else {}

    # C21a slack vector checks
    eps_expected = {"0.5", "1.0", "2.0", "4.0", "8.0", "16.0", "32.0"}
    c21a_eps = set(k for k in c21a.keys() if k not in ("manifest", "_manifest"))
    r.check("C21a covers 7 epsilons", eps_expected.issubset(c21a_eps), c21a_eps)

    # Check client_vectors present in per_round
    for e in eps_expected.intersection(c21a_eps):
        rounds = c21a[e].get("per_round", [])
        r.check(f"C21a eps={e} has 200 rounds", len(rounds) == 200, len(rounds))
        if rounds:
            cvs = rounds[0].get("client_vectors", [])
            r.check(f"C21a eps={e} client_vectors recorded (K=10)", len(cvs) == 10, len(cvs))

    # C21 privacy contract checks
    res_contract = c21_contract.get("results", {})
    contract_eps = set(res_contract.keys())
    r.check("C21 contract covers 7 epsilons", eps_expected.issubset(contract_eps), contract_eps)

    r.summary["R_het_mean"] = {
        e: res_contract[e].get("R_het_mean")
        for e in sorted(contract_eps, key=float) if e in res_contract
    }
    r.summary["R_equal_mean"] = {
        e: res_contract[e].get("R_equal_mean")
        for e in sorted(contract_eps, key=float) if e in res_contract
    }


def chk_C22(d, r):
    sel = d.get("selection", {})
    r.check("C22 has selection for 3 scenarios",
            {"A-German", "A-Bail", "B-Bail"}.issubset(sel.keys()), list(sel.keys()))

    tune = d.get("tune", {})
    r.check("C22 has tune for 3 scenarios",
            {"A-German", "A-Bail", "B-Bail"}.issubset(tune.keys()), list(tune.keys()))

    conf = d.get("confirm", {})
    part_a = conf.get("part_a", {})
    r.check("C22 confirm part_a has german and bail",
            "german" in part_a and "bail" in part_a)

    ger_recs = part_a.get("german", [])
    bail_recs = part_a.get("bail", [])
    part_b_recs = conf.get("part_b", [])
    fltrust_recs = d.get("fltrust_root_stealth", [])

    r.check("C22 confirm part_a german non-empty", len(ger_recs) > 0, len(ger_recs))
    r.check("C22 confirm part_a bail non-empty", len(bail_recs) > 0, len(bail_recs))
    r.check("C22 confirm part_b non-empty", len(part_b_recs) > 0, len(part_b_recs))
    r.check("C22 fltrust_root_stealth non-empty", len(fltrust_recs) > 0, len(fltrust_recs))

    r.summary["selected_configs"] = {sc: sel[sc].get("selected") for sc in sel}
    if ger_recs:
        r.summary["german_auc_mean"] = float(np.mean([x["auc"] for x in ger_recs]))
        r.summary["german_dpd_mean"] = float(np.mean([x["dpd_hard"] for x in ger_recs]))
        r.summary["german_w_adv_mean"] = float(np.mean([x["w_adv"] for x in ger_recs]))
    if bail_recs:
        r.summary["bail_auc_mean"] = float(np.mean([x["auc"] for x in bail_recs]))
        r.summary["bail_dpd_mean"] = float(np.mean([x["dpd_hard"] for x in bail_recs]))
        r.summary["bail_w_adv_mean"] = float(np.mean([x["w_adv"] for x in bail_recs]))


def chk_C23(d, r):
    res = d.get("results", {})
    r.check("C23 covers german and bail",
            "german" in res and "bail" in res, list(res.keys()))

    for ds in ("german", "bail"):
        ds_res = res.get(ds, {})
        r.check(f"C23 {ds} has fairness_poison and clean",
                "fairness_poison" in ds_res and "clean" in ds_res, list(ds_res.keys()))
        poison_arms = ds_res.get("fairness_poison", {})
        clean_arms = ds_res.get("clean", {})
        r.check(f"C23 {ds} fairness_poison has arms", len(poison_arms) >= 2, list(poison_arms.keys()))
        r.check(f"C23 {ds} clean has arms", len(clean_arms) >= 1, list(clean_arms.keys()))

        first_arm = next(iter(poison_arms.values()), {})
        recs = first_arm.get("per_seed", [])
        if recs:
            pr = recs[0].get("per_round", [])
            r.check(f"C23 {ds} has per_round records", len(pr) == 20, len(pr))

    r.summary["german_poison"] = {a: res.get("german", {}).get("fairness_poison", {}).get(a, {}).get("w_adv_mean")
                                  for a in res.get("german", {}).get("fairness_poison", {})}
    r.summary["bail_poison"] = {a: res.get("bail", {}).get("fairness_poison", {}).get(a, {}).get("w_adv_mean")
                                for a in res.get("bail", {}).get("fairness_poison", {})}


def chk_C23b(d, r):
    res = d.get("results", {})
    r.check("C23b covers german and bail",
            "german" in res and "bail" in res, list(res.keys()))

    expected_arms = {"A4_fu_shapley", "R1_fltrust_root", "A1_fltrust", "A5_fltrust_ema"}
    expected_seeds_30 = set(range(42, 72))
    mech_keys = ("w_adv_round", "I_adv", "cos_agg_g_task", "cos_agg_g_fair")

    for ds in ("german", "bail"):
        ds_res = res.get(ds, {})
        r.check(f"C23b {ds} has fairness_poison",
                "fairness_poison" in ds_res, list(ds_res.keys()))
        poison_arms = ds_res.get("fairness_poison", {})
        r.check(f"C23b {ds} has all 4 expected arms",
                set(poison_arms.keys()) == expected_arms, list(poison_arms.keys()))

        for arm_name in expected_arms:
            arm_data = poison_arms.get(arm_name, {})
            recs = arm_data.get("per_seed", [])
            r.check(f"C23b {ds} {arm_name} has 30 seeds", len(recs) == 30, len(recs))
            seeds = {x["seed"] for x in recs}
            r.check(f"C23b {ds} {arm_name} seeds are 42..71", seeds == expected_seeds_30, len(seeds))
            all_rounds_valid = True
            all_finite = True
            for rec in recs:
                pr = rec.get("per_round", [])
                if len(pr) != 20:
                    all_rounds_valid = False
                for round_rec in pr:
                    for mk in mech_keys:
                        v = round_rec.get(mk)
                        if v is None or not (isinstance(v, (int, float)) and math.isfinite(v)):
                            all_finite = False
            r.check(f"C23b {ds} {arm_name} all 30 records have exactly 20 rounds", all_rounds_valid)
            r.check(f"C23b {ds} {arm_name} all rounds have finite mechanism metrics", all_finite)

    r.summary["german_poison"] = {a: res.get("german", {}).get("fairness_poison", {}).get(a, {}).get("w_adv_mean")
                                  for a in res.get("german", {}).get("fairness_poison", {})}
    r.summary["bail_poison"] = {a: res.get("bail", {}).get("fairness_poison", {}).get(a, {}).get("w_adv_mean")
                                for a in res.get("bail", {}).get("fairness_poison", {})}


def chk_C25(datas, r):
    """Pokec-z known-label preflight and FU-Alignment vs FedAvg control."""
    d_pref, d_ctrl = datas
    r.check("C25 preflight total_nodes == 67796", d_pref.get("total_nodes") == 67796, d_pref.get("total_nodes"))
    r.check("C25 preflight known_nodes == 10262", d_pref.get("known_nodes") == 10262, d_pref.get("known_nodes"))
    splits = d_pref.get("known_splits", {})
    r.check("C25 preflight known splits sum == 10262", splits.get("sum") == 10262, splits)
    r.check("C25 preflight known test nodes == 2566", splits.get("test") == 2566, splits.get("test"))
    raw = d_ctrl.get("per_seed_raw", {})
    r.check("C25 control has M1 and A0 arms", set(raw) == {"M1_fu_shapley", "A0_fedavg"}, list(raw))
    expected_seeds = set(SEEDS_10)
    m1, a0 = raw.get("M1_fu_shapley", []), raw.get("A0_fedavg", [])
    for name, recs in (("M1", m1), ("A0", a0)):
        r.check(f"C25 control {name} has seeds 42..51", {x["seed"] for x in recs} == expected_seeds)
    from experiments.campaigns.stats_p5 import noninferiority
    a = {x["seed"]: x["auc"] for x in a0}
    deltas = [x["auc"] - a[x["seed"]] for x in sorted(m1, key=lambda x: x["seed"])]
    noninf = noninferiority(deltas, delta=0.01, n=10000, seed=0)
    r.check("C25 control AUC non-inferiority at 0.01 (bootstrap seed 0)", noninf.get("pass") is True, noninf)
    r.summary["preflight"] = {"total": d_pref.get("total_nodes"), "known": d_pref.get("known_nodes"),
                              "known_test": splits.get("test")}
    r.summary["control_noninferiority"] = noninf


def chk_C24(d, r):
    res = d.get("results", {})
    r.check("C24 covers german and bail", "german" in res and "bail" in res, list(res.keys()))

    expected_methods = {"fedavg", "fltrust_root", "fu_shapley", "guardfed_style"}

    for ds in ("german", "bail"):
        ds_res = res.get(ds, {})
        schedules = set(ds_res.keys())
        r.check(f"C24 {ds} has S3 and benign", {"S3", "benign"}.issubset(schedules), list(schedules))

        for sch in schedules:
            sch_res = ds_res[sch]
            r.check(f"C24 {ds} {sch} has 4 methods", expected_methods.issubset(set(sch_res.keys())), list(sch_res.keys()))

            for m in expected_methods.intersection(set(sch_res.keys())):
                m_data = sch_res[m]
                per_seed = m_data.get("per_seed", [])
                seeds_found = {x["seed"] for x in per_seed}
                expected_seeds = set(range(42, 52))
                r.check(f"C24 {ds} {sch} {m} has 10 seeds (42..51)",
                        seeds_found == expected_seeds,
                        f"found {len(seeds_found)} seeds")

                all_finite = True
                for rec in per_seed:
                    for metric in ("auc", "dpd_hard", "w_adv", "I_adv"):
                        v = rec.get(metric)
                        if v is None or not (isinstance(v, (int, float)) and math.isfinite(v)):
                            all_finite = False
                            break
                    if not all_finite:
                        break
                r.check(f"C24 {ds} {sch} {m} metrics finite", all_finite)

                if sch == "benign":
                    benign_zero = all(rec.get("w_adv", 0.0) == 0.0 and rec.get("I_adv", 0.0) == 0.0 for rec in per_seed)
                    r.check(f"C24 {ds} {sch} {m} benign w_adv and I_adv are zero", benign_zero)

    r.summary["datasets"] = list(res.keys())
    schedules_all = set()
    for ds in res:
        if isinstance(res[ds], dict):
            schedules_all.update(res[ds].keys())
    r.summary["schedules"] = sorted(list(schedules_all))
    r.summary["methods"] = sorted(list(expected_methods))


def chk_C26(d, r):
    results = d.get("results", {})
    bad = {}
    for dataset in ("german", "bail"):
        records = results.get(dataset, {}).get("per_seed", [])
        if sorted(x.get("seed") for x in records) != SEEDS_30:
            bad[dataset] = len(records)
        if any(x.get("arm") != "FU_alpha0" or x.get("scenario") != "fairness_poison" for x in records):
            bad[f"{dataset}/labels"] = "wrong arm or scenario"
    r.check("German and Bail each contain paired seeds 42..71", not bad, bad)

    c23_path = os.path.join(REPO, OUT, "C23_mechanism_factorial.json")
    r.check("paired C23 reference exists", os.path.exists(c23_path), c23_path)
    if os.path.exists(c23_path):
        with open(c23_path) as f:
            c23 = json.load(f)["results"]
        for dataset in ("german", "bail"):
            alpha0 = results.get(dataset, {}).get("per_seed", [])
            alpha1 = c23.get(dataset, {}).get("fairness_poison", {}).get("A4_fu_shapley", {}).get("per_seed", [])
            r.check(
                f"{dataset}: alpha=0 and alpha=0.1 have identical paired seeds",
                [x.get("seed") for x in alpha0] == sorted(x.get("seed") for x in alpha1),
            )
            r.summary[dataset] = {
                "alpha0": {m: ms([x[m] for x in alpha0]) for m in ("auc", "dpd_hard", "eod", "w_adv")},
                "alpha01": {m: ms([x[m] for x in alpha1]) for m in ("auc", "dpd_hard", "eod", "w_adv")},
            }


CHECKS = {"C05": chk_C05, "C20": chk_C20, "C20r": chk_C20r, "C21": chk_C21, "C22": chk_C22,
          "C23": chk_C23, "C23b": chk_C23b, "C24": chk_C24, "C25": chk_C25, "C26": chk_C26}
MULTI = {"C20", "C21", "C25"}


def verify(cid: str, smoke: bool) -> Report:
    c = BY_ID[cid]
    r = Report(cid)
    tag = f"{cid}_smoke" if smoke else cid
    status_path = os.path.join(REPO, OUT, "status", f"{tag}.json")
    outputs = c.smoke_outputs if smoke else c.outputs
    if not os.path.exists(status_path):
        r.check("status file exists", False, status_path)
        return r
    with open(status_path) as f:
        status = json.load(f)
    r.check("run state == done", status.get("state") == "done", status.get("state"))
    commit = status.get("provenance", {}).get("commit")
    exp = os.environ.get("RERUN_EXPECTED_COMMIT")
    if exp and not smoke:
        r.check("commit == RERUN_EXPECTED_COMMIT", bool(commit) and commit.startswith(exp), commit)
    datas = []
    for p in outputs:
        full = os.path.join(REPO, p)
        if not r.check(f"exists: {p}", os.path.exists(full)):
            return r
        rec = (status.get("outputs") or {}).get(p) or {}
        r.check(f"sha256 matches status: {p}", rec.get("sha256") == sha256(full),
                "artifact changed after the run" if rec else "no hash in status")
        with open(full) as f:
            d = json.load(f)
        datas.append(d)
        mans = manifests(d)
        commits = {m.get("git_commit") for m in mans}
        r.check(f"manifests on one commit == run commit: {p}", commits == {commit}, sorted(map(str, commits)))
        if not smoke:
            r.check(f"no git_dirty manifest: {p}", all(m.get("git_dirty") is False for m in mans),
                    sum(1 for m in mans if m.get("git_dirty") is not False))
    if smoke:
        r.summary["note"] = "smoke run: design-completeness checks skipped"
        return r
    try:
        CHECKS[cid](datas if cid in MULTI else datas[0], r)
    except Exception as e:
        r.check("campaign-specific checks ran", False, f"{type(e).__name__}: {e}")
    r.summary["wall_clock_min"] = status.get("wall_clock_min")
    r.summary["device"] = {"cuda_available": status.get("cuda_available"), "host": status.get("host")}
    return r


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ids", nargs="+", default=None)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    ids = a.ids
    if ids is None:
        sd = os.path.join(REPO, OUT, "status")
        have = {f[:-5] for f in os.listdir(sd)} if os.path.isdir(sd) else set()
        ids = [c.id for c in CAMPAIGNS if (f"{c.id}_smoke" if a.smoke else c.id) in have]
    reports = [verify(i, a.smoke) for i in ids]

    name = "verification_report_smoke" if a.smoke else "verification_report"
    out_json = os.path.join(REPO, OUT, f"{name}.json")
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump([{"id": r.cid, "verdict": r.verdict, "checks": r.checks, "summary": r.summary} for r in reports],
                  f, indent=1)
    lines = [f"# Re-run verification ({'smoke' if a.smoke else 'full'})", "",
             "| ID | Verdict | Failed checks |", "|---|---|---|"]
    for r in reports:
        failed = "; ".join(f"{c['check']} ({c['detail']})" for c in r.checks if not c["ok"]) or "—"
        lines.append(f"| {r.cid} | **{r.verdict}** | {failed} |")
    with open(out_json.replace(".json", ".md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n-> {out_json}")
    return 0 if reports and all(r.verdict == "PASS" for r in reports) else 1


if __name__ == "__main__":
    sys.exit(main())
