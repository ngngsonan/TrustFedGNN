"""Campaign registry for the TrustFedGNN article.

Single source of truth for WHAT is run: command line, where it runs, where the
artifact lands, which audit finding it closes, and a smoke variant used to
check the pipeline before the full run. verify.py holds the matching
acceptance checks; docs/REPRODUCIBILITY.md describes HOW to run.

All paths are relative to the repository root. Every full-run artifact is
written under results/campaigns/. A corrected re-run gets a new ID (C20r) and
never overwrites the artifact it supersedes.

Runtime estimates are derived from wall-clock times recorded in the previous
artifacts (manifests); campaigns marked "est" were never timed at
this protocol -- the smoke run gives the per-run time to extrapolate from.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

OUT = "results/campaigns"
SMOKE = f"{OUT}/smoke"
SEEDS_10 = [str(s) for s in range(42, 52)]


@dataclass(frozen=True)
class Campaign:
    id: str
    title: str
    priority: str                 # P0 (blocker) | P1 (major) | P2 (supporting)
    findings: tuple               # revision-tracking IDs this campaign addresses
    where: str                    # "cpu-local" | "gpu-colab"
    cmds: tuple                   # tuple of argv lists, run in order
    outputs: tuple                # artifact paths produced by cmds
    smoke_cmds: tuple             # cheap variant: same code path, tiny budget
    smoke_outputs: tuple
    est_minutes: str              # human estimate, with source
    env: Dict[str, str] = field(default_factory=dict)
    feeds: str = ""               # manuscript tables / sentences that consume it
    notes: str = ""


PY = "python"

CAMPAIGNS: List[Campaign] = [
    Campaign(
        id="C05", title="Tier-2 adaptive stealth poisoner, 11 rules incl. FLAME, one commit",
        priority="P0", findings=("A-01", "A-05", "A-16", "A-18"), where="cpu-local",
        cmds=([PY, "experiments/revision/adaptive_poisoner.py",
               "--aggregators", "fedavg", "krum", "multikrum", "median", "trimmed_mean", "flame",
               "bfwa", "robust_bfwa", "fltrust", "fu_shapley", "robust_fu_shapley",
               "--byz-ratios", "0.1", "0.2", "0.3", "0.4", "--seeds", *SEEDS_10, "--rounds", "15",
               "--out-json", f"{OUT}/C05_adaptive_poisoner.json",
               "--out-tex", f"{OUT}/tables/C05_adaptive_poisoner.tex"],),
        outputs=(f"{OUT}/C05_adaptive_poisoner.json",),
        smoke_cmds=([PY, "experiments/revision/adaptive_poisoner.py",
                     "--aggregators", "fedavg", "flame", "fltrust", "fu_shapley", "robust_fu_shapley",
                     "--byz-ratios", "0.2", "--seeds", "42", "--rounds", "2",
                     "--out-json", f"{SMOKE}/C05_smoke.json", "--out-tex", f"{SMOKE}/C05_smoke.tex"],),
        smoke_outputs=(f"{SMOKE}/C05_smoke.json",),
        est_minutes="~130 (old: 17.6 s/run x 440 runs)",
        feeds="tab_adaptive_poisoner Panels A+B, fig_robustness_byz, sec4:143-148",
        notes="Protocol kept as published (Bail, K=10, R=15, E=1 hard-coded in the runner); the manuscript "
              "must state E=1/R=15 for this experiment (audit A-14). FLTrust pinned to fu_alpha=0.0.",
    ),
    Campaign(
        id="C20", title="End-to-end report-capture audit (q-FFL-eq, root-FLTrust; German, Bail)",
        priority="P0", findings=("E04", "E09", "E10"), where="cpu-local",
        cmds=(
            [PY, "experiments/revision/metadata_capture_endtoend.py", "--dataset", "german",
             "--rules", "qffl_eq", "fltrust_root", "--out-json", f"{OUT}/C20b_audit_german.json"],
            [PY, "experiments/revision/metadata_capture_endtoend.py", "--dataset", "bail",
             "--rules", "qffl_eq", "fltrust_root", "--out-json", f"{OUT}/C20c_audit_bail.json"],
        ),
        outputs=(
            f"{OUT}/C20b_audit_german.json",
            f"{OUT}/C20c_audit_bail.json",
        ),
        smoke_cmds=(),
        smoke_outputs=(),
        est_minutes="~10 (Mac CPU)",
        feeds="RQ1 end-to-end invariance of root-FLTrust (text); q-FFL-eq arm not used as evidence",
        notes="The fixed-state replay that used to be C20a is superseded by C20r.",
    ),
    Campaign(
        id="C20r", title="Fixed-update report audit, re-run with a falsified report for every rule and per-run seeding",
        priority="P0", findings=("C20a-no-lie", "C20a-unseeded"), where="cpu-local",
        cmds=([PY, "experiments/campaigns/c20_fixed_state_replay.py", "--workers", "6", "--rounds", "20",
               "--out-json", f"{OUT}/C20r_fixed_state_replay.json"],),
        outputs=(f"{OUT}/C20r_fixed_state_replay.json",),
        smoke_cmds=([PY, "experiments/campaigns/c20_fixed_state_replay.py", "--smoke",
                     "--out-json", f"{SMOKE}/C20r_smoke.json"],),
        smoke_outputs=(f"{SMOKE}/C20r_smoke.json",),
        est_minutes="~10-20 (Mac CPU, 6 workers)",
        feeds="tab_c20_report_audit, abstract, RQ1",
        notes="Supersedes C20a: rules outside BEST_RESPONSE_LIE receive a union lie (n kept honest); every run calls set_seed.",
    ),
    Campaign(
        id="C21", title="BFWA DP-induced slack vector and heterogeneous privacy contract",
        priority="P0", findings=("E17",), where="cpu-local",
        cmds=(
            [PY, "experiments/revision/bfwa_slack_analysis.py", "--log-sigma-vector",
             "--dataset", "german", "--num-clients", "10", "--rounds", "20", "--tau", "0.05",
             "--out-json", f"{OUT}/C21a_bfwa_slack_vector.json"],
            [PY, "experiments/campaigns/c21_privacy_contract.py",
             "--in-json", f"{OUT}/C21a_bfwa_slack_vector.json",
             "--out-json", f"{OUT}/C21_privacy_contract.json"],
        ),
        outputs=(
            f"{OUT}/C21a_bfwa_slack_vector.json",
            f"{OUT}/C21_privacy_contract.json",
        ),
        smoke_cmds=(
            [PY, "experiments/revision/bfwa_slack_analysis.py", "--log-sigma-vector",
             "--dataset", "german", "--seeds", "42", "--epsilons", "8.0", "--rounds", "2",
             "--num-clients", "10", "--tau", "0.05",
             "--out-json", f"{SMOKE}/C21a_smoke.json"],
            [PY, "experiments/campaigns/c21_privacy_contract.py",
             "--in-json", f"{SMOKE}/C21a_smoke.json",
             "--out-json", f"{SMOKE}/C21_smoke.json"],
        ),
        smoke_outputs=(f"{SMOKE}/C21_smoke.json",),
        est_minutes="~5 (Mac CPU)",
        feeds="tab_ldp_barrier, sec3:Theorem 4 / Le Cam, sec4:RQ2",
        notes="Extracts client sigma vectors across epsilons and evaluates heterogeneous minimax risk floor R*_het.",
    ),
    Campaign(
        id="C22", title="GuardFed parameter-matched baseline (tuning + confirm, German & Bail)",
        priority="P0", findings=("E18",), where="cpu-local",
        cmds=(
            [PY, "experiments/campaigns/c22_guardfed_matched.py",
             "--workers", "6", "--out-json", f"{OUT}/C22_guardfed_matched.json"],
        ),
        outputs=(f"{OUT}/C22_guardfed_matched.json",),
        smoke_cmds=(
            [PY, "experiments/campaigns/c22_guardfed_matched.py",
             "--smoke", "--workers", "2", "--out-json", f"{SMOKE}/C22_smoke.json"],
        ),
        smoke_outputs=(f"{SMOKE}/C22_smoke.json",),
        est_minutes="~25 (Mac CPU, 6 workers)",
        feeds="tab_sota_main, tab_adaptive_poisoner, sec4:RQ3",
        notes="Tunes 14 GuardFed configs on seeds 1000..1004, confirms on 30 seeds (Part A) and 10 seeds (Part B), plus fltrust_root stealth.",
    ),
    Campaign(
        id="C23", title="Mechanism factorial: 2x2 order x magnitude + faithful baselines (German, Bail, 30 seeds)",
        priority="P0", findings=("E19",), where="cpu-local",
        cmds=(
            [PY, "experiments/campaigns/c23_mechanism_factorial.py",
             "--workers", "6", "--out-json", f"{OUT}/C23_mechanism_factorial.json"],
        ),
        outputs=(f"{OUT}/C23_mechanism_factorial.json",),
        smoke_cmds=(
            [PY, "experiments/campaigns/c23_mechanism_factorial.py",
             "--smoke", "--workers", "2", "--out-json", f"{SMOKE}/C23_smoke.json"],
        ),
        smoke_outputs=(f"{SMOKE}/C23_smoke.json",),
        est_minutes="~35 (Mac CPU, 6 workers)",
        feeds="tab_factorial_2x2, Table 3, sec4:RQ4",
        notes="Evaluates 9 arms across fairness_poison and clean scenarios on German & Bail with log_mechanism=True.",
    ),
    Campaign(
        id="C23b", title="Mechanism subset: 4 arms fairness_poison (German, Bail, 30 seeds)",
        priority="P0", findings=("E19",), where="cpu-local",
        cmds=(
            [PY, "experiments/campaigns/c23b_mechanism_subset.py",
             "--workers", "6", "--out-json", f"{OUT}/C23b_mechanism_subset.json"],
        ),
        outputs=(f"{OUT}/C23b_mechanism_subset.json",),
        smoke_cmds=(
            [PY, "experiments/campaigns/c23b_mechanism_subset.py",
             "--smoke", "--workers", "2", "--out-json", f"{SMOKE}/C23b_smoke.json"],
        ),
        smoke_outputs=(f"{SMOKE}/C23b_smoke.json",),
        est_minutes="~20 (Mac CPU, 6 workers)",
        feeds="tab_mechanism_factorial, Table 3",
        notes="Evaluates 4 key arms under fairness_poison with fixed mechanism logger and bit-exact gating against C23.",
    ),
    Campaign(
        id="C25", title="Pokec-z known-label protocol: preflight and FU-Alignment vs FedAvg control",
        priority="P0", findings=("E32",), where="gpu-colab",
        cmds=(
            [PY, "experiments/campaigns/c25_preflight.py"],
            [PY, "experiments/colab/run_revision_gpu.py", "--job", "pokecz_control", "--device", "cuda",
             "--pokec-label-protocol", "known_only", "--pokecz-out", f"{OUT}/C25b_control_pokecz_known.json"],
        ),
        outputs=(
            f"{OUT}/C25_preflight_pokec_known.json",
            f"{OUT}/C25b_control_pokecz_known.json",
        ),
        smoke_cmds=([PY, "experiments/campaigns/c25_preflight.py"],),
        smoke_outputs=(f"{OUT}/C25_preflight_pokec_known.json",),
        est_minutes="~20 on a T4 GPU",
        feeds="tab_c25_known_labels; known-label counts in Section 4.1",
        notes="Only the preflight and the aggregator control are retained for the article.",
    ),
    Campaign(
        id="C24", title="Adaptive access sweeps (S1/S2/S3 schedules, 4 methods, German, Bail, 10 seeds)",
        priority="P0", findings=("W4",), where="cpu-local",
        cmds=(
            [PY, "experiments/campaigns/c24_adaptive_access.py",
             "--workers", "6", "--out-json", f"{OUT}/C24_adaptive_access.json"],
        ),
        outputs=(f"{OUT}/C24_adaptive_access.json",),
        smoke_cmds=(
            [PY, "experiments/campaigns/c24_adaptive_access.py",
             "--smoke", "--workers", "1", "--out-json", f"{SMOKE}/C24_smoke.json"],
        ),
        smoke_outputs=(f"{SMOKE}/C24_smoke.json",),
        est_minutes="~180 on an 8-core Linux CPU",
        feeds="tab_adaptive_access, Table 4",
        notes="Evaluates defence resilience under constrained adversary access schedules (S1, S2, S3) and clean baseline.",
    ),
    Campaign(
        id="C26", title="FU-Alignment fairness-reference isolation (alpha=0, German and Bail, 30 seeds)",
        priority="P0", findings=("DiscoverAI-B4",), where="cpu-local",
        cmds=([PY, "experiments/campaigns/c26_fu_alpha0.py", "--workers", "6",
               "--out-json", f"{OUT}/C26_fu_alpha0.json"],),
        outputs=(f"{OUT}/C26_fu_alpha0.json",),
        smoke_cmds=([PY, "experiments/campaigns/c26_fu_alpha0.py", "--smoke", "--workers", "1",
                     "--out-json", f"{SMOKE}/C26_smoke.json"],),
        smoke_outputs=(f"{SMOKE}/C26_smoke.json",),
        est_minutes="~4 (60 CPU runs; one C23 arm)",
        feeds="fairness-reference ablation in main Results and SI",
        notes="Pairs with C23 A4_fu_shapley at identical seeds and protocol; only fu_alpha changes.",
    ),
]

BY_ID: Dict[str, Campaign] = {c.id: c for c in CAMPAIGNS}

def get(cid: str) -> Optional[Campaign]:
    return BY_ID.get(cid)
