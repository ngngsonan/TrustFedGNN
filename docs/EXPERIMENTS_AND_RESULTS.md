# Experiments and result ledger

This is an artifact-oriented ledger. It describes what each registered campaign tests and where its evidence lives. It does not assign results to a paper section or table.

## Evidence levels

Use these labels consistently:

- **Recorded:** the campaign status is `done`, with commit and output SHA-256 in `results/campaigns/status/`.
- **Verified:** `experiments.campaigns.verify` returned `PASS` for that campaign against the current artifact and status record.
- **Interpreted:** a statistical contrast or scientific conclusion was computed from the artifact. Its estimator, direction, and uncertainty must be named.

At this revision, every C01-C25 campaign (including C23b) has a recorded `done` status. The checked-in `verification_report.json` covers C20, C21, C22, C23, C23b, C24, and C25. Re-run the verifier before using a stronger scope claim.

## Campaign catalogue

| ID | Purpose | Primary artifact(s) | Historical execution target |
|---|---|---|---|
| C01 | Canonical benchmark, component ablations, FSER sign, two-tier defence | `C01_canonical_suite.json` | Local CPU |
| C02 | Pokec-z comparison matrix | `C02_sota_pokecz.json` | Colab GPU |
| C03 | Credit Default comparison matrix | `C03_sota_credit.json` | Colab GPU |
| C04 | Matched-backbone FU-Alignment versus FedAvg control | `C04_aggregator_control_pokecz.json` | Colab GPU |
| C05 | Adaptive stealth poisoner across eleven server rules | `C05_adaptive_poisoner.json` | Local CPU |
| C06 | Dirichlet label-skew and client-count sweep | `C06_dirichlet_sweep.json` | Local CPU |
| C07 | FLTrust/FU-Alignment factorial and EMA control | `C07a_factorial_german.json`, `C07b_factorial_bail.json` | Local CPU |
| C08 | Bail metadata-capture replication | `C08_metadata_capture_bail.json` | Colab GPU/mixed |
| C09 | German task-only FLTrust reproduction | `C09_metadata_german_fltrust.json` | Local CPU |
| C10 | White-box alignment adversary | `C10_alignment_adversary.json` | Local CPU |
| C11 | Partition topology stress | `C11_metis_partition.json` | Local CPU |
| C12 | FLTrust, FLTrust+EMA, and TrustFedGNN stability | `C12_fltrust_ema_pokecz.json` | Colab GPU |
| C13 | Common-definition weight instability | `C13_omega_stability.json` | Local CPU |
| C14 | Task-neutral fairness adversary | `C14_task_neutral.json` | Local CPU |
| C15 | Skewed server-holdout stress | `C15a_skewed_holdout_german.json`, `C15b_skewed_holdout_bail.json` | Local CPU |
| C16 | Server-rule hyperparameter sensitivity | `C16_sensitivity_german.json` | Local CPU |
| C17 | Projection/cosine mechanism probe and round ledger | `C17_mechanism.json` | Local CPU |
| C18 | Adapted GuardFed-style baseline | `C18_guardfed.json` | Local CPU |
| C19 | Connected-holdout topology probe | `C19_holdout_topology.json` | Local CPU |
| C20 | Fixed-state report audit and end-to-end metadata capture | `C20a_fixed_state_replay.json`, `C20b_audit_german.json`, `C20c_audit_bail.json` | Local CPU |
| C21 | DP-induced BFWA slack and heterogeneous privacy contract | `C21a_bfwa_slack_vector.json`, `C21_privacy_contract.json` | Local CPU |
| C22 | Parameter-matched GuardFed-style tuning and confirmation | `C22_guardfed_matched.json` | Local CPU |
| C23 | Mechanism factorial and faithful reference baselines | `C23_mechanism_factorial.json` | Local CPU |
| C23b | Four-arm mechanism subset with full round logging | `C23b_mechanism_subset.json` | Local CPU |
| C24 | Adaptive access schedules S1-S3 | `C24_adaptive_access.json` | G14 CPU |
| C25 | Pokec-z known-label protocol and controls | `C25_preflight_pokec_known.json`, `C25a_sota_pokecz_known.json`, `C25b_control_pokecz_known.json`, `C25c_fltrust_ema_pokecz_known.json` | Colab GPU + Local |

All artifact paths above are relative to `results/campaigns/`. The registry and exact commands are in `experiments/campaigns/campaigns.py`. Historical commits, hosts, wall time, and hashes are in `status/<ID>.json`; campaigns C01-C25 span multiple commits and must not be described as a single-commit experiment suite.

## Audited observations from C20-C25

These statements are deliberately narrower than a submission narrative.

### C20: report dependence is rule-specific

The fixed-state replay changes only the supplied metadata while holding model updates fixed. FairFed, q-FFL/q-FFL-eq, F2GNN, FedGraph-Fair, PoPETs-FairFed, and BFWA variants show non-zero response in the replay; CGSV, FLTrust, root-update FLTrust, and FU-Alignment show zero response. In the end-to-end q-FFL-eq audit, mean adversarial weight changes from `0.199633` to `0.505045` on German and from `0.200245` to `0.501409` on Bail when the report changes from honest to falsified. Root-update FLTrust remains `0.142726` on German and `0.129217` on Bail in both report arms.

Evidence: `C20a_fixed_state_replay.json`, `C20b_audit_german.json`, `C20c_audit_bail.json`; verifier: `chk_C20` in `verify.py`. The q-FFL fixed-state summary contains a non-finite aggregate field, so no q-FFL mean-difference claim should be made from that summary without an explicit finite-value policy.

### C21: privacy noise changes the BFWA contract

Across epsilon values `0.5, 1, 2, 4, 8, 16, 32`, the recorded heterogeneous residual mean decreases from `0.492251` to `0.222162`. At epsilon `8`, it is `0.406840`; all seven cells evaluate 200 rounds and record zero mean exclusions under the configured threshold `tau=0.05`.

Evidence: `C21_privacy_contract.json`; implementation: `experiments/campaigns/c21_privacy_contract.py`; verifier: `chk_C21`.

### C22: GuardFed-style settings were selected per protocol

The validation procedure selects `gs_tau1_gam0.03` for A-German, `gs_tau1_gam0.0` for A-Bail, and `gs_tau10_gam0.001` for B-Bail. These names encode the recorded `gs_tau` and `gs_gamma` settings; they are not universal defaults.

Evidence: `C22_guardfed_matched.json` under `selection`; implementation and score definition: `c22_guardfed_matched.py`; verifier: `chk_C22`.

### C23/C23b: mechanism advantages and trade-offs are separable

Under the 30-seed `fairness_poison` protocol, FU-Alignment versus root-update FLTrust records:

| Dataset | Rule | mean `w_adv` | mean AUC | mean DPD |
|---|---|---:|---:|---:|
| German | FU-Alignment | 0.062994 | 0.641402 | 0.071998 |
| German | root-update FLTrust | 0.189233 | 0.620505 | 0.116519 |
| Bail | FU-Alignment | 0.045125 | 0.723406 | 0.019680 |
| Bail | root-update FLTrust | 0.195904 | 0.699038 | 0.026543 |

Evidence: `C23_mechanism_factorial.json`, paths `results/<dataset>/fairness_poison/{A4_fu_shapley,R1_fltrust_root}`. C23b logs the four mechanism metrics for every round and is checked for agreement with the corresponding C23 cells. Significance and confidence-interval claims must come from the paired statistical analysis, not from the means in this table.

### C24: adaptive access does not support a universal endpoint claim

C24 contains 320 runs: two datasets, four schedules including benign, four methods, and ten seeds. FU-Alignment reduces mean adversarial weight and processed influence relative to FedAvg in several cells, but the direction of AUC and fairness endpoints varies with dataset and schedule. For example, German S1 records FU-Alignment/FedAvg mean DPD `0.080230/0.146399` and AUC `0.640655/0.645981`; German S2 records `w_adv` `0.222147/0.202000`. The appropriate conclusion is conditional mechanism behavior, not universal defence.

Evidence: `C24_adaptive_access.json`; design checks: `tests/test_c24_adaptive_access.py`; acceptance logic: `chk_C24`. Any claim about bootstrap intervals or a defence gate must cite the separately generated endpoint analysis and its exact decision rule.

### C25: known-label evaluation changes the evidentiary basis

The preflight retains 10,262 of 67,796 Pokec-z nodes (`15.1366%`) under the known-label protocol. In the matched FU-Alignment minus FedAvg control, mean AUC difference is `+0.002409` across ten paired seeds; the stored per-seed deltas show seven positive and three negative values. FU-Alignment mean DPD is `0.032066` versus `0.028635` for FedAvg, so the control does not establish a fairness improvement.

Evidence: `C25_preflight_pokec_known.json` and `C25b_control_pokecz_known.json`; protocol test: `tests/test_c25_pokec_protocol.py`; verifier: `chk_C25`. Non-inferiority depends on a prespecified margin and interval method; state both whenever reporting it.

## Statistical conventions

- Summary SD is sample SD (`ddof=1`) unless an artifact explicitly says otherwise.
- Paired comparisons must join by seed before computing differences.
- AUC is higher-better; DPD, EOD, `w_adv`, `I_adv`, and `omega_w` are lower-better in their stated roles.
- A non-significant test is not evidence of equality. Equivalence or non-inferiority requires a prespecified margin.
- Selection/tuning cells and confirmation cells must remain separate.
- GPU output is not assumed bit-exact. Bit-exact claims require an explicit deterministic CPU check.

Implementation: `experiments/stats.py`, `experiments/campaigns/stats_p5.py`, `experiments/campaigns/analysis_endpoints.py`.

## Verification commands

```bash
# Registry and cost/target overview — Local
python -m experiments.campaigns.run_campaign --list

# Verify the currently documented audit block — Local or G14; no training
python -m experiments.campaigns.verify --ids C20 C21 C22 C23 C23b C24 C25

# Inspect the exact status/hash for one campaign — Local
python -m json.tool results/campaigns/status/C24.json

# Run implementation invariants — Local
python -m pytest -q tests -m 'not slow'
```

For training commands and safe dispatch to G14 or Colab, follow `REPRODUCIBILITY.md`.
