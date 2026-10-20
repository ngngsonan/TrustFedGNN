# Reproducibility

This runbook separates cheap integrity checks, which read the stored artifacts, from re-running a campaign, which retrains models.

## 1. Check the stored evidence (no training)

```bash
pip install -r requirements.txt
python -m experiments.campaigns.verify                 # every campaign that has a status file
python -m experiments.campaigns.analyze_discover_ai    # results/campaigns/analysis/discover_ai.json
python -m experiments.campaigns.make_tables            # regenerate every table into build/tables/
pytest -m "not slow"
```

`make_tables --check` additionally diffs the output with `manuscript/tables/`, which exists only once the manuscript sources are released.

`verify` writes `results/campaigns/verification_report.{json,md}`. For each campaign it checks that:

1. `status/<ID>.json` records `state: done`;
2. every declared output exists and matches the SHA-256 recorded at run time;
3. every manifest inside the artifact carries the status commit with `git_dirty: false`;
4. the design is complete (datasets × arms × seeds) and the protocol invariants hold;
5. the summary statistics recompute from the per-seed records.

## 2. Re-run a campaign

```bash
python -m experiments.campaigns.run_campaign --list            # IDs, commands, runtime estimates
python -m experiments.campaigns.run_campaign --id C20r --smoke --allow-dirty   # cheap variant, same code path
python -m experiments.campaigns.run_campaign --id C20r         # full run; refuses a dirty tree
python -m experiments.campaigns.verify --ids C20r
```

A full run records the commit, the command lines, the exit codes and the output hashes in `results/campaigns/status/<ID>.json`, and its log in `results/campaigns/logs/`. An existing output produced by another commit is moved aside, never overwritten. Smoke artifacts live in `results/campaigns/smoke/` and are not evidence.

## 3. Environment of each campaign

Client partitions are drawn with torch's random number generator. The same seed can therefore give a different split under a different torch version, and a re-run is expected to reproduce a campaign's conclusions but not its exact numbers unless the environment matches. Each manifest records its platform and torch version. Comparisons in the article are made only within a campaign, where every rule shares the environment and the seeds.

| Campaign | Study in the article | Environment |
|---|---|---|
| C20r | fixed-update report audit | macOS arm64, CPU, torch 2.14.0 |
| C20 (C20b/c) | end-to-end audit (root-FLTrust invariance) | macOS arm64, CPU, torch 2.14.0 |
| C21 | report-privacy diagnostic (SI) | macOS arm64, CPU, torch 2.14.0 |
| C22 | GuardFed-style tuning (SI) | macOS arm64, CPU, torch 2.14.0 |
| C23, C23b, C26 | controlled fairness poisoning, logged replay, α = 0 | macOS arm64, CPU, torch 2.14.0 |
| C24 | adaptive access | Linux x86-64, CPU, torch 2.5.1 |
| C25 | Pokec-z known-label control | preflight: macOS CPU; control: Colab T4 GPU, torch 2.11.0 |
| C05 | supporting stealth sweep (SI) | macOS arm64, CPU, torch 2.14.0 |

GPU results are numerically reproducible, not bit-exact.

## 4. Known limitations of the stored artifacts

These are documented rather than fixed by a re-run, because they do not affect a reported number. Each is handled where it matters.

- **C24, reference evasion × FedAvg.** The adversaries' report omitted the sample count `n`, which FedAvg reads (`aggregate` defaults it to 1). These 20 runs are excluded from every table and claim (`analyze_discover_ai.py::C24_INVALID_CELLS`). The other three methods do not read `n` (`tests/test_campaigns.py`). The runner has since been fixed.
- **C24 round indexing.** Stored rounds are 1-indexed, while the schedules are defined on the 0-indexed round of the trainer; `active_poison_round` converts between them (`tests/test_campaigns.py::test_c24_active_round_boundary`).
- **C23 `I_adv` field.** It was written before the mechanism logger was fixed and holds the adversarial weight; the article takes processed influence from C23b and C24 only.
- **C20 (C20b/c), q-FFL-eq arm.** Its updates are about 10⁻³ of root-FLTrust's and its clean AUC is near chance (0.54 German, 0.51 Bail). The arm is not used as evidence; only the root-FLTrust invariance of this campaign is cited.
- **C25 control.** The manifest does not store the label protocol. The registry passes `--pokec-label-protocol known_only` to `experiments/colab/run_revision_gpu.py`, which forwards it to `ExperimentConfig`; the preflight records 2,566 known-label test nodes.
- **C21.** The decision residual is measured on a fresh noise draw from each client's post-training model, not on the release that BFWA consumed.
- **C26** reproduces, bit for bit, the `A3_fu_alpha0` arm of an earlier development campaign (C07), which is not part of this repository.
