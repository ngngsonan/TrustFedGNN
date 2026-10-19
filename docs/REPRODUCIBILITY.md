# Reproducibility operations

This runbook separates cheap integrity checks from training. Never regenerate an artifact merely to update prose.

## Before any run

```bash
# Local only, from the repository root
source ../.venv-local/bin/activate

git status --short
git rev-parse HEAD
python -m experiments.campaigns.run_campaign --list
python -m pytest -q tests -m 'not slow'
```

Target: Local. A full campaign requires a clean non-results tree. Record the intended campaign, expected commit, target, seed set, and output path before dispatch.

## Local checks and smoke runs

```bash
# Read-only verifier; does not train
python -m experiments.campaigns.verify --ids C20 C21 C22 C23 C23b C24 C25

# Cheap pipeline smoke test; use only when the registry supplies a smoke variant
python -m experiments.campaigns.run_campaign --id C20 --smoke --allow-dirty
```

Target: Local. Smoke artifacts live under `results/campaigns/smoke/` and cannot replace full-run evidence.

## Full CPU campaign on G14

Run these from the workspace root `/Users/anson/DS/Research`:

```bash
python3 .agents/scripts/g14_runner.py sync-to \
  1_Paper/01.GNN/02.TrustFedGNN/FedFairGNN/ \
  --dest 02.TrustFedGNN/FedFairGNN

python3 .agents/scripts/g14_runner.py run \
  "RERUN_EXPECTED_COMMIT=<FULL_COMMIT> python -m experiments.campaigns.run_campaign --id <CAMPAIGN> --threads 8" \
  --name <task_name> --env fedfairgnn --cwd 02.TrustFedGNN/FedFairGNN

python3 .agents/scripts/g14_runner.py status --name <task_name>

python3 .agents/scripts/g14_runner.py fetch \
  02.TrustFedGNN/FedFairGNN/results/campaigns/ \
  1_Paper/01.GNN/02.TrustFedGNN/FedFairGNN/results/campaigns/
```

Target: G14. Replace every placeholder explicitly. After fetching, run the named verifier locally and compare artifact SHA-256 with the fetched status record. The sync helper excludes `.git`, so provenance must remain attributable through the expected commit/environment contract; inspect the resulting embedded manifest before accepting the artifact.

## Full GPU campaign on Colab

The supported flow is implemented by `experiments/colab/00_pack.sh`, `01_setup.py`, and `experiments/campaigns/colab_{launch,poll,collect}.py`.

```bash
# Local: package a clean committed tree
SP=<STAGING_DIR> bash experiments/colab/00_pack.sh

# Colab: after upload/setup, queue one campaign ID per line and launch
python3 experiments/campaigns/colab_launch.py

# Local/Colab control path: poll, collect, then verify the collected campaign
python3 -m experiments.campaigns.verify --ids <CAMPAIGN>
```

Target: packaging and final verification on Local; training on Colab. The package step records commit and tarball SHA-256 and excludes raw data, results, manuscripts, internal docs, and caches. Preserve `manifest_local.json` with collected outputs.

## Acceptance checklist

For each campaign:

1. `status/<ID>.json` has `state: done` and zero exit codes.
2. Every declared output exists and matches the status SHA-256.
3. Artifact manifests agree with the status commit and record `git_dirty: false`.
4. The verifier passes design completeness and protocol invariants for that ID.
5. Any quoted statistic is recomputed from per-seed records with its direction and uncertainty defined.
6. GPU results are treated as numerically reproducible, not bit-exact, unless an explicit bit-exact test proves otherwise.

Verification:

```bash
python -m experiments.campaigns.verify --ids <CAMPAIGN>
python -m json.tool results/campaigns/status/<CAMPAIGN>.json
shasum -a 256 results/campaigns/<ARTIFACT>.json
```

Target: Local or G14. The verifier writes `verification_report.json` and `.md`; record the exact ID list because it defines the report's scope.

## Cost control

- Use Local for reading, artifact analysis, verifier runs, and smoke tests expected under two minutes.
- Use G14 for CPU training, multi-seed studies, grid searches, or checks expected over two minutes.
- Use Colab only for campaigns registered as `gpu-colab` or for a documented GPU requirement.
- Do not launch a new experiment when an existing immutable artifact already answers the question.
- Do not alter an artifact by hand. Change code, commit, rerun into a new artifact, then verify.
