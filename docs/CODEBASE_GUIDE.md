# Codebase guide

This document describes the implemented research system. Source code is authoritative when it differs from this guide. The aggregator called FU-Alignment in the article is `fu_shapley` in the code.

## System map

```text
TrustFedGNN/
├── src/
│   ├── config.py                 serialisable ExperimentConfig and canonical preset
│   ├── data/                     dataset loaders, graph partitions, server holdout, sampler
│   ├── models/gnn.py             TrustFedGNN/FSER and comparison backbones
│   ├── federated/                clients, trainer, attacks, aggregation rules
│   ├── trust/                    FU-Alignment, privacy accounting, diagnostics
│   └── utils/                    metrics, logging, provenance
├── experiments/
│   ├── campaigns/                registry, runners, verifier, statistics, table generator
│   ├── revision/                 runners reused by C05, C20b/c and C21
│   └── colab/                    provenance-preserving Colab packaging
├── results/campaigns/            immutable JSON artifacts and campaign status records
├── tests/                         executable invariants
└── docs/                          technical reference and the claim ledger
```

## Execution path

1. A runner creates `ExperimentConfig.canonical(...)` and records explicit overrides.
2. `src/data/datasets.py` loads a graph. `src/data/partition.py` creates client partitions and, when requested, a server-owned holdout.
3. `src/federated/client.py` trains each client and returns a model update plus the metadata enabled by the selected rule.
4. `src/federated/attacks.py` transforms Byzantine updates or reports under the configured threat model.
5. `src/federated/aggregation.py::aggregate` dispatches the server rule. FU-Alignment calls `src/trust/incentive.py`.
6. `src/federated/trainer.py` applies the aggregate and records endpoint and round-level metrics.
7. The runner writes a JSON artifact containing the full configuration and a provenance manifest. `run_campaign.py` writes a separate status record with output SHA-256 hashes.

Evidence: `src/config.py`, `src/federated/trainer.py`, `src/federated/aggregation.py`, `src/utils/provenance.py`, and `experiments/campaigns/run_campaign.py`.

## Canonical TrustFedGNN preset

`ExperimentConfig.canonical()` currently fixes the following fields; every other field comes from the dataclass default unless the runner overrides it.

```python
dataset="german"
num_clients=5
rounds=20
dirichlet_alpha=0.3
model="trustfedgnn"
aggregator="fu_shapley"
fu_alpha=0.1
fu_ema_beta=0.9
fairness_weight=1.0
beta_init=0.5
fser_mode="sub"
dp_enabled=True
dp_mode="ftgd"
fu_val_source="server_holdout"
```

Do not infer a run's protocol from this preset. Runners override client count, rounds, local epochs, privacy, attack, partition, and device. The artifact's `config` object is the evidence for a reported number.

Verification:

```bash
python3 - <<'PY'
from src.config import ExperimentConfig
print(ExperimentConfig.canonical().to_dict())
PY
```

Target: any machine with the pinned requirements.

## TrustFedGNN components

### FU-Alignment server rule

For client update `Delta_k = theta_global - theta_local`, the implementation performs:

1. build `g_target = g_task + alpha * g_fair` on the server holdout;
2. clip `Delta_k` to norm `C` when `fu_magnitude="clip"`;
3. score the update using the configured projection or cosine rule;
4. update a signed exponential moving average;
5. rectify, mask degenerate updates, and normalise weights to the simplex;
6. apply the weighted aggregate;
7. optionally log utility/fairness score components, adversarial weight, adversarial influence, aggregate norm, cosine diagnostics, and timing.

The exact branches are in `src/trust/incentive.py::compute_fu_weights` and the `fu_shapley` branch of `src/federated/aggregation.py::aggregate`. Tests: `tests/test_incentive.py`, `tests/test_mechanism_logger.py`, `tests/test_c23_runner.py`.

### FSER backbone

`src/models/gnn.py::FSERLayer` implements the sensitive-relation attention component used by the TrustFedGNN backbone. `fser_mode`, `beta_init`, and `freeze_beta` control the relation term and its ablations. Tests: `tests/test_fser_init.py`, `tests/test_core.py`.

### FTGD client mode

`src/federated/client.py::_gradient_surgery` removes the task-gradient component parallel to the fairness gradient and then adds the fairness direction. When the fairness-gradient norm falls below `ftgd_min_fair_norm`, the projection is skipped and counted. The released group statistics use Gaussian noise calibrated in `src/trust/privacy.py`; `dp_statistic_s_blind=True` keeps the released statistic independent of the sensitive-aware FSER path. Tests: `tests/test_core.py`, `tests/test_dp_mode_resolution.py`, `tests/test_no_single_feature_leakage.py`.

## Data and evaluation contracts

Supported loader names are defined in `src/data/datasets.py`; the article uses German Credit, Bail and Pokec-z. Large-graph runs can enable the dependency-free neighbour sampler in `src/data/sampler.py`.

Primary endpoint fields are:

- `auc`: ROC AUC; higher is better;
- `dpd_hard`: demographic-parity difference on hard predictions; lower is better;
- `eod`: equal-opportunity difference; lower is better;
- `w_adv`: aggregation weight assigned to Byzantine clients;
- `I_adv`: norm-based processed adversarial influence used by the mechanism audits;
- `omega_w`: temporal instability of client weights under the definition in `src/utils/metrics.py`.

Definitions and edge-case handling are executable in `src/utils/metrics.py`. A number should be quoted only with its artifact path, protocol cell, seed set, and summary rule.

Pokec-z has two label protocols. `legacy_negative` maps unknown labels through the older negative-label path. `known_only` filters evaluation and training splits to nodes with known labels. C25 records 10,262 known nodes out of 67,796 (`0.1513658623`) in `results/campaigns/C25_preflight_pokec_known.json`. Tests: `tests/test_c25_pokec_protocol.py`.

## Provenance and failure behavior

`src/utils/provenance.py::build_manifest` records commit, dirty state, device, timestamp, and configuration context. `run_campaign.py` refuses a full run when provenance is unknown, the non-results tree is dirty, `RERUN_EXPECTED_COMMIT` does not match, or a required CUDA device is absent. Existing outputs from another commit are moved to a timestamped stale file.

Status and artifact have different roles:

- `results/campaigns/status/<ID>.json` proves process completion and pins output hashes;
- the artifact proves protocol and observations;
- `verify.py` checks hashes, manifests, design completeness, configuration invariants, and selected recomputations;
- `verification_report.json` proves only the campaign IDs present in that report. A status of `done` is not equivalent to a current verifier `PASS`.

## Test surface

```bash
python -m pytest -q tests -m 'not slow'
python -m experiments.campaigns.run_campaign --list
python -m experiments.campaigns.verify
```

The verifier is CPU-only. It reads existing artifacts and does not retrain models. See `REPRODUCIBILITY.md` before launching any campaign.
