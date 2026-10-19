# Baseline implementation and source ledger

Comparison methods are reimplemented in this repository and share the project's loaders, partitions, seeds, and evaluators. This is a fidelity ledger: adapted methods are described as adaptations, and omitted components are explicit. Bibliographic records for a submission must be independently checked against the publisher, DOI registry, proceedings site, or official preprint; citation keys alone are not evidence.

Method presets are in `experiments/methods.py`. Server rules dispatch through `src/federated/aggregation.py::aggregate`; backbones are in `src/models/gnn.py`; client modes are in `src/federated/client.py`.

## Server rules

| Rule | Config value | Implemented scope | Fidelity boundary | Code/test evidence |
|---|---|---|---|---|
| FedAvg | `fedavg` | sample-size-weighted client-update mean | standard control | `aggregation.py`; `test_baselines_integrity.py` |
| FLTrust | `fltrust` | non-negative cosine trust against server target and norm rescaling | project target-gradient variant; C23 adds a faithful root-update control | `aggregation.py`; `test_fltrust_bitexact.py`; C23 |
| FLAME | `flame` | cosine-distance clustering, norm clipping, survivor mean | average-linkage substitutes for HDBSCAN; adaptive noise step omitted | `aggregation.py`; `test_flame_aggregator.py` |
| Krum/Multi-Krum | `krum`, `multikrum` | nearest-neighbour squared-distance selection | assumes configured Byzantine count | `aggregation.py`; `test_baselines_integrity.py` |
| Coordinate median/trimmed mean | `median`, `trimmed_mean` | coordinate-wise robust aggregation | trim fraction is a configuration field | `aggregation.py`; `test_baselines_integrity.py` |
| FairFed | `fairfed` | reweights from reported local fairness gap | consumes client-reported metadata | `aggregation.py`; C20 |
| q-FFL | `qffl`, `qffl_eq` | loss-powered reweighting | consumes reported loss; `qffl_eq` records pre-training loss | `aggregation.py`; C20 |
| F2GNN | `f2gnn` | fairness-aware weights plus reported group-rate balance | project adaptation | `aggregation.py`; C20 |
| FairGFL | `fairgfl` | inverse-imbalance weighting | substitutes data imbalance for original inter-graph overlap | `aggregation.py`; baseline integrity tests |
| FedGraph-Fair | `fedgraphfair` | DRO multiplier update from reported loss | personalisation and similarity-graph mixing omitted | `aggregation.py`; C20 |
| PoPETs-FairFed | `popets_fairfed` | degree-two polynomial FairFed statistic | threshold-CKKS transport omitted | `aggregation.py`; C20 |
| CGSV | `cgsv` | endogenous cosine-gradient contribution | no server holdout | `aggregation.py`; C20 |
| BFWA | `bfwa`, `robust_bfwa` | Lagrangian/Frank-Wolfe weights from reported telemetry | earlier project method, not an independent external baseline | `aggregation.py`; C20-C21 |
| GuardFed-style | `guardfed_style` | exponential fairness penalty and survivor threshold | adapted score, explicitly labelled “style” | `aggregation.py`; `test_guardfed_style.py`; C18/C22 |
| FU-Alignment | `fu_shapley` | server-holdout target, clipping/rescaling, score EMA, rectified weights | proposed server rule | `incentive.py`; `test_incentive.py`; C23/C24 |

The external method identities and local citation keys are maintained in `experiments/methods.py` and manuscript bibliographies. Before publication, verify each author list, title, venue, year, pages, DOI/URL, and that the cited source actually defines the implemented component.

## Client and model comparisons

| Method | Preset | Implemented scope | Fidelity boundary | Code/test evidence |
|---|---|---|---|---|
| FairGNN | `fairgnn` | adversarial sensitive-attribute debiasing | sensitive estimator omitted because the protocol observes the attribute | `gnn.py`, `client.py`; method registry tests |
| FairSIN | `fairsin` | heterophilous-neighbour feature augmentation | local project adaptation | `gnn.py`; method registry tests |
| FaVGNN | `favgnn` | completion-driven heterogeneous feature fusion plus adversary | adapted from vertical to horizontal federation | `gnn.py`; method registry tests |
| DP-FedAvg | `dp-fedavg` | clipped noisy full-gradient training | privacy control for FTGD | `client.py`, `privacy.py`; DP tests |
| FDP-Fair | `fdp-fair` | DP training and group-offset calibration | binary project implementation | `client.py`; method registry tests |
| PUFFLE | `puffle` | DP-SGD plus fairness-weight controller | sensitive-count sharing omitted | `client.py`; method registry tests |
| FedFACT | `fedfact` | global and local binary post-hoc offsets | general multiclass solver omitted | `client.py`; method registry tests |
| EquFL-style calibration | configured server calibration | late server fairness-gradient correction | project adaptation of the calibration idea | trainer/config; experiment integrity tests |
| TrustFedGNN | `fedfairgnn`/canonical config | FSER backbone, FTGD client mode, FU-Alignment server rule | proposed composite method | canonical-config, core, incentive, and mechanism tests |

## Required citation audit

The current historical bibliography is not a trusted source of truth. Before reusing it:

1. parse every BibTeX entry and reject malformed or duplicate records;
2. verify metadata through DOI/Crossref, official proceedings, publisher pages, OpenReview, PMLR, or arXiv as appropriate;
3. match each local implementation to the exact cited algorithm and record adaptations in the paper;
4. remove uncited entries and resolve every in-text key;
5. store DOI as a full `https://doi.org/...` link when one exists.

Verification commands:

```bash
# Local: syntax and duplicate-key gate; replace TARGET with the active bibliography
python3 - <<'PY' manuscript_discoverAI/ref.bib
import re, sys
p=sys.argv[1]; s=open(p, encoding='utf-8').read()
keys=re.findall(r'@[A-Za-z]+\s*\{\s*([^,\s]+)', s)
dups=sorted({k for k in keys if keys.count(k)>1})
print({'entries': len(keys), 'duplicate_keys': dups})
raise SystemExit(bool(dups))
PY

# Local: make sure each configured method has implementation coverage
python3 -m pytest -q tests/test_baselines_integrity.py tests/test_method_registry_invariants.py
```

Bibliographic correctness remains an evidence gap until the external-source audit is completed; passing the local parser does not close it.
