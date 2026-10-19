# Baseline implementation and source ledger

Comparison methods are reimplemented in this repository and share the project's loaders, partitions, seeds, and evaluators. This is a fidelity ledger: adapted methods are described as adaptations, and omitted components are explicit. Bibliographic records for a submission must be independently checked against the publisher, DOI registry, proceedings site, or official preprint; citation keys alone are not evidence.

Server rules dispatch through `src/federated/aggregation.py::aggregate`; backbones are in `src/models/gnn.py`; client modes are in `src/federated/client.py`.

## Server rules

| Rule | Config value | Implemented scope | Fidelity boundary | Code/test evidence |
|---|---|---|---|---|
| FedAvg | `fedavg` | sample-size-weighted client-update mean | standard control | `aggregation.py` |
| FLTrust | `fltrust` | non-negative cosine trust against server target and norm rescaling | project target-gradient variant, used only in the audit (C20r) | `aggregation.py`; `test_fltrust_bitexact.py` |
| FLAME | `flame` | cosine-distance clustering, norm clipping, survivor mean | average-linkage substitutes for HDBSCAN; adaptive noise step omitted | `aggregation.py`; `test_flame_aggregator.py` |
| Krum/Multi-Krum | `krum`, `multikrum` | nearest-neighbour squared-distance selection | assumes configured Byzantine count | `aggregation.py` |
| Coordinate median/trimmed mean | `median`, `trimmed_mean` | coordinate-wise robust aggregation | trim fraction is a configuration field | `aggregation.py` |
| FairFed | `fairfed` | reweights from reported local fairness gap | consumes client-reported metadata | `aggregation.py`; C20r |
| q-FFL | `qffl`, `qffl_eq` | loss-powered reweighting | consumes reported loss; `qffl_eq` records pre-training loss | `aggregation.py`; C20r |
| F2GNN | `f2gnn` | fairness-aware weights plus reported group-rate balance | project adaptation | `aggregation.py`; C20r |
| FedGraph-Fair | `fedgraphfair` | DRO multiplier update from reported loss | personalisation and similarity-graph mixing omitted | `aggregation.py`; C20r |
| PoPETs-FairFed | `popets_fairfed` | degree-two polynomial FairFed statistic | threshold-CKKS transport omitted | `aggregation.py`; C20r |
| CGSV | `cgsv` | endogenous cosine-gradient contribution | no server holdout | `aggregation.py`; C20r |
| BFWA | `bfwa`, `robust_bfwa` | Lagrangian/Frank-Wolfe weights from reported telemetry | earlier project method, not an independent external baseline | `aggregation.py`; C20r, C21 |
| GuardFed-style | `guardfed_style` | exponential fairness penalty and survivor threshold | adapted score, explicitly labelled “style” | `aggregation.py`; `test_guardfed_style.py`; C22, C24 |
| FU-Alignment | `fu_shapley` | server-holdout reference, clipping, projection score, signed EMA, rectified weights | proposed server rule | `incentive.py`; `test_incentive.py`; C20r, C23, C24 |
| root-update FLTrust | `fltrust_root` | ReLU cosine against a root update trained on the server holdout, norm rescaling | faithful to Cao et al. (NDSS 2021) | `aggregation.py`, `incentive.py::get_server_root_update`; C20r, C23, C24 |

Root-update FLTrust (`fltrust_root`) computes its reference update by training on the same server holdout as FU-Alignment and is the faithful FLTrust control in every study after the audit. The citations for each rule are in the article's bibliography.
