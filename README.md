# TrustFedGNN

Code and per-seed results for the article *TrustFedGNN: Auditing and Mitigating Report-Dependent Trust in Federated Graph Learning* (Nguyen and Dang, submitted to *Discover Artificial Intelligence*, collection "Trustworthy and Responsible Federated Learning").

## In 90 seconds

- **Problem.** Several fairness-aware federated aggregation rules weight clients by statistics the clients report about themselves (loss, disparity). The server cannot check those reports.
- **Audit.** A fixed-update intervention replays identical client updates and server state twice and changes only the adversary's report. Every report-dependent rule moves weight towards the adversary; rules that read no report do not move.
- **Method.** FU-Alignment scores each clipped client update by its projection onto a reference gradient that the server computes on a small confidential holdout graph. It smooths the signed score over rounds and weights clients by the positive part. It never reads a client report.
- **Evidence.** Under controlled fairness poisoning, FU-Alignment gives the adversary less weight than root-update FLTrust on both datasets. A factorial ablation attributes this to clipping and to smoothing before rectification. An adaptive-access study maps where the advantage holds and where it fails.

Every number in the article traces to a file in `results/campaigns/` through [`docs/CLAIMS.md`](docs/CLAIMS.md).

## Quick verification (CPU, no training, about one minute)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m experiments.campaigns.verify                 # integrity and design checks for every campaign
python -m experiments.campaigns.analyze_discover_ai    # paired statistics -> results/campaigns/analysis/
python -m experiments.campaigns.make_tables            # every table of the article -> build/tables/
pytest -m "not slow"                                   # unit and invariant tests
```

The verifier recomputes each artifact's SHA-256 and compares it with the status file written when the campaign ran. It also checks that every manifest inside the artifact carries one clean commit, and that the design is complete (cells × seeds). `make_tables` fails instead of printing a caption the numbers do not support.

## Names in the code and in the article

| Article | Code |
|---|---|
| FU-Alignment | aggregator `fu_shapley` (historical name; the score is a first-order projection, **not** a Shapley value) |
| root-update FLTrust | `fltrust_root` (`fltrust` is an earlier gradient-reference variant, used only in the audit) |
| GuardFed-style adaptation | `guardfed_style` |
| FSER, FTGD (fixed client stack, adapted from prior work) | `src/models/gnn.py::FSERLayer`, `src/federated/client.py::_ftgd_step` |
| holdout $D_{\rm root}$ | `src/data/partition.py::carve_server_holdout` |
| clipping, score, signed EMA, weights and update | `src/trust/incentive.py::compute_fu_weights`, `src/federated/aggregation.py` (branch `fu_shapley`) |
| task and fairness reference gradients | `src/trust/incentive.py::get_server_target_gradients` |
| $w_{\rm adv}$, $I_{\rm adv}$ | `src/federated/trainer.py` (`w_adv_round`, `I_adv`) |

## Layout

```
src/                     model, clients, aggregation rules, attacks, data loading and partitioning
experiments/campaigns/   one runner per campaign, the registry, verifier, statistics and table generator
experiments/revision/    runners reused by C05, C20b/c and C21
results/campaigns/       per-seed artifacts (C*.json), status files, derived analysis
tests/                   unit and invariant tests
docs/                    CLAIMS (number -> artifact -> command), reproduction, code guide, baseline sources
```

Datasets are downloaded on first use from the public NIFTY and FairGNN releases into `data/raw/` (see `src/data/datasets.py`).

## Reproducing a campaign

`python -m experiments.campaigns.run_campaign --list` prints every campaign with its command line and runtime estimate. `--id <ID> --smoke` runs a cheap variant of the same code path. A full run refuses to start on a dirty tree. See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for the environment of each campaign. Client partitions are drawn with torch's random number generator, so exact numbers depend on the torch version recorded in each manifest.

## Provenance note

The artifacts were produced in the authors' development repository before this snapshot was created. Their manifests therefore record development-repository commits. The two exceptions are C20r and anything produced later, which record commits of this repository. Each status file lists the exact command lines that produced its artifacts. A few artifact files still name an older results directory in their internal path strings; the files themselves live in `results/campaigns/`, and their hashes are unchanged.

## License

MIT, see [LICENSE](LICENSE).
