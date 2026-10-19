# Technical documentation index

These documents describe the research artifact independently of any submission narrative.

| Document | Question answered |
|---|---|
| `CODEBASE_GUIDE.md` | What is implemented, and how does data flow through the system? |
| `EXPERIMENTS_AND_RESULTS.md` | Which experiment produced each artifact, and what can the artifact support? |
| `BASELINES_AND_SOURCES.md` | What part of each comparison method is implemented, adapted, or omitted? |
| `REPRODUCIBILITY.md` | How are runs dispatched, checked, collected, and verified on Local, G14, and Colab? |
| `PROOFS.md` | Which narrow algebraic properties have formal proof artifacts? |

## Source-of-truth order

When records disagree, use this order:

1. source and test at the artifact's recorded commit;
2. immutable artifact plus its embedded configuration and manifest;
3. campaign status record and SHA-256;
4. verifier report for the named campaign IDs;
5. technical documentation;
6. submission prose.

An artifact generated at an older commit must be interpreted with that commit's source. Current code is not retroactive evidence for historical execution.

## Fast integrity check

```bash
source ../.venv-local/bin/activate
python -m experiments.campaigns.run_campaign --list
python -m experiments.campaigns.verify --ids C20 C21 C22 C23 C23b C24 C25
python -m pytest -q tests -m 'not slow'
```

Target: Local. Use G14 for tests or computations expected to exceed two minutes; see `REPRODUCIBILITY.md`.
