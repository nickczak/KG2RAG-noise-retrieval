# Experiment records

Use the repository's [shared workflow](../WORKFLOW.md) for new experiments.
The three historical development runs below are complete and should be preserved.

```bash
./experiment compare baseline50 sem50 sem50_t03
```

| Directory | Method | Threshold | Questions | Answer F1 | Evidence precision | Evidence recall | Mean sentences |
|---|---|---:|---:|---:|---:|---:|---:|
| `baseline50` | Original KG²RAG pipeline | — | 50 | 52.55% | 21.30% | 83.00% | 9.58 |
| `sem50` | SEM before existing graph organization | 0.5 | 50 | 43.23% | 36.67% | 58.20% | 5.38 |
| `sem50_t03` | SEM before existing graph organization | 0.3 | 50 | 52.07% | 22.29% | 77.93% | 8.76 |

All three have 50 answer/evidence predictions and no local checkpoint errors.
They share the question-content and loaded-graph fingerprints, model names, and
`top_k=10`. The baseline's source fingerprint differs from the SEM revision.
Historical SEM runs did not save separate environment/model snapshots; do not
assume the baseline snapshot fully describes them. Their configuration and
predictions are retained, but provenance is incomplete compared with new runs.

The findings show a development precision/recall tradeoff, not a proven overall
improvement or a causal explanation of individual wrong answers. No token savings,
controlled latency improvement, statistical significance, or held-out performance
has been established. New runs should record those measurements as appropriate.

`smoke/` contains the original one-question walkthrough. New workflow smoke runs
are isolated under `output/<name>/smoke/` and ignored by Git. Graphs and checkpoint
sidecars remain local; a fresh clone must rebuild or separately restore graph
caches before running inference. Completed saved predictions can be compared
without graphs or checkpoint sidecars.
