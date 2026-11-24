# Historical SEM development run: threshold 0.3

All 50 questions completed without recorded errors. These saved results are
preserved; create a new experiment name when testing current source/settings.

| Metric | Result |
|---|---:|
| Answer EM | 36.0% |
| Answer F1 | 52.1% |
| Evidence precision | 22.3% |
| Evidence recall | 77.9% |
| Evidence F1 | 33.7% |
| Mean selected sentences | 8.76 |

The run uses the fixed baseline question set and cached graphs, Llama 3,
mxbai initial embeddings, additional MiniLM SEM scoring, and original BGE graph
organization with `top_k=10`. SEM 0.5 improves precision but loses recall and
answer F1; SEM 0.3 recovers much of the recall loss. This is development evidence,
not a held-out result. Historical environment/model provenance is incomplete.

Use the shared workflow for inspection:

```bash
./experiment evaluate sem50_t03
./experiment compare baseline50 sem50 sem50_t03
```

For a new run, follow [WORKFLOW.md](../../WORKFLOW.md); do not resume this
historical checkpoint with edited source. `config.json`, `predictions.json`, and
`metrics.json` describe the completed run. Graphs and checkpoint sidecars remain
local and ignored. Final evidence metrics do not isolate the stage that removed
a fact, and sentence counts are not token counts.

`run_settings.json` preserves the historical launch arguments. The obsolete
per-folder `run.sh` and `evaluate.py` have been removed; `./experiment` replaces
both and validates coverage/settings before evaluation.
