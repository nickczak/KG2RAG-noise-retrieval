# Historical SEM development run: threshold 0.5

All 50 questions completed without recorded errors. These saved results are
preserved; create a new experiment name when testing current source/settings.

| Metric | Result |
|---|---:|
| Answer EM | 28.0% |
| Answer F1 | 43.2% |
| Evidence precision | 36.7% |
| Evidence recall | 58.2% |
| Evidence F1 | 38.1% |
| Mean selected sentences | 5.38 |

The run uses the fixed baseline question set and cached graphs, Llama 3,
mxbai initial embeddings, additional MiniLM SEM scoring, and original BGE graph
organization with `top_k=10`. SEM 0.5 improves precision but loses recall and
answer F1; SEM 0.3 recovers much of the recall loss. This is development evidence,
not a held-out result. Historical environment/model provenance is incomplete.

Use the shared workflow for inspection:

```bash
./experiment evaluate sem50
./experiment compare baseline50 sem50 sem50_t03
```

For a new run, follow [WORKFLOW.md](../../WORKFLOW.md); do not resume this
historical checkpoint with edited source. `config.json`, `predictions.json`, and
`metrics.json` describe the completed run. Graphs and checkpoint sidecars remain
local and ignored. Final evidence metrics do not isolate the stage that removed
a fact, and sentence counts are not token counts.
