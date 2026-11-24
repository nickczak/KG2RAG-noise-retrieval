# CROSS sentence-filter development experiment

Completed on all 50 fixed development questions with no recorded failures.
The one-question live check also completed before the full run.

## Configuration

- Method: `cross`, after graph expansion and before the original graph/BGE stage.
- Filter: `cross-encoder/ms-marco-MiniLM-L6-v2`, CPU.
- Cutoff: raw relevance score >= 0.5; not a probability or cosine threshold.
- If none pass: keep the single highest-scoring sentence.
- Questions/graphs: reuse `output/baseline50/hotpot_50.json` and `output/baseline50/kgs`.
- Generator/embeddings: `llama3:8b` / `mxbai-embed-large`.
- Original BGE reranker: `BAAI/bge-reranker-large`, CPU, full precision.
- `top_k=10`; actual selected sentence counts can be smaller.

See `experiment.json` for the launch plan, `config.json` for actual checkpoint
settings/fingerprints, `provenance.json` for command/source hashes, and local
`source/` snapshots. The corrected default model ID removes the extra hyphen
from the initially entered `MiniLM-L-6-v2` name.

## Results

Scores are percentages. Computed by the repository evaluator after verifying
complete prediction coverage, evidence locations, and checkpoint consistency.

| Metric | Original baseline | CROSS 0.5 |
|---|---:|---:|
| Answer exact match | 32.0% | 38.0% |
| Answer F1 | 52.6% | 56.7% |
| Supporting-fact precision | 21.3% | 43.0% |
| Supporting-fact recall | 83.0% | 68.3% |
| Supporting-fact F1 | 33.4% | 47.8% |
| Joint F1 | 19.4% | 27.7% |
| Mean selected sentences | 9.58 | 4.90 |

The run selected about 49% fewer sentences, with higher answer/evidence F1 in
this sample, but lower supporting-fact recall. This is a development finding,
not statistical significance or a held-out generalization claim. Source revisions
and unrecorded historical environment differences limit causal attribution.
Token counts and filtering-only latency have not been measured. Wall time was
346.9 seconds including model startup; historical runs lack comparable timings.

## Inspect or recompute

```bash
./experiment status cross50
./experiment evaluate cross50
./experiment compare baseline50 sem50 sem50_t03 cross50
```

The full run is complete. Use a new experiment name for changed code/settings.
Resume only under identical pipeline code and configuration:

```bash
./experiment run cross50 --resume
```

Generated checkpoints and local source snapshots remain ignored. Predictions,
metrics, configuration, provenance, environment/model records, and timing are
trackable. No graph extraction was performed for this experiment.
