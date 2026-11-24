# Experiment records

Use the repository's [shared workflow](../WORKFLOW.md) for new experiments.
The initial seven development runs below are complete and should be preserved.

```bash
./experiment compare baseline50 sem50 sem50_t03 cross50 sem_cross50 joint50 joint50_k5
```

| Directory | Method | Threshold | Questions | Answer F1 | Evidence precision | Evidence recall | Mean sentences |
|---|---|---:|---:|---:|---:|---:|---:|
| `baseline50` | Original KG²RAG pipeline | — | 50 | 52.55% | 21.30% | 83.00% | 9.58 |
| `sem50` | SEM before existing graph organization | 0.5 | 50 | 43.23% | 36.67% | 58.20% | 5.38 |
| `sem50_t03` | SEM before existing graph organization | 0.3 | 50 | 52.07% | 22.29% | 77.93% | 8.76 |
| `cross50` | CROSS before existing graph organization | 0.5 raw score | 50 | 56.75% | 42.96% | 68.30% | 4.90 |
| `sem_cross50` | SEM → CROSS before existing graph organization | SEM 0.3; CROSS 0.5 | 50 | 53.61% | 42.46% | 66.63% | 4.82 |
| `joint50` | Joint SEM/CROSS rank fusion | cap 10, CROSS weight 0.7 | 50 | 53.50% | 21.85% | 83.67% | 9.20 |
| `joint50_k5` | Joint SEM/CROSS rank fusion | cap 5, CROSS weight 0.7 | 50 | 55.69% | 37.03% | 73.70% | 4.74 |

All seven have 50 answer/evidence predictions and no local checkpoint errors.
They share the question-content and loaded-graph fingerprints, model names, and
`top_k=10`. The baseline's source fingerprint differs from the SEM revision.
Historical SEM runs did not save separate environment/model snapshots; do not
assume the baseline snapshot fully describes them. Their configuration and
predictions are retained, but provenance is incomplete compared with new runs.

The findings show a development precision/recall tradeoff, not a proven overall
improvement or a causal explanation of individual wrong answers. The initial runs
did not record token use or controlled latency. Current prediction runs record
Ollama prompt/completion tokens and per-question query/pipeline latency in
`usage.json`; see each run's `summary.json` for the usage report.

`smoke/` contains the original one-question walkthrough. New workflow smoke runs
are isolated under `output/<name>/smoke/` and ignored by Git. Graphs and checkpoint
sidecars remain local; a fresh clone must rebuild or separately restore graph
caches before running inference. Completed saved predictions can be compared
without graphs or checkpoint sidecars.

The Matryoshka GNN prototype was evaluated and is no longer in the runnable
pipeline. Its generated training/checkpoint artifacts are local and ignored;
the matched 50-question metrics remain documented in `gnn_baseline_k7/README.md`
and `gnn_mrl_heldout/README.md` as historical comparison records.

CROSS 0.5 raises answer/evidence F1 in this development sample with about 49%
fewer selected sentences, but loses gold-support recall. See `cross50/README.md`
for actual settings, runtime, and limitations. This run has the workflow
provenance records that were missing for the historical SEM runs.

The combined run is complete. It scored below CROSS alone on answer/evidence F1
in this sample, with final evidence identical in 47/50 questions. Do not attribute
answer-score differences solely to SEM; generation variation and held-out
validation remain unresolved. See `sem_cross50/README.md` for full results.

`joint50/` completed all 50 questions. At cap 10 and CROSS weight 0.7 it retains
more evidence and gold-support recall than CROSS alone, but has lower answer and
evidence F1 in this sample. See `joint50/README.md` for the keep rule, actual
settings, runtime, and limitations.

`joint50_k5/` is complete and contains a per-question `evidence_review.json`.
Cap five improves on cap ten; relative to CROSS it trades some answer F1 and
precision for more gold-support recall at a similar average context size. The
review includes the Joggers missed-answer example and formatting anomalies observed
in sampled graphs. Existing caches are preserved; any cleanup is separate work.


- `native_baseline50/` and `native_reuse50/` completed 50-question runs. The
  exact-score reuse variant reduced mean selected sentences from 9.66 to 5.00
  and raised supporting-fact F1 from 33.05% to 42.72%, while answer F1 fell
  from 51.12% to 46.29%. See their run records for all settings and metrics.
- `score_cache/`: ignored local BGE SQLite cache, shared by opted-in experiments.
  Trackable `score_reuse.json` files record counters and native selection traces.

## Controlled path-selection sweep

The 500-question tuning sample and matched five-seed/four-sentence
baseline/path run instructions are in [`validation500/README.md`](validation500/README.md).
This sample excludes question IDs from earlier baseline/heldout/smoke runs. It is
development data; reserve a separate fresh 500-question sample for final held-out
evaluation after tuning.

These historical runs use the same 50 development questions, KG files, Llama 3 model, reranker,
and recorded Ollama `temperature=0.0`. The seed-count runs intentionally vary
`top_k`; compare their tradeoffs without treating them as one fixed-setting
ablation. Supporting-fact metrics measure retrieval quality; `sentences` is the
mean selected context size, not a token count.

| Run | top_k | Final cap | Answer EM | Answer F1 | SP precision | SP recall | SP F1 | Sentences |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `baseline_k7_temp0` | 7 | 7 | 42.00% | 60.09% | 28.01% | 78.70% | 40.69% | 6.80 |
| `path_k3_temp0` | 3 | 7 | 46.00% | 62.60% | 41.35% | 74.43% | 51.62% | 4.66 |
| `path_k4_temp0` | 4 | 7 | 44.00% | 61.87% | 38.10% | 82.93% | 51.09% | 5.52 |
| `path_k5_temp0` | 5 | 7 | 48.00% | 66.33% | 33.92% | 83.93% | 47.53% | 6.14 |
| `path_k6_temp0` | 6 | 7 | 50.00% | 65.33% | 31.76% | 86.27% | 45.73% | 6.66 |
| `path_k7_temp0` | 7 | 7 | 46.00% | 64.26% | 29.43% | 85.43% | 43.13% | 7.00 |
| `path_k5_cap4_temp0` | 5 | 4 | 54.00% | 66.27% | 48.00% | 79.60% | 58.83% | 4.00 |

`path_k5_cap4_temp0` beats both the seven-seed baseline and the three-seed path
run on answer EM/F1, supporting-fact precision/recall/F1, and mean evidence
sentence count. It uses five initial seeds and caps the final evidence at four
sentences. The three runs have matching question and KG hashes, model names,
reranker/device settings, and temperature; `top_k` and final cap intentionally
differ because those are the retrieval settings being optimized. This is a
promising result on the same 50-question development sample, not independent
final-test confirmation. See `path_k5_cap4_temp0/README.md` for run provenance
and full metrics.

Additional controlled path variants are preserved in their named folders:

| Run | Main result | Conclusion |
|---|---|---|
| `path_confident_fill_k3_temp0` | SP F1 51.89%, 4.70 sentences; answer F1 62.37% | Small evidence gain; answer F1 slightly below `path_k3_temp0` |
| `path_fill70_k3_temp0` | Answer F1 64.16%, SP F1 51.17%, 4.80 sentences | Better answer F1; slightly lower SP F1 than the three-seed path |
| `path_ppr_k3_temp0_v2` | Answer F1 62.60%, SP F1 52.02%, 4.66 sentences | Best evidence F1 by 0.40 points; answer ties path; changed evidence on 1/50 questions |
| `path_zscore_k3_temp0`, `path_smooth_k3_temp0` | Same metrics/evidence as the PPR run | No broader selection change |
| `path_crossscore_k3_temp0` | Answer F1 62.27%, SP F1 51.13% | Joint path scoring underperformed the base path |
| `path_marginal_k3_temp0` | Answer F1 61.35%, SP F1 46.44%, 6.10 sentences | Rejected; added evidence without preserving precision |

`path_ppr_k3_temp0/` is retained as a failed run because PageRank did not
converge on 33 questions under the initial tolerance. `path_ppr_k3_temp0_v2/`
uses a relaxed numerical tolerance and completed. The `path_seed3_from5_k7_temp0`
run selected the same evidence and scored identically to `path_k5_temp0`.

All these variants are still development-set experiments. The thresholds and
weights were explored on these same 50 questions; reserve a new question sample
for final confirmation. Historical runs without a recorded temperature are
exploratory answer-score references, not controlled comparisons.
