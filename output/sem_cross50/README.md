# Combined SEM → CROSS experiment

Completed all 50 development questions with no recorded failures, using the same
questions and extracted graphs as earlier runs. Smoke outputs live separately
under `smoke/` and do not establish performance on the development set.
The live one-question check completed without recorded errors: it selected both
gold supporting sentences but generated an incorrect answer. All 36 offline tests
passed. This verifies execution and stage wiring, not an improvement in accuracy.

## Pipeline

```text
Seed sentences → KG expansion → SEM (cosine >= 0.3)
               → CROSS on survivors (raw relevance score >= 0.5)
               → original graph organization/BGE reranking → Llama 3
```

SEM uses `sentence-transformers/all-MiniLM-L6-v2`; CROSS uses
`cross-encoder/ms-marco-MiniLM-L6-v2`. Both load once on CPU. They filter existing
sentence objects without replacing their original retrieval scores. If either
filter removes everything, that stage retains its single highest-scoring input.
CROSS cannot recover evidence already removed by SEM.

`experiment.json` records both model IDs and thresholds. These initial cutoffs
reuse earlier development settings, not a guaranteed optimum. Do not modify this
configuration once prediction/checkpoint files exist.

## Results

Scores are percentages, recomputed after complete prediction and checkpoint checks.

| Metric | Baseline | CROSS alone | SEM → CROSS |
|---|---:|---:|---:|
| Answer exact match | 32.0% | 38.0% | 36.0% |
| Answer F1 | 52.6% | 56.7% | 53.6% |
| Supporting-fact precision | 21.3% | 43.0% | 42.5% |
| Supporting-fact recall | 83.0% | 68.3% | 66.6% |
| Supporting-fact F1 | 33.4% | 47.8% | 47.0% |
| Joint F1 | 19.4% | 27.7% | 25.3% |
| Mean selected sentences | 9.58 | 4.90 | 4.82 |

CROSS alone scored higher in this single development comparison. Adding SEM
changed the final evidence in only 3 of 50 questions; 47 had identical ordered
supporting-fact locations. The difference in answer F1 therefore cannot be
attributed solely to filtering. Repeated runs with controlled generation and
held-out evaluation are needed before claiming a reliable gain or regression.

Prediction wall time was 365.8 seconds including startup, versus 346.9 seconds
for the recorded CROSS run. These uncontrolled single wall times do not establish
filtering overhead or a reliable latency comparison. Intermediate candidate
counts and input token usage were not recorded; fewer sentences are not a
measured token or CROSS-computation saving.

## Inspect or recompute

```bash
./experiment status sem_cross50
./experiment evaluate sem_cross50
./experiment compare baseline50 sem50 sem50_t03 cross50 sem_cross50
```

The full run is complete. Use a new experiment name for changed code/settings.
Under identical source/configuration, `./experiment run sem_cross50 --resume`
skips completed questions and retries any recorded failures. Predictions,
metrics, actual configuration, provenance, and timing are saved in this directory;
checkpoints and source snapshots remain local and ignored. Earlier results have
not been overwritten.
