# `path_k5_cap4_temp0`

Completed matched 50-question evaluation of path-based retrieval using five
initial seeds and a final evidence cap of four sentences.

| Metric | Result |
|---|---:|
| Answer EM | 54.00% |
| Answer F1 | 66.27% |
| Supporting-fact precision | 48.00% |
| Supporting-fact recall | 79.60% |
| Supporting-fact F1 | 58.83% |
| Mean selected evidence sentences | 4.00 |

This run exceeds both `baseline_k7_temp0` and `path_k3_temp0` on answer EM/F1,
supporting-fact precision/recall/F1, and mean selected evidence size. The three
runs share the same 50-question and KG hashes, Llama 3 and embedding model,
reranker on CPU, and `temperature=0.0`. Seed count and final evidence cap vary
intentionally as retrieval parameters.

This is promising development-sample evidence, not an independent final-test
result. The threshold/cap was selected using this same 50-question sample, so
confirm it on a fresh held-out sample before making a general performance claim.

The run's `config.json`, `predictions.json`, `metrics.json`, and logs preserve
the settings and outputs.
