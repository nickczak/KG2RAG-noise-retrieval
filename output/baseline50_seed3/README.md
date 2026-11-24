# Matched held-out baseline, top 3 seeds

Completed all 50 questions with no recorded failures. Uses the held-out question
set, shared extracted graphs, `top_k=3`, and the original KG2RAG graph/BGE
selection.

| Metric | Result |
|---|---:|
| Answer EM / F1 | 46.0% / 60.5% |
| Evidence precision / recall / F1 | 59.7% / 60.7% / 57.6% |
| Mean selected sentences | 2.62 |

Compare with `./experiment compare baseline50_seed3 path_reuse50_k7_seed3`.
This is a disjoint sample from HotpotQA development, not the official test split.
