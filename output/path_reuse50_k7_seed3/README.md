# Path-aware reuse, held-out 50 questions

Completed all 50 questions with no recorded failures. Uses the same held-out
questions and KG files as `baseline50_seed3`, with `top_k=3`, a seven-sentence
maximum, native semantic/BGE endpoint scoring, and shortest KG-linked routes to
retrieval seeds.

| Metric | Result |
|---|---:|
| Answer EM / F1 | 46.0% / 63.8% |
| Evidence precision / recall / F1 | 41.4% / 74.4% / 51.6% |
| Mean selected sentences | 4.66 |

Against the matched top-3 baseline, answer EM tied and answer F1 rose 3.3 points;
evidence recall rose 13.7 points, while evidence precision fell 18.3 points and
evidence F1 fell 6.0 points. The method supplied about two more sentences per
question. This suggests path retention may recover useful bridge evidence, while
also adding sentences that are not annotated as support.

The audit reports 50 questions, 38 with at least one nonzero-hop retained path,
149 zero-hop paths, 79 one-hop paths, and 5 two-hop paths. These are paths in a
candidate graph induced by shared entities in extracted triples, so their quality
can be affected by KG extraction errors. The results are from a disjoint sample
of HotpotQA development, not the official test split; they do not establish a
reliable gain. See `score_reuse.json` for per-question paths and scores.
