# Joint SEM/CROSS selection experiment

Completed all 50 questions without recorded failures. The one-question live smoke
run also completed. Results remain separate from `sem_cross50` and `cross50`.

## Keep rule

1. Both encoders score every expanded candidate sentence.
2. Convert SEM scores and CROSS scores separately to rank percentiles. Equal
   scores get average ranks; raw logits and cosine values are never averaged.
3. Compute `0.7 * CROSS_rank + 0.3 * SEM_rank`.
4. Reserve the strongest CROSS candidate, then fill remaining slots by fused rank,
   keeping at most ten candidates.
5. Pass the original selected objects in their original order to the existing
   graph/BGE stage, which can remove further evidence.

There are no score thresholds. A weak SEM score cannot prevent CROSS from seeing
a sentence. The CROSS anchor remains reserved even if the rank weight is zero.
Original retrieval scores are unchanged. This is a heuristic candidate selector,
not new graph recovery, a trained selector, a truth checker, or an improvement
claim. It may keep irrelevant evidence when the entire candidate pool is weak.

The live smoke example had ten expanded candidates and a ten-candidate cap, so
the selector retained all ten. This verifies execution but not actual reduction
on that example. Offline tests cover nontrivial pruning and selection behavior.
Use a new experiment name for a different cap or weight; do not tune on one case.

## Results

Scores are percentages, recomputed after verifying complete question coverage,
valid evidence locations, configuration, and checkpoint consistency.

| Metric | Original baseline | CROSS alone | Joint selection |
|---|---:|---:|---:|
| Answer exact match | 32.0% | 38.0% | 34.0% |
| Answer F1 | 52.6% | 56.7% | 53.5% |
| Supporting-fact precision | 21.3% | 43.0% | 21.9% |
| Supporting-fact recall | 83.0% | 68.3% | 83.7% |
| Supporting-fact F1 | 33.4% | 47.8% | 34.2% |
| Joint F1 | 19.4% | 27.7% | 20.7% |
| Mean selected sentences | 9.58 | 4.90 | 9.20 |

At this ten-candidate cap and 0.7 CROSS rank weight, the joint selector retains
much more evidence than CROSS thresholding, raising supporting-fact recall but
lowering precision and answer/evidence F1 relative to CROSS alone. It remains
close to the original baseline in evidence volume and scores. This does not
establish a reliable improvement, regression, or optimal weight/budget. Final
ordered evidence matched the original baseline for 16 questions and CROSS alone
for one; generation variation and source revisions remain confounding factors.

Prediction wall time was 668.3 seconds including startup (about 11 min 8 sec).
This is a single uncontrolled wall time, not measured filter-only overhead or a
reliable latency comparison. Sentence counts are not token counts. Larger-context
processing may contribute to timing differences, but was not isolated here.

## Inspect or recompute

```bash
./experiment status joint50
./experiment evaluate joint50
./experiment compare baseline50 cross50 sem_cross50 joint50
```

Use a new experiment name for a different cap, weight, or implementation. Under
identical source and settings, `./experiment run joint50 --resume` skips successful
questions and retries failures. No graph extraction was performed for this run.

The pure `select_joint_chunks` function in `code/util/kg_post_processor.py` is the
place to change the selection objective. Checkpoint metadata includes both model
IDs, `joint_keep_k=10`, and `cross_weight=0.7`; SEM/CROSS cutoff fields are inactive
(`None`). Per-chunk score traces are not saved; candidate/kept counts are printed.
Maintain the same questions, graphs, generator, and final evidence budget across
comparisons, report actual context sizes, and reserve held-out questions for final
validation. Future budget/weight trials should use separate output directories.
