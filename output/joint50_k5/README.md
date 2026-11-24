# Joint selection with a five-chunk cap

Completed all 50 development questions without recorded failures. Compared with
`joint50`, only `joint_keep_k` changes from 10 to 5; rank weight remains 0.7.
Pipeline source, dataset, loaded-graph fingerprints, model settings, and `top_k=10`
match the ten-chunk joint run. The cap acts before original graph/BGE selection.
Actual final context sizes can be smaller than five sentences.

## Results

Scores are percentages, recomputed after complete coverage/checkpoint checks.

| Metric | CROSS alone | Joint cap 10 | Joint cap 5 |
|---|---:|---:|---:|
| Answer exact match | 38.0% | 34.0% | 36.0% |
| Answer F1 | 56.7% | 53.5% | 55.7% |
| Supporting-fact precision | 43.0% | 21.9% | 37.0% |
| Supporting-fact recall | 68.3% | 83.7% | 73.7% |
| Supporting-fact F1 | 47.8% | 34.2% | 48.5% |
| Joint F1 | 27.7% | 20.7% | 29.7% |
| Mean selected sentences | 4.90 | 9.20 | 4.74 |

The smaller cap improves answer/evidence F1 over cap 10 while reducing evidence
volume. CROSS still has higher answer F1 and evidence precision in this sample;
joint cap 5 has higher evidence recall, evidence F1, and joint F1. Answer exact
match differs by one question between CROSS and cap 5. These are alternative
precision/recall tradeoffs, not a statistically established winner.

The new run and ten-chunk joint run use identical pipeline source, but generation
variation remains uncontrolled. CROSS's pipeline-source fingerprint differs.
Average context sizes are now similar; the methods still use different per-query
selection rules and do not have identical evidence budgets on every question.
This is development data used for tuning, not held-out evaluation.

Wall time was 313.6 seconds (about 5 min 14 sec), including startup. This single
uncontrolled timing is not a reliable filter-latency or speedup measurement.
Sentence counts are not token counts.

## Evidence review

`evidence_review.json` records all 50 questions' predictions, selected evidence,
gold-reference labels, and differences across CROSS, joint cap 10, and cap 5.
Gold labels are used retrospectively, not by the inference selector.

- Cap 5 adds 14 gold fact occurrences missing from CROSS across 13 questions.
- It loses 6 gold fact occurrences selected by CROSS across 6 questions.
- Cap 10 had 23 gold occurrences missing from CROSS across 18 questions; cap 5
  retains 14 of those 23 occurrences.
- Final ordered evidence is identical between CROSS and cap 5 on only 3 questions.

These are question-specific occurrence counts, not macro-averaged HotpotQA scores.
Non-gold sentences are not automatically irrelevant, and a larger gold overlap
is not proof that the necessary answer fact survives.

### Example: useful evidence survives

For “Which has more species, Lysiloma or Hydrocotyle umbellata?”, cap 5 retains the
reference sentence identifying Lysiloma as a genus. CROSS omits that sentence.
The reference answer is `Lysiloma`; the saved CROSS answer is `Fewer.`, while both
joint runs answer `Lysiloma.`. This is an observed example, not causal proof.

### Example: a compact selection still loses a needed answer fact

For the Joggers question asking what the lead singer's chemist father received
the highest rating in, the reference answer is `Hirsch index rating`. Joint cap
10 retains `George M. Whitesides`, sentence 2, explicitly stating that rating,
and answers correctly. CROSS and cap 5 omit that sentence and answer `Nobel Prize.`.
Cap 5 retains more annotated facts than CROSS here (3 versus 2), but still misses
the direct answer sentence. This motivates evaluating bridge/answer-fact retention,
not just shrinking context or maximizing aggregate gold overlap.

### Observed graph-format problems: separate future audit

The sampled cached graphs for this example contain extraction-response headers
inside entity strings and a triple with an empty endpoint. Exact examples are
saved in the review JSON. Existing graph validation checks JSON/triple shape,
not semantic or formatting quality. These observations are not a full graph audit
or proof that graph defects caused the scores above. No graphs were changed.
Audit/cleaning and graph recovery should be separate experiments with their own
input fingerprints and matched baselines.

## Inspect or recompute

```bash
./experiment status joint50_k5
./experiment evaluate joint50_k5
./experiment compare baseline50 cross50 joint50 joint50_k5
```

Do not overwrite this completed run. A different weight, cap, code revision, or
graph-cleaning policy requires a fresh experiment. CROSS 0.5 and joint cap 5 / 0.7
are reasonable development shortlist candidates; freeze settings and use separate,
disjoint questions for the next validation rather than tuning indefinitely on
these 50. Any held-out extraction/evaluation is a separate run, not performed here.
