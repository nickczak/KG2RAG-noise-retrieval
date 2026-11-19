# 50-question development baseline

This directory contains the first completed development baseline, not a final
held-out evaluation. Questions were sampled with seed 42 and prompt examples
excluded; see `selection.json` for IDs and sampling settings.

These saved results used the original pipeline, without SEM filtering. The runner
now has an optional SEM stage; that does not change these historical results.
Use the separate `output/sem50/README.md` instructions for the new experiment.
Do not overwrite this directory's configuration or predictions with SEM output.
For new experiments, use [WORKFLOW.md](../../WORKFLOW.md). To recompute the
comparison without running models: `./experiment compare baseline50 sem50 sem50_t03`.

## Completion checks

- 50 questions, 50 answers, and 50 supporting-fact predictions.
- No recorded failures in the local checkpoint.
- Graph files exist for all 500 candidate paragraph titles.
- 496 paragraph graphs contain at least one extracted fact; four are empty.
  File coverage does not establish extraction correctness or downstream graph use.

## Results

Scores below are percentages, recomputed using the repository evaluator.

| Metric | Score |
|---|---:|
| Answer exact match | 32.0% (16/50) |
| Answer F1 | 52.6% |
| Supporting-fact precision | 21.3% |
| Supporting-fact recall | 83.0% |
| Supporting-fact F1 | 33.4% |
| Supporting-fact exact match | 0.0% |
| Joint F1 | 19.4% |

The pipeline retrieves much of the gold support but also selects substantial
additional evidence. It selected ten sentences for 38 questions, nine for nine
questions, five for two questions, and eight for one question. Zero supporting-fact
exact match means no selected evidence set exactly equaled the annotated set;
it does not mean no useful evidence was retrieved.

Some inspected answers get a comparison wrong (for example, choosing the wrong
person for a birth-date comparison). Other answers are broader or use a shorter
name than the reference. These examples need evidence inspection before assigning
a cause. Extra non-gold sentences are not automatically harmful or irrelevant.

These results establish a comparison point. They do not establish that SEM, CROSS,
GRAPH recovery, or a graph-enriched bi-encoder improves the pipeline. Keep this
development set fixed, measure changes against it, and reserve separate questions
for final evaluation. Extraction and prediction runtimes have not been established
by these saved metrics.

## Files to inspect

- `hotpot_50.json`: questions, candidate context, answers, and supporting-fact labels.
- `selection.json`: fixed question IDs, seed, and exclusions.
- `predictions.json`: completed answer and evidence predictions.
- `metrics.json`: full evaluator output, expressed as fractions.
- `config.json`: settings and input/source fingerprints copied from the completed
  checkpoint; these identify the original run, not subsequent source edits.
- `environment.txt`, `commit.txt`, `ollama-version.txt`, and `ollama-models.txt`:
  recorded software/source/model information. A commit alone does not capture
  uncommitted source changes.

Generated `kgs/`, `*.checkpoint.json`, and working-tree snapshots remain local
and ignored. Preserve them locally to reuse extraction and resume predictions.
Do not rerun into the completed prediction path with changed settings; use a
different run directory and retain its own records.

To recompute metrics from the repository root:

```bash
python code/util/hotpot_evaluate_v1.py \
  output/baseline50/predictions.json output/baseline50/hotpot_50.json
```

Verify complete ID coverage before evaluating: this evaluator skips missing
answers rather than enforcing that all dataset questions were processed.
