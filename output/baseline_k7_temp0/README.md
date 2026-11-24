# baseline_k7_temp0

Completed controlled baseline: 50/50 questions, temperature 0.0, `top_k=7`.
Metrics: answer EM 42.00%, answer F1 60.09%, supporting-fact precision 28.01%,
recall 78.70%, F1 40.69%; mean selected sentences 6.80.

1. `./experiment graphs baseline_k7_temp0`
2. `./experiment run baseline_k7_temp0 --smoke`
3. `./experiment evaluate baseline_k7_temp0 --smoke`
4. `./experiment run baseline_k7_temp0`
5. `./experiment evaluate baseline_k7_temp0`

Append `--resume` to a run command after interruption. Read WORKFLOW.md before changing settings or comparing results.
