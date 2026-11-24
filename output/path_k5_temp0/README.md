# path_k5_temp0

Completed path seed-count sweep point: 50/50 questions, temperature 0.0,
`top_k=5`, final sentence cap 7. Metrics: answer EM 48.00%, answer F1 66.33%,
supporting-fact precision 33.92%, recall 83.93%, F1 47.53%; mean selected
sentences 6.14. It beats the controlled baseline on answer and evidence recall,
but trails the three-seed path on supporting-fact precision/F1 and context size.

1. `./experiment graphs path_k5_temp0`
2. `./experiment run path_k5_temp0 --smoke`
3. `./experiment evaluate path_k5_temp0 --smoke`
4. `./experiment run path_k5_temp0`
5. `./experiment evaluate path_k5_temp0`

Append `--resume` to a run command after interruption. Read WORKFLOW.md before changing settings or comparing results.
