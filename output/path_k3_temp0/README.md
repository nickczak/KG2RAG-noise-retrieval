# path_k3_temp0

Completed path baseline: 50/50 questions, temperature 0.0, `top_k=3`, final
sentence cap 7. Metrics: answer EM 46.00%, answer F1 62.60%, supporting-fact
precision 41.35%, recall 74.43%, F1 51.62%; mean selected sentences 4.66.

1. `./experiment graphs path_k3_temp0`
2. `./experiment run path_k3_temp0 --smoke`
3. `./experiment evaluate path_k3_temp0 --smoke`
4. `./experiment run path_k3_temp0`
5. `./experiment evaluate path_k3_temp0`

Append `--resume` to a run command after interruption. Read WORKFLOW.md before changing settings or comparing results.
