# path_k6_temp0

Completed controlled path sweep: 50 questions, temperature 0.0, `top_k=6`, cap 7.
Answer EM/F1: 50.00%/65.33%; supporting-fact precision/recall/F1:
31.76%/86.27%/45.73%; mean selected sentences: 6.66.

1. `./experiment graphs path_k6_temp0`
2. `./experiment run path_k6_temp0 --smoke`
3. `./experiment evaluate path_k6_temp0 --smoke`
4. `./experiment run path_k6_temp0`
5. `./experiment evaluate path_k6_temp0`

Append `--resume` to a run command after interruption. Read WORKFLOW.md before changing settings or comparing results.
