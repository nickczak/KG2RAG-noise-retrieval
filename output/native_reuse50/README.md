# native_reuse50

Prepared experiment; full 50-question results pending. The one-question smoke
run and evaluation completed successfully as workflow checks.

1. `./experiment graphs native_reuse50`
2. `./experiment run native_reuse50 --smoke`
3. `./experiment evaluate native_reuse50 --smoke`
4. `./experiment run native_reuse50`
5. `./experiment evaluate native_reuse50`

Append `--resume` to a run command after interruption. Read WORKFLOW.md before changing settings or comparing results.

The smoke answer was incorrect; supporting-fact recall was 1.0. This does not
establish answer quality. See WORKFLOW.md for native score and cache semantics.

Smoke audit: 10 seed scores reused, zero expanded sentences, 9/10 exact BGE
inputs served from cache and one newly scored input. Five sentences selected.
No auxiliary MiniLM encoders were loaded.
