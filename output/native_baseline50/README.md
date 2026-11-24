# native_baseline50

Prepared experiment; full 50-question results pending. The one-question smoke
run and evaluation completed successfully as workflow checks.

1. `./experiment graphs native_baseline50`
2. `./experiment run native_baseline50 --smoke`
3. `./experiment evaluate native_baseline50 --smoke`
4. `./experiment run native_baseline50`
5. `./experiment evaluate native_baseline50`

Append `--resume` to a run command after interruption. Read WORKFLOW.md before changing settings or comparing results.

The smoke answer was incorrect; supporting-fact recall was 1.0. This does not
establish answer quality. See WORKFLOW.md for native score and cache semantics.

Smoke audit: 10 BGE inputs computed and cached; ten sentences selected. Original
baseline selection is retained. This fresh control shares the native cache model
fingerprint/settings.
