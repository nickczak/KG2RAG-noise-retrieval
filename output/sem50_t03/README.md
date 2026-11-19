# SEM development experiment: threshold 0.3

This historical run completed all 50 questions without recorded errors.
Answer EM/F1: 36.0%/52.1%; evidence precision/recall/F1: 22.3%/77.9%/33.7%.
Mean selected sentences: 8.76. SEM 0.3 recovers much of SEM 0.5's lost recall;
answer F1 is close to the baseline, not a clear overall improvement.

For new code/method tests, use [WORKFLOW.md](../../WORKFLOW.md). Compare these
historical results with `./experiment compare baseline50 sem50 sem50_t03`.
The scripts below are retained to document how this run was launched. Do not
launch them into the completed output under edited source; create a new run name.
Uses the same 50 questions, saved graphs, generator, and final `top_k=10` setting
as the baseline and SEM 0.5 runs. Only the intended SEM threshold changes to 0.3.
See `run_settings.json` for launch arguments, and preserve earlier results.

## 1. Record the environment

From the repository root, with Ollama running:

```bash
source .venv/bin/activate
python -m pip freeze > output/sem50_t03/environment.txt
git rev-parse HEAD > output/sem50_t03/commit.txt
git diff > output/sem50_t03/working-tree.patch
git status --short > output/sem50_t03/git-status.txt
ollama --version > output/sem50_t03/ollama-version.txt
ollama list > output/sem50_t03/ollama-models.txt
```

## 2. Run the 50 questions

```bash
time bash output/sem50_t03/run.sh
```

The script uses the project's `.venv` Python and reads `run_settings.json`.
It runs SEM with threshold 0.3 and writes only into this experiment directory.
No graph extraction is needed. If interrupted or questions fail, retry with:

```bash
time bash output/sem50_t03/run.sh --resume
```

Do not change the code, inputs, model settings, or threshold during a run.
Keep the checkpoint sidecar locally; changing settings requires a new output path.
Do not launch two processes into the same output path.

## 3. Verify, evaluate, and compare

After prediction completes:

```bash
.venv/bin/python output/sem50_t03/evaluate.py
```

This verifies complete question-ID coverage, no recorded errors, matching exported
predictions/checkpoint, and the intended SEM method/threshold. It saves
`metrics.json` and the actual checkpoint `config.json`, then prints a comparison
with the original baseline and SEM 0.5. It does not modify those earlier runs.

The script can be rerun to recompute the current metrics. Inspect answer F1 and
evidence precision/recall together: a smaller context is not automatically better.
Sentence counts are not token counts. No causal attribution or statistical
significance is established by this small development comparison. Use separate
held-out questions after freezing the final settings.

## Files

- `run_settings.json`: intended launch parameters (not evidence of a completed run).
- `run.sh`: launch/resume command.
- `evaluate.py`: coverage checks, evaluation, and comparison.
- `predictions.json`, `metrics.json`, `config.json`: generated once the run and
  evaluation finish; kept trackable alongside environment and model records.
- `predictions.json.checkpoint.json`, working-tree snapshots, and logs: ignored,
  but retained locally.
