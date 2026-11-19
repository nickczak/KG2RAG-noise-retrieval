# SEM experiment: same 50 questions, reused graphs

This historical SEM 0.5 run completed all 50 questions without recorded errors.
Answer EM/F1: 28.0%/43.2%; evidence precision/recall/F1: 36.7%/58.2%/38.1%.
Mean selected sentences: 5.38. Compared with the original baseline, precision rose
but recall and answer F1 fell. This is a development result, not a held-out claim.

For new runs use [WORKFLOW.md](../../WORKFLOW.md); keep these records unchanged.
Compare current saved predictions with `./experiment compare baseline50 sem50 sem50_t03`.
The commands below document the original manual workflow; use a new output path
when code/settings change rather than attempting to resume this completed run.

## 1. Activate the environment and record settings

From the repository root:

```bash
source .venv/bin/activate
mkdir -p output/sem50
python -m pip freeze > output/sem50/environment.txt
git rev-parse HEAD > output/sem50/commit.txt
git diff > output/sem50/working-tree.patch
git status --short > output/sem50/git-status.txt
ollama --version > output/sem50/ollama-version.txt
ollama list > output/sem50/ollama-models.txt
```

Keep Ollama running. The encoder model may download on first use. SEM and BGE run
on CPU in these commands; Ollama manages Llama 3 GPU placement separately.

## 2. Run a one-question SEM smoke test

```bash
python -u code/kg_rag_distractor.py \
  --dataset hotpotqa \
  --data_path output/smoke/hotpot_one.json \
  --kg_dir output/smoke/kgs \
  --result_path output/sem50/smoke_predictions.json \
  --model_name llama3:8b \
  --embed_model_name mxbai-embed-large \
  --reranker BAAI/bge-reranker-large \
  --reranker_device cpu \
  --method sem \
  --sem_model sentence-transformers/all-MiniLM-L6-v2 \
  --sem_threshold 0.5 \
  --top_k 10
```

Check that the answer/evidence export exists and the checkpoint `errors` is empty.
This question appears in the prompt examples, so it checks execution only. If the
run fails after SEM leaves just one candidate, inspect the downstream reranker
score handling described in `CLASS_DOCS.md`. Do not report partial results.

## 3. Run the fixed development set

Only proceed once the live smoke test completes. Reuse existing graphs; no
extraction is needed. Write into a new directory, preserving the original baseline.

```bash
time python -u code/kg_rag_distractor.py \
  --dataset hotpotqa \
  --data_path output/baseline50/hotpot_50.json \
  --kg_dir output/baseline50/kgs \
  --result_path output/sem50/predictions.json \
  --model_name llama3:8b \
  --embed_model_name mxbai-embed-large \
  --reranker BAAI/bge-reranker-large \
  --reranker_device cpu \
  --method sem \
  --sem_model sentence-transformers/all-MiniLM-L6-v2 \
  --sem_threshold 0.5 \
  --top_k 10
```

To retry failures or resume after interruption, append `--resume` to the same
command. Do not change the threshold, model, source, data, or graphs mid-run.
Each different threshold requires its own output directory and records.

## 4. Verify coverage, evaluate, and save configuration

The following checks IDs and errors before evaluating and exporting metrics.
It does not modify the saved original baseline metrics.

```bash
python - <<'PY'
import ast
import contextlib
import importlib.util
import io
import json
from pathlib import Path

out = Path('output/sem50')
data = json.loads(Path('output/baseline50/hotpot_50.json').read_text())
pred = json.loads((out / 'predictions.json').read_text())
state = json.loads((out / 'predictions.json.checkpoint.json').read_text())
ids = {sample['_id'] for sample in data}
assert set(pred['answer']) == set(pred['sp']) == ids, 'Incomplete prediction IDs'
assert not state['errors'], 'Resolve failed questions first'
assert pred == state['prediction'], 'Prediction export differs from checkpoint'
spec = importlib.util.spec_from_file_location('hotpot_eval', 'code/util/hotpot_evaluate_v1.py')
evaluator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluator)
stream = io.StringIO()
with contextlib.redirect_stdout(stream):
    evaluator.eval(str(out / 'predictions.json'), 'output/baseline50/hotpot_50.json')
metrics = ast.literal_eval(stream.getvalue().strip())
(out / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
(out / 'config.json').write_text(json.dumps(state['config'], indent=2) + '\n')
print(json.dumps(metrics, indent=2))
PY
```

Compare answer F1 and supporting-fact precision/recall/F1 against
`output/baseline50/metrics.json`. Keep `top_k=10` comparable, and measure actual
evidence/token counts: SEM can produce fewer than ten sentences. Runtime, token
counts, and before/after SEM gold-support retention are not yet automatically
recorded. Final evidence metrics alone cannot isolate which stage removed a fact.

The threshold is a development parameter. A single successful run does not show
that SEM is better, and development improvements need held-out evaluation.
