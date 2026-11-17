# Baseline walkthrough and next experiment

Run commands from the repository root unless a step says otherwise.
This directory contains the completed one-question workflow check. Generated
datasets, graphs, predictions, and environment snapshots are ignored by Git;
this README is tracked.

## Current checkpoint

The smoke test completed with ten graph files. Its question asks which government
position Shirley Temple held. The gold answer is `Chief of Protocol`; the model
returned `Diplomat.`. All three gold supporting facts were selected among ten
sentences (supporting-fact precision 0.30, recall 1.00, F1 about 0.462).

This verifies execution, not baseline accuracy. This question also appears in the
runner's example prompt, so exclude it and the other prompt examples from future
experimental subsets. Do not tune the pipeline to fix this single answer.

## 1. Activate and verify the working environment

```bash
source .venv/bin/activate
python -m pip check
python -c 'import numpy, pyarrow; from FlagEmbedding import FlagReranker; print(numpy.__version__, pyarrow.__version__)'
ollama list
```

The repaired environment uses NumPy 1.26.4 and PyArrow 25.0.0. PyArrow 26.0.1
failed to import with NumPy 1.26.4. If that exact conflict recurs:

```bash
python -m pip install 'numpy==1.26.4' 'pyarrow==25.0.0'
```

Required Ollama models are `llama3:8b` and `mxbai-embed-large`.
Start `ollama serve` in another terminal if the server is not running.
While Llama 3 is loaded, use `ollama ps` to inspect CPU/GPU placement and
`nvidia-smi` to inspect GPU activity. Ollama manages GPU use independently of
PyTorch. On this 8 GB GPU, keep the BGE reranker on CPU for the initial baseline.

## 2. Save reproducibility information

```bash
python -m pip freeze > output/smoke/environment.txt
git rev-parse HEAD > output/smoke/commit.txt
git diff > output/smoke/working-tree.patch
git status --short > output/smoke/git-status.txt
ollama --version > output/smoke/ollama-version.txt
ollama list > output/smoke/ollama-models.txt
```

Record the run command, model identifiers, `top_k`, prompt, device placement,
and any extraction or generation settings with each subsequent experiment.
The commit alone does not capture uncommitted changes or untracked source files.

## 3. Recreate the temporary CPU reranker runner

`/tmp` files may disappear after a reboot. The following recreates the runner
used by the smoke test without editing the original implementation:

```bash
python - <<'PY'
from pathlib import Path

source = Path('code/kg_rag_distractor.py').read_text()
original = 'FlagReranker(model_name_or_path=args.reranker,device=3)'
replacement = ('FlagReranker(model_name_or_path=args.reranker,'
               'devices=["cpu"],use_fp16=False)')
assert original in source, 'Reranker initialization changed; inspect before proceeding.'
Path('/tmp/kg_rag_distractor_cpu.py').write_text(source.replace(original, replacement))
PY
```

## 4. Rerun and evaluate the smoke test if needed

This reuses `hotpot_one.json` and the extracted graphs in `kgs/`. It overwrites
the smoke prediction file, so copy that file first if preserving an earlier run.

```bash
cd code
PYTHONPATH="$PWD" python -u /tmp/kg_rag_distractor_cpu.py \
  --dataset hotpotqa \
  --data_path ../output/smoke/hotpot_one.json \
  --kg_dir ../output/smoke/kgs \
  --result_path ../output/smoke/predictions.json \
  --model_name llama3:8b \
  --embed_model_name mxbai-embed-large \
  --reranker BAAI/bge-reranker-large \
  --top_k 10
cd ..
python code/util/hotpot_evaluate_v1.py \
  output/smoke/predictions.json output/smoke/hotpot_one.json
```

Check the runner's log for a nonzero loaded-graph count. Graph files on disk
alone do not establish that the runner loaded or used them.

## 5. Next implementation milestone: a repeatable 50-question baseline

The following work is planned; the existing runner does not yet implement it.

1. Load the reranker once per run rather than once per question.
2. Save predictions atomically after each question and support resume by question
   ID. Record failures separately so they can be retried rather than silently
   treating empty answers as successful completed work.
3. Select 50 questions with a fixed random seed, excluding all four examples in
   the answer prompt. Save the selected IDs and sampling settings. Treat these as
   development questions; reserve separate questions for final evaluation.
4. Extract graphs for those questions into a separate output directory. Reuse
   completed paragraph graphs and record extraction settings and elapsed time.
5. Run the original retrieval pipeline with the same CPU reranker configuration.
   Record answer EM/F1, supporting-fact precision/recall/F1, extraction time,
   prediction time, and any failures. Keep extraction timing separate from
   inference timing and distinguish cold model loads from warm runs.
6. Inspect several successes and failures before expanding the evaluation.

Once this baseline works, compare SEM, CROSS, and automated GRAPH recovery after
expansion using the same questions, cached graphs, generator, and evidence budget.
The graph-enriched bi-encoder is a subsequent comparison against a text-only
bi-encoder. Keep thresholds and other tuning decisions confined to development
data, and freeze settings before evaluating held-out questions.
