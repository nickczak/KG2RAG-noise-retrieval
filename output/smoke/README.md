# Baseline walkthrough and next experiment

For new experiments, use the shorter single-command workflow in
[WORKFLOW.md](../../WORKFLOW.md). This page retains the original walkthrough.

Run commands from the repository root unless a step says otherwise.
This directory contains the completed one-question workflow check. Generated
smoke datasets, graphs, predictions, and environment snapshots are ignored by Git;
this README is tracked. Small research records in `output/baseline50` are also
trackable, including the fixed dataset, selection, configuration, completed
predictions, metrics, and environment/model versions. Graphs, checkpoints, and
working-tree snapshots remain ignored.

## Current checkpoint

The smoke test completed with ten graph files. Its question asks which government
position Shirley Temple held. The gold answer is `Chief of Protocol`; the model
returned `Diplomat.`. All three gold supporting facts were selected among ten
sentences (supporting-fact precision 0.30, recall 1.00, F1 about 0.462).

This verifies execution, not baseline accuracy. This question also appears in the
runner's example prompt, so exclude it and the other prompt examples from future
experimental subsets. Do not tune the pipeline to fix this single answer.

## 1. Activate and verify the working environment

**What it does:** Selects the project's Python environment and checks imports,
dependency consistency, and available Ollama models.

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

**What it does:** Records the software and source state of the smoke test so
later experiments can be compared with a known setup.

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

## 3. Check the baseline runner — choose CPU reranking

**What it does:** Uses the repository runner directly with CPU/full-precision
reranking, leaving GPU memory available for Ollama. No temporary script is needed.

```bash
python code/kg_rag_distractor.py --help
```

Confirm that `--reranker_device` and `--resume` appear. CPU/full precision is the
default. Keep the same settings when resuming a run.

## 4. Rerun and evaluate the smoke test if needed

**What it does:** Rechecks the complete one-question prediction and evaluation
workflow. This is optional because the smoke test already completed.

This reuses `hotpot_one.json` and the extracted graphs in `kgs/`. It writes a new
`predictions_checkpointed.json` file, preserving the original smoke result.
If this new run already exists, add `--resume` to its prediction command.

```bash
cd code
python -u kg_rag_distractor.py \
  --dataset hotpotqa \
  --data_path ../output/smoke/hotpot_one.json \
  --kg_dir ../output/smoke/kgs \
  --result_path ../output/smoke/predictions_checkpointed.json \
  --model_name llama3:8b \
  --embed_model_name mxbai-embed-large \
  --reranker BAAI/bge-reranker-large \
  --top_k 10
cd ..
python code/util/hotpot_evaluate_v1.py \
  output/smoke/predictions_checkpointed.json output/smoke/hotpot_one.json
```

Check the runner's log for a nonzero loaded-graph count. Graph files on disk
alone do not establish that the runner loaded or used them.

## 5. Verify the recoverable runner — implemented

**What it does:** Loads the reranker once per run, atomically checkpoints after
questions, and supports resuming completed work and retrying failures.

Run the automated interruption/resume tests without downloading models:

```bash
python -m unittest discover -s tests -v
```

The authoritative sidecar is `<result_path>.checkpoint.json`. It contains answers,
evidence, errors, and configuration fingerprints. Keep it together with the
prediction file. Resume skips successful IDs and retries failed IDs. A run with
failed questions exits with an error; inspect its sidecar before evaluation.

The runner refuses to overwrite existing outputs without `--resume`. Legacy
predictions without a sidecar require a new output path. Resume rejects changed
source, dataset content, loaded graphs, or explicit model/retrieval settings.
Model names alone do not detect changed model weights or server defaults, so
keep the model IDs and environment records from Step 7 too. Do not run two
processes against the same output path simultaneously.

**Completion check:** Automated tests pass. For a live workflow check, run Step 4.
The tests simulate model results; they do not establish answer quality.

## 6. Select 50 development questions — fixed the experimental inputs

**What it does:** Creates one reproducible question set that every method will
use. It excludes questions already included as examples in the answer prompt.

Run from the repository root. This refuses to overwrite an existing selection.

```bash
python - <<'PY'
import json
import random
from pathlib import Path

out = Path('output/baseline50')
out.mkdir(parents=True, exist_ok=True)
target = out / 'hotpot_50.json'
manifest = out / 'selection.json'
if target.exists() or manifest.exists():
    raise SystemExit('Subset already exists; reuse it and its manifest.')
excluded = {
    'Were Scott Derrickson and Ed Wood of the same nationality?',
    'Who was born earlier, Emma Bull or Virginia Woolf?',
    'The arena where the Lewiston Maineiacs played their home games can seat how many people?',
    'What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?',
}
source = Path('data/hotpotqa/hotpot_dev_distractor_v1.json')
data = json.loads(source.read_text())
eligible = [x for x in data if x['question'].strip() not in excluded]
seed = 42
samples = random.Random(seed).sample(eligible, 50)
target.write_text(json.dumps(samples, indent=2))
manifest.write_text(json.dumps({
    'source': str(source), 'seed': seed, 'size': len(samples),
    'excluded_questions': sorted(excluded),
    'question_ids': [x['_id'] for x in samples],
}, indent=2))
print('Saved', len(samples), 'questions to', target)
print('Candidate sentences:', sum(len(s) for x in samples for _, s in x['context']))
PY
```

**Completion check:** The dataset and manifest contain the same 50 question IDs.
Reuse these questions for development; reserve different IDs for final evaluation.

## 7. Record the configuration — made the baseline reproducible

**What it does:** Saves the environment and source state after Step 5 is complete.

```bash
python -m pip freeze > output/baseline50/environment.txt
git rev-parse HEAD > output/baseline50/commit.txt
git diff > output/baseline50/working-tree.patch
git status --short > output/baseline50/git-status.txt
ollama --version > output/baseline50/ollama-version.txt
ollama list > output/baseline50/ollama-models.txt
```

Also save exact launch commands and settings: `top_k=10`, prompt, model IDs,
CPU/full-precision reranker, context length, and generation/extraction settings.
Copy any untracked source files separately; `git diff` does not capture them.

**Completion check:** Another run can identify the same inputs, code, and settings.
Do not change settings midway through a resumable run.

## 8. Extract the knowledge graphs — turned sentences into relational facts

**What it does:** Calls Llama 3 for each candidate sentence and saves triples
linked to their paragraph and sentence index. Uses a separate baseline directory.
Completed paragraph files are skipped when rerunning after an interruption.

```bash
mkdir -p output/baseline50/kgs
time python -u code/preprocess/hotpot_extraction.py \
  --data_path output/baseline50/hotpot_50.json \
  --kg_dir output/baseline50/kgs \
  --model_name llama3:8b
```

**Completion check:** Extraction finishes without an exception and writes graph
JSON files. The progress bar advances per question, not per sentence. Record the
elapsed extraction time separately from prediction time. An interrupted paragraph
may need to be processed again; completed paragraph files remain reusable.

## 9. Check graph coverage — verified the extracted files

**What it does:** Checks for missing files and counts paragraphs with usable facts.
An existing JSON file can contain no triples.

```bash
python - <<'PY'
import json
from pathlib import Path

data = json.loads(Path('output/baseline50/hotpot_50.json').read_text())
titles = {title for sample in data for title, _ in sample['context']}
missing = []
nonempty = 0
for title in titles:
    path = Path('output/baseline50/kgs') / (title.replace('/', '_') + '.json')
    if not path.exists():
        missing.append(title)
        continue
    graph = json.loads(path.read_text())
    nonempty += any(bool(triples) for triples in graph.values())
print('Expected paragraph titles:', len(titles))
print('Missing files:', len(missing))
print('Paragraphs with facts:', nonempty)
if missing:
    print('First missing titles:', missing[:10])
PY
```

**Completion check:** Investigate missing files, malformed JSON, or zero usable
graphs before prediction. Inspect filename collisions if distinct titles become
identical after replacing `/` with `_`. Coverage does not establish fact accuracy.

## 10. Run baseline prediction — selected evidence and generate answers

**What it does:** Embeds candidate sentences, retrieves seed sentences, expands
through the graph, organizes/ranks evidence, and asks Llama 3 for an answer.
Predictions include the answer and supporting-fact locations.

Run from the repository root after Steps 5–9:

```bash
time python -u code/kg_rag_distractor.py \
  --dataset hotpotqa \
  --data_path output/baseline50/hotpot_50.json \
  --kg_dir output/baseline50/kgs \
  --result_path output/baseline50/predictions.json \
  --model_name llama3:8b \
  --embed_model_name mxbai-embed-large \
  --reranker BAAI/bge-reranker-large \
  --reranker_device cpu \
  --top_k 10
```

To resume after interruption or retry failures, run the same command with
`--resume` appended. No need to regenerate the dataset or completed graph files.
Use a new result path if changing the source, dataset, graphs, or model settings.

Use these settings; paths below are relative to the repository root:

| Setting | Value |
|---|---|
| Dataset | `hotpotqa` |
| Questions | `output/baseline50/hotpot_50.json` |
| Graphs | `output/baseline50/kgs` |
| Predictions | `output/baseline50/predictions.json` |
| Generator | `llama3:8b` |
| Embeddings | `mxbai-embed-large` |
| Reranker | `BAAI/bge-reranker-large`, CPU, full precision |
| Retrieval/evidence setting | `top_k=10` |

**Completion check:** Nonzero graphs are loaded, incremental checkpoints appear,
and all 50 question IDs finish successfully. Resolve or report failures explicitly.
Record prediction runtime and model placement; distinguish cold and warm runs.

## 11. Evaluate the baseline — measureed answer and evidence quality

**What it does:** Compares predicted answers and supporting facts with the labels.
First verify all 50 IDs have successful predictions. This repository's evaluator
skips missing answers and divides by the number present, which can make an
incomplete run misleading. It does not enforce complete coverage.

```bash
python code/util/hotpot_evaluate_v1.py \
  output/baseline50/predictions.json \
  output/baseline50/hotpot_50.json
```

| Metric | Meaning |
|---|---|
| `em` | Exact normalized answer match |
| `f1` | Answer token overlap |
| `sp_prec` | Fraction of selected evidence labeled as supporting facts |
| `sp_recall` | Fraction of gold supporting facts selected |
| `sp_f1` | Balance of supporting-fact precision and recall |
| `joint_f1` | Combined answer and supporting-fact performance |

**Completion check:** Save these metrics alongside the configuration, elapsed
times, and success/failure counts. A wrong answer alone is not an execution error.

## 12. Inspect examples — determine where the pipeline fails

**What it does:** Separates missing evidence from incorrect answers despite
having the necessary evidence. Inspect several correct and incorrect answers.

For each question, compare predicted `[title, index]` pairs with gold
`supporting_facts` and read the corresponding context sentences. Record missing
gold facts, extra selected sentences, and whether the model used the evidence.
Extra non-gold context is not automatically harmful; measure its effect.

**Completion check:** Have a short error analysis and measured runtime before
scaling beyond this development subset.

## 13. Compare filtering methods — test the research hypothesis

**What it does:** Tests whether selecting evidence after graph expansion improves
results relative to the saved baseline. The SEM class and optional runner stage
are now implemented. See `CLASS_DOCS.md` for the code map and
`output/sem50/README.md` for offline checks, a live smoke run, and the 50-question
comparison commands. A live SEM experiment has not been verified here. CROSS
sentence filtering and automated GRAPH recovery remain future implementation work.

Compare the original pipeline with SEM, CROSS, and CROSS plus automated GRAPH
recovery. State whether methods are alternatives or sequential stages. Use the
same questions, cached graphs, generator, and evidence budget. Tune thresholds
and graph rules on development questions only.

Measure supporting-fact precision/recall/F1, answer EM/F1, input tokens,
filtering overhead, and end-to-end prediction time. Then compare a text-only
bi-encoder with a graph-enriched bi-encoder. Freeze settings before running
separate held-out questions. Improvements on these 50 development questions
alone do not establish final research performance.
