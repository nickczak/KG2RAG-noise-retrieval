# Implement, test, and compare retrieval changes

## 1. Set up once

Existing machine: `.venv`, Ollama models, and extracted graphs already exist.
No need to reinstall packages or re-extract graphs for each method.

For a fresh machine, from the repository root:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r requirements.txt
```

Install Ollama with NVIDIA support using its Linux instructions, then:

```bash
ollama pull llama3:8b
ollama pull mxbai-embed-large
```

Keep `ollama serve` running if no service is serving Ollama. `ollama ps` reports
loaded model placement; CPU PyTorch does not prevent Ollama from using the GPU.
The first model run can download BGE/MiniLM weights. Root `requirements.txt`
supersedes the upstream `code/requirements.txt` for this extension. It pins direct
dependencies, not every transitive dependency. Historical `environment.txt` files
can include machine-specific packages and should not be installed wholesale.

`./experiment` always selects this checkout's `.venv`; you do not need to activate
it first. Commands below are shown from the repository root.

## 2. Check an implementation without running models

```bash
./experiment check
./experiment check --pattern test_sem_filter.py
```

This parses Python source, runs offline tests, and checks diff whitespace. It does
not call Ollama, extract graphs, or download models. Model responses/embeddings in
the tests are mocked. A green check is followed by a live smoke run, not treated
as proof of answer quality.

To implement another method:

1. Add a top-level `BaseNodePostprocessor` subclass in
   `code/util/kg_post_processor.py`, or a separate module under `code/util/`.
2. Keep candidate IDs and source sentence mappings. Decide explicitly whether the
   method filters candidates, changes scores, or changes grouping.
3. Add the method choice and its settings to `code/kg_rag_distractor.py`. Load
   models once in the run loop and pass them into sample processing. Keep the
   `baseline` branch unchanged.
4. Record every result-affecting option in `run_config`. Source edits intentionally
   invalidate resume; create a new experiment after editing pipeline code.
5. Extend the corresponding config/CLI handling in `scripts/experiments.py`.
   The workflow currently accepts only implemented `baseline` and `sem` methods.
6. Add tests with fixed vectors or mocked scorers. Test empty inputs, one survivor,
   threshold equality, fallback, original-score preservation, pipeline ordering,
   and checkpoint mismatches where relevant.
7. Update `CLASS_DOCS.md`, run `check`, then use a new named experiment below.

## 3. Create a named experiment

For example, to try SEM 0.3 on the existing fixed development set:

```bash
./experiment new sem03_next --method sem --threshold 0.3
```

This only creates `output/sem03_next/experiment.json` and a short README.
It refuses to reuse an existing folder. All launch settings live in one JSON
file instead of copied shell commands. You can edit it before the first run;
once outputs exist, use a new name for changes.

Defaults reuse `output/baseline50/hotpot_50.json` and `output/baseline50/kgs`.
Other options include `--top-k`, `--data`, `--kgs`, `--model`, `--embedding`,
`--reranker`, and `--sem-model`. Run `./experiment new --help` for details.
The workflow uses CPU/full-precision reranking and disables triple-text augmentation.

To establish a fresh matched baseline for a new implementation revision:

```bash
./experiment new baseline_next --method baseline
```

Preserve the completed historical `baseline50`, `sem50`, and `sem50_t03` folders.
They can be evaluated/compared but cannot be launched by this workflow without
creating a new experiment. This prevents accidental modification of past runs.

## 4. Check or create graphs

```bash
./experiment graphs sem03_next
```

The check verifies filename collisions, graph-file presence, JSON, and triple
shape. Empty paragraph graphs are reported but allowed; file availability is not
evidence of factual correctness. Missing graphs block prediction.

If needed (for example, on a fresh clone):

```bash
./experiment extract sem03_next
```

This is a model operation and can take a long time. It extracts only missing
paragraph files with Llama 3, using the original extraction prompt and parser.
The script now has a CLI and sentence-level progress, creates directories, and
writes files atomically. Completed files are reused after interruption. Use a
separate graph directory when deliberately changing extraction settings. Keep
graph extraction separate from filtering experiments.

## 5. Run and evaluate one question first

```bash
./experiment run sem03_next --smoke
./experiment evaluate sem03_next --smoke
```

This uses the first question of the selected development set and writes into
`output/sem03_next/smoke/`. It does not mix smoke predictions with the full run.
This checks the live pipeline, including downloads, Pydantic types, reranker
shapes, graph postprocessors, and answer generation.

To resume the smoke run:

```bash
./experiment run sem03_next --smoke --resume
```

## 6. Run the fixed development set

```bash
./experiment run sem03_next
./experiment status sem03_next
```

At first launch, the workflow records the environment, commit, model list, exact
command, and hashes/local copies of runner and util source. Model calls use the
runner's existing checkpoint support. Each attempt saves elapsed wall time and
exit status in `timing.json`. Time includes startup and model loading; it is not
filter-only latency or a controlled performance benchmark.

After an interruption or a recorded failure:

```bash
./experiment run sem03_next --resume
```

Successful questions are skipped and failed questions retried. Resume requires
matching source, input data, loaded graphs, and recorded settings. Restarting
after a source/threshold change requires a new name. The wrapper holds a local
advisory lock to prevent two wrapper commands writing the same output; do not
launch the raw runner concurrently against that output. Graph extraction has a
separate lock on the shared graph directory. Do not change graphs during prediction.

## 7. Evaluate and compare

```bash
./experiment evaluate sem03_next
./experiment compare baseline50 sem50 sem50_t03 sem03_next
```

Evaluation verifies exact question-ID coverage, nonempty answers, valid unique
evidence locations, data fingerprints, checkpoint errors/export consistency,
and the intended run settings. Then it recomputes `metrics.json`, saves the actual
`config.json`, and writes `summary.json` with mean evidence sentence counts.
Without those checks the repository evaluator can silently skip missing answers.

Comparison recomputes metrics rather than trusting potentially stale summaries.
It rejects different data, graph fingerprints, generator/embedding/reranker
settings, or `top_k`. Different pipeline-source fingerprints produce a warning:
review the code differences before attributing an effect to filtering. Model names
alone do not guarantee identical weights/server defaults. Check saved model IDs,
environment, and repeated-run variation for a controlled research comparison.

Optimize answer quality, supporting-fact recall/precision, and context size
together. Mean sentence counts are not tokens. These 50 questions are development
data, not the final held-out evaluation.

## 8. Prepare a different fixed question set, when needed

```bash
./experiment prepare dev100 --size 100 --seed 42
./experiment new baseline100 --method baseline \
  --data output/dev100/hotpot.json --kgs output/dev100/kgs
```

Preparation excludes the four answer-prompt examples and saves the sampled IDs.
It does not guarantee disjointness from previous samples: explicitly exclude
development IDs before selecting the final held-out set. Use the same fixed
questions and graphs for all methods in a comparison.

## 9. What goes into Git

Tracked/trackable: source, tests, docs, fixed question sets, manifests, launch
settings, completed predictions, metrics, summaries, provenance, timing, and
environment/model records.

Ignored: virtual environments, downloaded weights, graph caches, checkpoint
sidecars, smoke artifacts, local source copies, raw logs, working-tree snapshots,
and credentials. A new experiment name does not require editing `.gitignore`.
The checkpoint sidecar is required to resume locally; completed published runs
can still be evaluated using their saved configuration when it is absent.

Before a commit, inspect `git status` and the diff. Publish complete experiments
and honest limitations; do not treat a passing smoke test or best development
threshold as a held-out research result.
