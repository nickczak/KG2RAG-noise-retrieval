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
is the dependency source of truth; `code/requirements.txt` forwards to it. It pins direct
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
not call Ollama, extract graphs, or download models. Model responses and embedding generation in the tests are mocked; some tests
also exercise the real LlamaIndex retrieval/index wiring. A green check is followed by a live smoke run, not treated
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
The workflow accepts `baseline`, `sem`, `cross`, `sem_cross`, `joint`, `reuse`,
and `path_reuse`. See `CLASS_DOCS.md` for selector behavior.
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

For the CROSS sentence-filter experiment:

```bash
./experiment new cross_next --method cross --threshold 0.5
./experiment graphs cross_next
./experiment run cross_next --smoke
./experiment evaluate cross_next --smoke
./experiment run cross_next
./experiment evaluate cross_next
./experiment compare baseline50 sem50 sem50_t03 cross_next
```

The default CROSS model is `cross-encoder/ms-marco-MiniLM-L6-v2`; override it with
`--cross-model` when creating a new experiment. Its default predictions are raw
relevance scores (logits), not cosine similarities or probabilities. Threshold
0.5 is an initial development choice, not 50% confidence or an optimal cutoff.
CROSS accepts any finite threshold; it does not share SEM's [-1, 1] score range.
The filter keeps sentences above the cutoff, retaining the single highest-scoring
sentence if none pass. The original downstream BGE group reranker stays in place.

To combine SEM followed by CROSS, use separate cutoffs for their different scales:

```bash
./experiment new sem_cross_next --method sem_cross \
  --sem-threshold 0.3 --cross-threshold 0.5
./experiment graphs sem_cross_next
./experiment run sem_cross_next --smoke
./experiment evaluate sem_cross_next --smoke
./experiment run sem_cross_next
./experiment evaluate sem_cross_next
./experiment compare baseline50 sem50_t03 cross50 sem_cross_next
```

The pipeline is expansion → SEM → CROSS → original graph organization/BGE →
presentation → generation. CROSS scores only SEM survivors; it cannot recover a
sentence removed by SEM. Each stage retains its single best sentence if none
pass, preserving original retrieval scores. Both encoders load once on CPU.
The initial cutoffs reuse the earlier development settings; no combined benefit
is assumed. `--threshold` is rejected for `sem_cross` to avoid confusing cosine
and raw relevance scores. Both model names and thresholds are checkpointed.

To score all candidates with both models and make one joint keep decision:

```bash
./experiment new joint_next --method joint --joint-keep-k 10 --cross-weight 0.7
./experiment graphs joint_next
./experiment run joint_next --smoke
./experiment evaluate joint_next --smoke
./experiment run joint_next
./experiment evaluate joint_next
./experiment compare baseline50 cross50 sem_cross50 joint_next
```

`joint` has no SEM/CROSS cutoff gates. Both models score the complete expanded
candidate pool. `select_joint_chunks` in `code/util/kg_post_processor.py` converts
each score list into rank percentiles (equal scores receive average tied ranks),
then combines them as `cross_weight * CROSS_rank + (1 - cross_weight) * SEM_rank`.
Raw cosine scores and logits are never averaged. The best CROSS match is reserved
first; remaining slots follow fused rank, up to `joint_keep_k`. Even at weight
zero the CROSS anchor is still reserved. Positive budgets guarantee at least one
candidate for nonempty input. Returned chunks keep their original order/scores.

The cap applies before the original graph organization/BGE stage; that later
stage may remove more evidence, including the reserved candidate. This is neither
a learned selector nor new graph recovery. It cannot verify factual correctness.
Per-chunk raw/fused score traces are not saved; candidate/kept counts are printed.
Use a fresh name for a new budget/weight. Both model IDs, budget, and weight are
checkpointed. `--threshold`, `--sem-threshold`, and `--cross-threshold` are rejected
for this method. Keep `top_k` comparable, record actual context size, and compare
with CROSS alone before claiming any benefit from joint selection.

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
`config.json`, and writes `summary.json` with mean evidence sentence counts and
the available `usage.json` report. Prediction writes `usage.json` after every
question, so token/latency totals update during a long run and survive an
interruption. Prompt/completion tokens come from Ollama's response counters when
present; otherwise LlamaIndex estimates them with its tokenizer. The report marks
whether all recorded token counts came directly from Ollama. It includes query
latency and full per-question pipeline latency; extraction calls are outside its
scope. `./experiment compare` also prints usage means ± standard deviations when
the runs have reports.
Without those checks the repository evaluator can silently skip missing answers.

Comparison recomputes metrics rather than trusting potentially stale summaries.
It rejects different data, graph fingerprints, generator/embedding/reranker
settings, or `top_k`. Different pipeline-source fingerprints produce a warning:
review the code differences before attributing an effect to filtering. Model names
alone do not guarantee identical weights/server defaults. Check saved model IDs,
environment, and repeated-run variation for a controlled research comparison.

Optimize answer quality, supporting-fact recall/precision, context size, token
use, and latency together. Mean sentence counts are not tokens. These 50 questions are development
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


## Native score reuse

The `reuse` and `path_reuse` methods use KG2RAG's existing mxbai seed similarities and BGE
reranker, rather than loading the additional MiniLM SEM/CROSS models. Original
methods remain available for reproducing the earlier experiments.

Seed cosine scores are retained from the original retrieval. For sentences added
by expansion, cosine similarity is calculated from the query and sentence vectors
already in the index, without another embedding-model call. Propagated graph
scores are kept separate: they are not semantic similarity scores.

BGE scores the same candidate group texts used by the original graph organizer.
A group's semantic score is the mean cosine of its unique source sentences.
Rank percentiles combine BGE (weight 0.7) and semantic relevance (weight 0.3).
The prepared reuse experiment keeps complete groups within a five-sentence
budget, protecting the
highest BGE group when it fits. Oversized groups are skipped. If no group fits,
individual sentences are scored with BGE and selected instead; group scores are
never substituted for sentence scores. This changes selection behavior and must
be evaluated as a new method.

An exact BGE cache is shared under `output/score_cache/`. Keys include checkpoint
file fingerprints, library versions, scoring settings, question, and exact input
text. New or changed inputs still require inference. Historical runs did not save
these scores and cannot populate the cache retroactively. Cached BGE weights load
only when a miss requires inference. The ordinary baseline can opt into this
cache with `--score-cache output/score_cache` when creating an experiment.
The reranker checkpoint must already be downloaded; an ordinary baseline smoke
run can download it first. Initial RAG indexing and query embedding still run;
this does not add a persistent embedding-index cache.

The original native-score reuse runs and the later path runs are complete. The
initial fixed-budget comparison showed that reuse can reduce context size, but
the first run did not beat the baseline on answer EM/F1. See `output/README.md`
for the current controlled comparisons and limitations.

```bash
./experiment run native_baseline50
./experiment evaluate native_baseline50
./experiment run native_reuse50
./experiment evaluate native_reuse50
./experiment compare native_baseline50 native_reuse50
```

Use `./experiment new NAME --method reuse --joint-keep-k 5 --cross-weight 0.7`
for a new experiment. Both methods must use the same downloaded BGE checkpoint
and scoring settings for a controlled comparison. Existing MiniLM joint results
are a different scoring method, not a before/after caching benchmark.

Each cache-enabled run writes `score_reuse.json` with hit/inference counters.
The reuse method also records seed/expanded-vector counts, group scores, and
selection traces. Model inference time excludes model loading. Cache contents
are ignored by Git; the small score audit is trackable. Smoke audits live in the
ignored `smoke/` directory.

The first live check reused 10 seed scores, made no extra embedding calls, and
served 9 of 10 BGE inputs from the baseline-populated cache. One changed group
required inference. This question added no expanded sentences; expanded-vector
reuse is verified by offline tests. Both smoke answers were incorrect, with
supporting-fact recall 1.0; this is a workflow check, not a quality result.


## Experimental path-aware native reuse

Create a distinct experiment with `--method path_reuse`. It uses the native
retrieval cosine scores and cached BGE sentence scores to rank candidate endpoints.
It builds an undirected candidate-sentence graph by connecting sentences whose
extracted triples share an entity. Unweighted shortest paths connect each selected endpoint to an original
dense-retrieval seed within `joint_keep_k`. The per-question
audit records seeds, paths, bridges, and endpoint scores in `score_reuse.json`.
This is a first heuristic adaptation of path retrieval, not the learned recurrent
retriever from Asai et al. Disconnected endpoints are excluded.

```bash
./experiment new path_k5_cap4_next --method path_reuse \
  --top-k 5 --joint-keep-k 4 --cross-weight 0.7
./experiment graphs path_k5_cap4_next
./experiment run path_k5_cap4_next --smoke
./experiment evaluate path_k5_cap4_next --smoke
./experiment run path_k5_cap4_next
./experiment evaluate path_k5_cap4_next
```

The triple-based links are noisy proxies for evidence transitions; inspect saved
paths before interpreting QA metrics. The selected configuration uses five
dense retrieval seeds, `cross_weight=0.7`, and a four-sentence final evidence
budget (`--top-k 5 --joint-keep-k 4`). Its completed development result is in
`output/path_k5_cap4_temp0/README.md`. Other path-algorithm implementations were
removed after evaluation; their output records remain for reproducibility.

For a matched baseline, use the same `--top-k` and evidence budget. The
baseline-only `--final-keep-k` option separates its final sentence cap from
seed count. For example, compare baseline `--top-k 5 --final-keep-k 4` with
`path_reuse --top-k 5 --joint-keep-k 4`. `./experiment compare` checks these
budgets, data/KG hashes, and model/runtime settings before printing metrics.

New named experiments set and record Ollama `temperature=0.0` so generated-answer
comparisons are more repeatable. Historical run records created before this
setting do not contain a temperature field and are not directly comparable on
answer metrics. Evidence precision/recall/F1 are determined by retrieval, but
still reflect the sampled question set.


The weighted Dijkstra, PCST-inspired, compact-seed, whole-path BGE, and
entity-diversity variants were removed after weaker results. Their completed
experiment records remain under `output/` for reference.
