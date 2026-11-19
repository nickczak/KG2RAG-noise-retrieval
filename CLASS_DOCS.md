# Classes and local changes

This file describes the local extension of the original KG²RAG code, including
the user's SEM implementation and the checkpointing support added during setup.
It is a code map, not a claim that the experimental method improves retrieval.

For the daily edit/test/run loop, use `./experiment` and [WORKFLOW.md](WORKFLOW.md).

## Pipeline and data objects

The distractor runner uses the ten candidate paragraphs supplied per HotpotQA
question. It creates sentence-sized `TextNode` objects, retrieves seed sentences,
and applies postprocessors in order before answer generation.

```text
baseline: sentence retrieval → KG expansion → graph organization/BGE ranking
          → presentation ordering → Llama 3 answer

sem:      sentence retrieval → KG expansion → SEM sentence filtering
          → graph organization/BGE ranking → presentation ordering → Llama 3 answer
```

`TextNode` holds sentence text and its ID. `NodeWithScore` wraps it with a retrieval
score. `QueryBundle` carries the question. For HotpotQA, IDs use
`paragraph title##sentence index`; the saved evidence uses `[title, index]` pairs.

`doc2kg[title][sentence_index]` holds extracted `[head, relation, tail]` triples.
`chunks_index[title][sentence_index]` holds the corresponding sentence text.
Here `ents` in the expansion processor means candidate paragraph titles, not every
entity phrase extracted from text.

## Postprocessor classes: code/util/kg_post_processor.py

### KGRetrievePostProcessor — original KG expansion

Receives the seed `NodeWithScore` objects. Reads their triples, matches endpoints
to candidate paragraph titles, and propagates title-level scores. After the
initial seed-triple lookup it performs an additional traversal loop (`hops = 1`).
It gathers additional sentences from eligible related paragraphs and strongly
represented seed paragraphs, attaches heuristic scores, and appends them to the
seeds. This uses saved triples; it does not ask an LLM to re-extract the graph.

### SemanticFilterPostProcessor — user's new SEM filter

Fields:

| Field | Meaning |
|---|---|
| `encoder` | A shared pretrained `SentenceTransformer`, loaded once per run |
| `threshold` | Minimum question–sentence cosine similarity to retain |

`_postprocess_nodes(nodes, query_bundle)`:

1. Returns an empty list immediately for empty input.
2. Independently embeds the question and sentence texts with normalization.
3. Computes one cosine score per sentence via matrix–vector multiplication.
4. Keeps sentences whose score is greater than or equal to the threshold.
5. If none qualify, keeps exactly one sentence with the highest semantic score
   (the first maximum if scores tie).

The returned objects retain their original IDs, text, order among retained nodes,
and retrieval scores. Semantic scores control selection but do not replace the
scores used by downstream graph organization. The class uses an existing encoder;
it does not fine-tune a bi-encoder or encode graph topology.

The runner supplies a `QueryBundle` and an initialized encoder. Direct callers
must do the same for nonempty inputs. Per-sentence SEM scores and removal counts
are not currently saved, and gold-support retention at the filtering boundary
is not yet instrumented.

### GraphFilterPostProcessor — original graph organization and reranking

Builds an undirected multigraph from candidate triples. Edges store relation,
source sentence ID, and the sentence's current score. It matches question mentions
using character overlap and adds artificial query-cooccurrence edges. Connected
components are organized with maximum spanning trees. Tree-edge source sentences
form groups that BGE scores jointly with the question. Groups are selected within
the `topk` unique-sentence budget. Without graph groups, individual sentences are
ranked instead. A small selection triggers an individual-sentence fill-in branch.

This is already a cross-encoder ranking stage. A new experimental CROSS sentence
filter has not been implemented. SEM runs before this existing processor.

Known inherited limitations, left unchanged for the saved baseline:

- Fill-in scores refer to a filtered candidate list, but their indices are applied
  to the original node list, which can select the wrong sentences.
- Group selection stops at the first group exceeding the budget rather than
  checking whether a later smaller group fits.
- Calls assume reranker results are iterable. A single candidate may produce a
  scalar depending on FlagEmbedding behavior; test the one-survivor SEM path
  end to end before a full run.

These are distinct from SEM implementation correctness. If corrected, record
the fixes and rerun a matching baseline before attributing changes to SEM.

### NaivePostprocessor — original presentation ordering

Groups selected evidence by paragraph, orders its sentences, and removes repeated
title prefixes. It formats evidence for the generator; it is not a relevance model.

## Runner changes: code/kg_rag_distractor.py

| Function | Responsibility and local changes |
|---|---|
| `init_model` | Configures Ollama generation and embedding models |
| `process_sample` | Creates sentence candidates and the query engine; now accepts a shared reranker and optional shared SEM encoder |
| `kgrag_distractor_predict` | Loads models once for pending questions; adds SEM encoder loading only for `method=sem`; skips successes, records failures, checkpoints after each attempt |
| `atomic_write` | Writes a temporary file in the destination directory, flushes it, then replaces the destination |
| `write_prediction` | Exports evaluator-compatible predictions using atomic writes |
| `run_config` | Fingerprints input data, loaded graphs, runner/util source, and explicit model/retrieval/SEM settings |
| `load_checkpoint` | Validates the run configuration and question IDs; refuses accidental overwrites |
| `save_checkpoint` | Saves the authoritative sidecar first, then prediction export |
| `main` | Reads data/graphs, validates checkpoint state, and starts or resumes prediction |

New CLI settings:

| Option | Default / behavior |
|---|---|
| `--reranker_device` | `cpu` |
| `--reranker_fp16` | Disabled unless supplied |
| `--resume` | Resume a matching sidecar and retry failures |
| `--method` | `baseline`; alternatively `sem` |
| `--sem_model` | `sentence-transformers/all-MiniLM-L6-v2` |
| `--sem_threshold` | `0.5`, an initial development choice rather than an established optimum |

The method, SEM model, and threshold are included in checkpoint configuration;
SEM model/threshold are `None` in baseline mode. The sidecar
`<result_path>.checkpoint.json` contains configuration, successful answers/evidence,
and recorded errors. It is authoritative if interruption occurs between sidecar
and prediction-export writes. Questions with exceptions or empty answers remain
retryable; a run with recorded failures exits with an error.

Source edits also change the fingerprint. The completed original baseline records
must remain as captured; do not rewrite their settings or try to resume their old
checkpoint under the edited source. Use a new output path for new experiments.
Fingerprints do not capture silent model-weight changes behind a model name,
server defaults, or every dependency; retain environment/model records as well.

## Review fixes and tests

The user's implementation added the SEM class, imports, optional pipeline stage,
one-time encoder loading, encoder argument passing, CLI settings, and SEM entries
in the run configuration. Syntax fixes included the threshold comparison and
encoder constructor comma. During the latest review, the remaining missing colon
in the `sem_threshold` configuration entry was corrected.

The requested test update adds `method='baseline'`, `sem_model=None`, and
`sem_threshold=None` to the test `argparse.Namespace`. This object stands in for
parsed CLI arguments; it must define the fields accessed by production code.

- `tests/test_checkpoint.py`: interruption/resume, failure retry, overwrite
  protection, changed inputs/settings, SEM configuration mismatch, and one-time
  encoder loading/passing.
- `tests/test_sem_filter.py`: fixed-vector checks for threshold equality,
  original-score preservation, semantic fallback, and empty input.

Run from the repository root with `.venv` active:

```bash
python -m py_compile code/kg_rag_distractor.py code/util/kg_post_processor.py
python -m unittest discover -s tests -v
```

These tests use mocked predictions/encoder outputs and do not download weights or
run Ollama. A live SEM smoke run and measured evaluation remain separate checks.
The initial SEM review passed nine offline tests, compilation, and the runner's
`--help` import/CLI check. The expanded suite also covers workflow orchestration;
run `./experiment check` for the current count. The user subsequently completed
SEM 0.5 and SEM 0.3 on all 50 development questions; see the root README for metrics.

## Experiment workflow: scripts/experiments.py

The `experiment` shell launcher selects `.venv` and calls the Python CLI.
The workflow module uses functions rather than classes:

| Function | Responsibility |
|---|---|
| `prepare` | Fixed-seed question sampling, prompt-example exclusions, ID manifest |
| `new` / `settings` | Named launch configuration, validation, overwrite protection |
| `graph_coverage` / `graphs` | Check missing files, filename collisions, and triple shape |
| `extract` | Run the extraction CLI, reusing completed graph files |
| `run` | Build runner command; separate smoke/full outputs; explicit resume |
| `record_environment` | Capture local packages, commit, model list, launch command, source snapshots |
| `lock` / `timed_call` | Prevent concurrent wrapper writers and record attempt timing |
| `completed` | Check exact prediction coverage, nonempty answers, evidence locations, fingerprints, and errors |
| `evaluate` | Recompute official metrics and save actual config/summary |
| `compare` | Recompute and compare compatible runs, warning about different source revisions |
| `status` | Report local completion/failure counts |
| `check` | Parse source, run all or selected offline tests, check whitespace |

`experiment.json` is the intended configuration; `config.json` records the actual
completed runner configuration. Existing historical runs can be compared without
creating new launch configurations or rewriting their provenance.

The extractor in `code/preprocess/hotpot_extraction.py` now has `main(args)` and
CLI path/model/timeout options instead of import-time execution. The triple prompt
and parser are unchanged. It reports sentence progress, detects filename collisions,
reuses valid completed JSON files, and atomically saves each paragraph. Previously
documented source-string replacement is no longer needed.

`tests/test_experiments.py` checks command construction, reproducible sampling,
coverage/config validation, malformed graph detection, output locks, and comparison
guards, provenance snapshots, and failed-attempt timing with temporary fixtures,
not model calls. `tests/test_extraction.py` checks graph-cache reuse, interrupted
paragraph writes, and filename collisions with mocked extraction responses.

## Experiment records

- `output/baseline50/`: completed original-pipeline development baseline. No SEM
  filtering was applied; its saved metrics must remain unchanged.
- `output/sem50/README.md`: SEM smoke/evaluation commands and result recording.
- `output/sem50_t03/README.md`: the completed lower-threshold experiment and its
  historical launch/evaluation scripts. New runs use the shared workflow instead.
- `output/smoke/README.md`: environment, dataset selection, extraction, and baseline
  walkthrough.

The fullwiki runner has not been wired to the new SEM stage. CROSS filtering,
automated graph recovery, and budgeted subgraph selection remain future work.
