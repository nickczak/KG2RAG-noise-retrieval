# Code map for this KG2RAG fork

For runnable steps see [WORKFLOW.md](WORKFLOW.md). For changes relative to the
original system see [CHANGES_FROM_UPSTREAM.md](CHANGES_FROM_UPSTREAM.md).
These extensions target the distractor runner, not the fullwiki runner.

## Pipeline and selection methods

The common pipeline is sentence indexing → dense seed retrieval → KG expansion →
method-specific selection/organization → paragraph presentation → Ollama answer.

| Method | Behavior after expansion |
|---|---|
| `baseline` | Original graph construction, spanning-tree groups, BGE ranking, budget selection |
| `sem` | MiniLM cosine cutoff, then original graph/BGE stage |
| `cross` | MiniLM raw relevance-score cutoff, then original graph/BGE stage |
| `sem_cross` | SEM, then CROSS on SEM survivors, then original graph/BGE stage |
| `joint` | MiniLM SEM/CROSS rank fusion and candidate cap, then original graph/BGE stage |
| `reuse` | Native cosine values and BGE group rank fusion inside graph organization |
| `path_reuse` | Rank endpoints by native cosine/BGE, connect sentences that share extracted entities, and retain shortest routes from initial retrieval seeds under the final evidence cap |

SEM/CROSS cutoff filters keep their highest scoring candidate if none pass.
They preserve node objects, order, and original retrieval scores. SEM cosine and
CROSS logits have different scales; a CROSS cutoff is not a probability.
Joint fusion uses average tied rank percentiles, reserves the best CROSS
candidate, and fills the remaining slots by fused rank. The downstream original
stage can still remove the joint anchor. These methods use pretrained scorers
and heuristic selection rules. The active `path_reuse` configuration uses five
initial seeds, rank fusion with CROSS weight 0.7, shortest seed-rooted paths,
and a four-sentence final cap. Its development result is recorded in
`output/path_k5_cap4_temp0/`; confirm the settings on fresh held-out questions.
Named runs retain settings, selected evidence, and metrics under `output/`.
New runs record Ollama temperature and default to `0.0`; older runs without a
recorded temperature are not directly comparable on generated-answer metrics.

## code/util/kg_post_processor.py

| Class/function | Responsibility |
|---|---|
| `KGRetrievePostProcessor` | Existing KG-guided sentence expansion; propagated node scores are graph heuristics, not new cosine scores |
| `GraphFilterPostProcessor` | Existing group construction/BGE organization plus an optional native reuse selection branch |
| `NaivePostprocessor` | Existing paragraph/sentence presentation ordering and title formatting |
| `SemanticFilterPostProcessor` | Additional bi-encoder cosine filter |
| `CrossEncoderFilterPostProcessor` | Additional cross-encoder sentence filter |
| `_rank_percentiles` | Average tied ranks on a common scale |
| `select_joint_chunks` | Validated rank fusion, CROSS anchor, budget, stable return order |
| `JointFilterPostProcessor` | Calls both auxiliary scorers over every expanded candidate and delegates selection |

The reuse branch computes each group's mean native cosine over unique source
sentences and fuses its rank with the BGE rank. It protects the best BGE group
when it fits, skips oversized groups, and keeps complete groups under the
sentence budget. If none fit, it scores individual sentences with BGE and applies
joint selection. Group scores are never treated as individual sentence scores.
Its audit records the raw/group/fused values and selected IDs.

Known legacy behavior retained for historical baseline compatibility: fill-in
ranking indexes a filtered candidate list but applies positions to the original
list, group selection stops at the first oversized group, and some calls assume
iterable reranker outputs. The native branch has its own group-budget/fallback
handling; this is a method change, not proof of unchanged baseline behavior.
Any general legacy fixes require fresh matched baseline runs.

## code/util/score_reuse.py

| Class/function | Responsibility |
|---|---|
| `pinned_reranker` | Resolve a downloaded checkpoint and hash its files and relevant library versions |
| `CachedReranker` | Lazy BGE scoring with persistent SQLite exact-input caching, batch deduplication, finite-output validation, and counters |
| `NativeScoreStore` | Retain original seed cosine scores and calculate added-chunk cosine from existing vectors |
| `NativeScoreRetriever` | Capture the original retriever's unchanged seed result and query vector; supports sync/async calls |
| `NativeSemanticScorePostprocessor` | Fill added-chunk similarities without extra embedding calls |

Cache keys include model identity, scoring settings, exact question/text, and
call options. Changed group text/order correctly misses the cache. Historical
runs did not save BGE scores, so they cannot populate it retroactively. Initial
index/query embeddings are still computed per run; only the added cosine
calculation avoids another embedding call. The persistent cache stores BGE scores,
not a reusable vector index.

## code/kg_rag_distractor.py

| Function | Responsibility |
|---|---|
| `init_model` | Ollama generator and embedding configuration plus per-call token/latency callback |
| `build_postprocessors` | Explicit method-specific chain |
| `process_sample` | Candidate/index/retriever/query engine construction and native score audit |
| `kgrag_distractor_predict` | Load needed models once, skip successes, retry failures, checkpoint each attempt |
| `atomic_write` / `write_prediction` | Atomic sidecar/export writes |
| `run_config` | Input, loaded-graph, source, model, and active method settings |
| `load_checkpoint` / `save_checkpoint` | Matching-state resume and overwrite protection |
| `main` | Data/graph loading, optional shared lazy reranker, prediction, usage and score-audit reporting |

The runner defaults to CPU/full-precision reranking. Active method model IDs,
thresholds, rank budget/weight, and native reranker identity are checkpointed;
inactive method settings are `None`. Raw runner flags use underscores, for
example `--joint_keep_k`, baseline `--final_keep_k`, and `--score_cache_dir`;
workflow flags use hyphens. `./experiment compare` validates both seed count
and the effective final evidence cap.
Use each command's `--help` for the complete defaults/options.

`<result_path>.checkpoint.json` is authoritative and includes answers, evidence,
errors, configuration, and native audits when applicable. Successful IDs are
skipped on resume; exceptions/empty answers remain retryable. A run with failures
exits with an error. Source/settings changes invalidate resume. Cache-enabled
runs also write `score_reuse.json` with per-attempt counters and question traces.
Prediction runs write `usage.json` after each question with prompt/completion
tokens, query latency, and full per-question pipeline latency; extraction is not
included.
The native cache fingerprints BGE files/settings; model names alone in older
runs cannot establish identical weights or Ollama server defaults.

## Workflow and extraction

`experiment` selects the checkout's `.venv` and calls `scripts/experiments.py`.

| Command/functions | Responsibility |
|---|---|
| `prepare` | Fixed-seed sampling, prompt-example exclusion, ID manifest |
| `new` / `settings` | Named configuration, validation, overwrite protection |
| `graph_coverage` / `graphs` | File presence, collisions, JSON/triple shape |
| `extract` | Extract missing graphs under a shared-directory lock |
| `run` | Separate smoke/full outputs, explicit resume, source/model/environment provenance |
| `lock` / `timed_call` | Prevent competing writers and record attempt wall time |
| `completed` / `evaluate` | Coverage/evidence/settings checks, official metrics, actual config and summary |
| `compare` | Recompute compatible-run metrics and flag source differences |
| `status` / `check` | Completion status / offline syntax, tests, whitespace |

`experiment.json` records intended settings; `config.json` records actual
completed settings. Historical runs without a launch manifest can be compared
but must not be relaunched into their completed folders.

`code/preprocess/hotpot_extraction.py` has import-safe CLI execution, path/model/
timeout options, sentence progress, collision checks, valid graph reuse, and
atomic paragraph writes. Original triple prompts/parser are retained. Source
string replacement and temporary CPU copies of the runner are obsolete.

The tested Matryoshka GNN prototype is no longer part of the runnable pipeline;
its historical experiment outputs remain under `output/` for reference.

## Verification and records

Run `./experiment check` from the root. The current suite has 59 offline tests:
checkpoint/resume guards, SEM/CROSS/cascade/joint selection, extraction recovery,
workflow compatibility/provenance, exact score cache invalidation/lazy loading,
unchanged legacy selection with fixed scores, and real LlamaIndex wiring with
mocked embedding generation to check there are no extra embedding calls.
No model download or Ollama call is needed by these tests.

Completed current and historical development runs are listed in [output/README.md](output/README.md).
Native control/reuse runs have completed one-question checks; full results are
pending. The native smoke reused 10 seed scores and 9/10 cached BGE inputs. Weighted
Dijkstra and PCST-inspired variants were removed after matched experiments; their
records remain under output/.
It added no expanded sentences; the expanded-vector path is covered offline.
Predictions, metrics, and provenance remain historical records; no successful
smoke run establishes a quality gain. Learned graph recovery, new subgraph
algorithms, and fullwiki method wiring remain future work.


The `path_reuse` option adds shortest-path retention inside `GraphFilterPostProcessor`.
Candidate sentence nodes connect when their extracted triples share a normalized
head/tail entity. Endpoint ranks fuse native semantic similarity with cached BGE
sentence relevance. Unweighted shortest paths retain connections to an original
retrieval seed. Disconnected candidates are skipped. The audit records path nodes,
hop count, bridge entities, and endpoint scores. This graph is a heuristic built
from extracted triples, not explicit paragraph hyperlinks or a trained path model.
