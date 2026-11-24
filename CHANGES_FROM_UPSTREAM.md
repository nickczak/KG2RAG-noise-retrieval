# Changes from original KG2RAG

Reference: upstream revision `66a28c0f69ad68d17017559294ce1814fae6ec91`,
the last original KG2RAG revision in this checkout before the fork-specific
requirements, walkthroughs, workflow, and filtering changes. This inventory
includes committed fork changes and current local edits, not just the latest
`git diff HEAD`. It does not assert that this is upstream's current remote tip.

## Original implementation files modified

| File | What changed |
|---|---|
| `code/kg_rag_distractor.py` | CPU/device configuration, shared model loading, optional SEM/CROSS/cascade/joint/native methods, atomic prediction/checkpoint writes, resume/error handling, fingerprints, native score audits and exact-cache wiring |
| `code/util/kg_post_processor.py` | SEM/CROSS/joint selectors and rank fusion; optional native-score group selection in GraphFilter; original baseline branch retained |
| `code/preprocess/hotpot_extraction.py` | Import-safe configurable CLI, reusable graph files, collision checks, sentence progress, atomic paragraph writes; original extraction prompt/parser retained |
| `README.md` | Fork scope, six methods, development results, current workflow and links |
| `code/preprocess/readme.md` | Current extraction CLI and cache reuse instructions |
| `model/readme.md` | Complete model-role list, Ollama embeddings, auxiliary-model requirements and native cache setup |

These are the only existing upstream Python implementation files changed: the
runner, postprocessor module, and HotpotQA extractor. The fullwiki runner,
HotpotQA evaluator, other dataset extractors, prompt utility files, and original
dataset files are unchanged relative to the reference. Existing prompts may be
used from modified files; extraction prompt/parser behavior is preserved. The
fullwiki runner imports shared utilities but has no new method/cache wiring.

## Files added by this fork

| File/group | Purpose |
|---|---|
| `code/util/score_reuse.py` | Native seed/vector reuse and exact-input persistent BGE cache |
| `path_reuse` method in `code/util/kg_post_processor.py` | Entity-linked shortest paths from dense retrieval seeds to BGE/semantic-ranked endpoints |
| `experiment`, `scripts/experiments.py` | One-command preparation, extraction, smoke/full execution, resume, validation and comparison |
| `requirements.txt` | Working Python 3.11 direct dependency pins |
| `code/requirements.txt` | Compatibility entry point forwarding to root pins; this file was added by the fork, not original upstream |
| `.gitignore` | Ignore environments, model/graph/score caches and transient artifacts while exposing small research records |
| `WORKFLOW.md` | Current step-by-step implementation/testing instructions |
| `CLASS_DOCS.md` | Current pipeline, class/function map, checkpoint and cache behavior |
| `CHANGES_FROM_UPSTREAM.md` | This upstream-relative inventory |
| `tests/test_checkpoint.py` | Prediction/checkpoint/retry/configuration behavior |
| `tests/test_experiments.py` | Workflow settings, commands, provenance, validation, locks and comparison |
| `tests/test_extraction.py` | Graph reuse, interruption-safe writes and collisions |
| `tests/test_sem_filter.py` | Bi-encoder semantic selection |
| `tests/test_cross_filter.py` | Cross-encoder sentence selection |
| `tests/test_sem_cross.py` | Sequential filter behavior |
| `tests/test_joint_filter.py` | Joint rank fusion, anchor, ties and budgets |
| `tests/test_score_reuse.py` | Exact cache invalidation/lazy loading, native vector reuse and group/fallback selection |
| `output/README.md`, `output/*/README.md` | Research record index and experiment-specific findings/instructions |
| `output/*/*.json`, selected `*.txt` | Fixed question subset/manifest, launch and actual settings, predictions, metrics, evidence review, runtime and provenance/model/environment records |

All `output/` research records are additions relative to upstream. The seven
completed development runs are `baseline50`, `sem50`, `sem50_t03`, `cross50`,
`sem_cross50`, `joint50`, and `joint50_k5`. `native_baseline50` and
`native_reuse50` have prepared launch manifests and completed local smoke checks;
full runs remain pending. Historical records retain their actual source/model
settings rather than being rewritten to describe the current code.

## Cleanup after native score reuse

- Replaced stale/manual smoke and SEM instructions with the shared workflow.
- Removed obsolete `output/sem50_t03/run.sh` and `evaluate.py`; kept historical
  `run_settings.json` and completed results. The shared workflow replaces both.
- Replaced conflicting legacy dependency pins under `code/` with a forwarding
  requirements file; root pins are the single dependency source of truth.
- Retained auxiliary SEM/CROSS/joint implementations: they remain comparison
  methods and are not obsolete merely because native score reuse is available.
- Kept only the selected `path_reuse` selector in the runnable path-method API.
  Removed exploratory fill, DP, marginal, PPR, smoothing, z-score, whole-path
  reranking, and seed-limit branches after they failed to beat the selected
  configuration consistently. Their completed output records remain available.
- Removed the Matryoshka GNN trainer and inference integration from the runnable
  pipeline after its matched result was mixed. Its metrics and local artifacts
  remain documented as historical research records.
- Preserved graphs, downloaded models, cache contents, completed results, and
  checkpoint/provenance records. New source edits require fresh experiment names.

This cleanup does not run a new full experiment, train a bi-encoder/GNN, implement
a new subgraph algorithm, or claim unchanged predictions for the native method.
Changes are local until explicitly committed/pushed.

## Complete current file inventory relative to the reference

`M` means modified original file; `A` means added by this fork. Ignored local
caches/checkpoints/smoke outputs are omitted. Removed fork-only manual scripts
are described above; no original upstream file was deleted.

```text
A  .gitignore
A  CHANGES_FROM_UPSTREAM.md
A  CLASS_DOCS.md
M  README.md
A  WORKFLOW.md
M  code/kg_rag_distractor.py
M  code/preprocess/hotpot_extraction.py
M  code/preprocess/readme.md
A  code/requirements.txt
M  code/util/kg_post_processor.py
A  code/util/score_reuse.py
A  experiment
M  model/readme.md
A  output/README.md
A  output/baseline50/README.md
A  output/baseline50/commit.txt
A  output/baseline50/config.json
A  output/baseline50/environment.txt
A  output/baseline50/hotpot_50.json
A  output/baseline50/metrics.json
A  output/baseline50/ollama-models.txt
A  output/baseline50/ollama-version.txt
A  output/baseline50/predictions.json
A  output/baseline50/selection.json
A  output/cross50/README.md
A  output/cross50/commit.txt
A  output/cross50/config.json
A  output/cross50/environment.txt
A  output/cross50/experiment.json
A  output/cross50/metrics.json
A  output/cross50/ollama-models.txt
A  output/cross50/ollama-version.txt
A  output/cross50/predictions.json
A  output/cross50/provenance.json
A  output/cross50/summary.json
A  output/cross50/timing.json
A  output/joint50/README.md
A  output/joint50/commit.txt
A  output/joint50/config.json
A  output/joint50/environment.txt
A  output/joint50/experiment.json
A  output/joint50/metrics.json
A  output/joint50/ollama-models.txt
A  output/joint50/ollama-version.txt
A  output/joint50/predictions.json
A  output/joint50/provenance.json
A  output/joint50/summary.json
A  output/joint50/timing.json
A  output/joint50_k5/README.md
A  output/joint50_k5/commit.txt
A  output/joint50_k5/config.json
A  output/joint50_k5/environment.txt
A  output/joint50_k5/evidence_review.json
A  output/joint50_k5/experiment.json
A  output/joint50_k5/metrics.json
A  output/joint50_k5/ollama-models.txt
A  output/joint50_k5/ollama-version.txt
A  output/joint50_k5/predictions.json
A  output/joint50_k5/provenance.json
A  output/joint50_k5/summary.json
A  output/joint50_k5/timing.json
A  output/native_baseline50/README.md
A  output/native_baseline50/experiment.json
A  output/native_reuse50/README.md
A  output/native_reuse50/experiment.json
A  output/sem50/README.md
A  output/sem50/config.json
A  output/sem50/metrics.json
A  output/sem50/predictions.json
A  output/sem50_t03/README.md
A  output/sem50_t03/config.json
A  output/sem50_t03/metrics.json
A  output/sem50_t03/predictions.json
A  output/sem50_t03/run_settings.json
A  output/sem_cross50/README.md
A  output/sem_cross50/commit.txt
A  output/sem_cross50/config.json
A  output/sem_cross50/environment.txt
A  output/sem_cross50/experiment.json
A  output/sem_cross50/metrics.json
A  output/sem_cross50/ollama-models.txt
A  output/sem_cross50/ollama-version.txt
A  output/sem_cross50/predictions.json
A  output/sem_cross50/provenance.json
A  output/sem_cross50/summary.json
A  output/sem_cross50/timing.json
A  output/smoke/README.md
A  requirements.txt
A  scripts/experiments.py
A  tests/test_checkpoint.py
A  tests/test_cross_filter.py
A  tests/test_experiments.py
A  tests/test_extraction.py
A  tests/test_joint_filter.py
A  tests/test_score_reuse.py
A  tests/test_sem_cross.py
A  tests/test_sem_filter.py
```

To inspect original implementation changes locally:

```bash
git diff 66a28c0 -- code/kg_rag_distractor.py code/util/kg_post_processor.py code/preprocess/hotpot_extraction.py
```

`git diff` omits untracked files; use `git status --short` to see pending additions.
