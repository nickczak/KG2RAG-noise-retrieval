# KG²RAG retrieval-noise experiments

This project extends [nju-websoft/KG2RAG](https://github.com/nju-websoft/KG2RAG),
the implementation of Zhu et al., *Knowledge Graph-Guided Retrieval Augmented
Generation* (NAACL 2025). The original framework and published results belong to
that work. This fork investigates sentence filtering after graph expansion.

The distractor runner supports these methods:

| Method | Selection stage |
|---|---|
| `baseline` | Original KG expansion and graph/BGE organization |
| `sem` | Additional MiniLM bi-encoder semantic cutoff |
| `cross` | Additional MiniLM cross-encoder relevance cutoff |
| `sem_cross` | SEM cutoff followed by CROSS on survivors |
| `joint` | Joint MiniLM SEM/CROSS rank fusion before graph/BGE organization |
| `reuse` | Native dense similarities and original BGE group scores, with exact score caching |
| `path_reuse` | Native SEM/BGE endpoint ranking, retaining shortest KG-linked paths from dense seeds |

The current research configuration is `path_reuse` with five dense seeds and a
four-sentence output cap. It ranks expanded candidates using cached native
semantic scores and BGE cross scores, links sentences that share extracted KG
entities, and retains high-ranked endpoints with their shortest routes back to
retrieval seeds. On the 50-question development sample it outperformed the
baseline and three-seed path configuration across answer and supporting-fact
metrics while selecting fewer sentences. See [the run record](output/path_k5_cap4_temp0/README.md);
confirm it on fresh held-out questions before treating it as a general result.
The fullwiki runner has not been extended with these modes.

## Daily workflow

With the project's `.venv` installed, run these from the repository root:

```bash
./experiment check
./experiment new sem03_next --method sem --threshold 0.3
./experiment graphs sem03_next
./experiment run sem03_next --smoke
./experiment evaluate sem03_next --smoke
./experiment run sem03_next
./experiment evaluate sem03_next
./experiment compare baseline50 sem50 sem50_t03 sem03_next
```

Use `./experiment run sem03_next --resume` after interruption. Each new method,
threshold, or code revision gets a new experiment name. Existing graphs are reused;
on a fresh checkout, generate them with `./experiment extract sem03_next`.
A model run is started only by `run` or `extract`; `check` stays offline.

See [WORKFLOW.md](WORKFLOW.md) for setup, the command reference, and how to
implement/test another method. [CLASS_DOCS.md](CLASS_DOCS.md) maps the classes and
checkpoint behavior. See [CHANGES_FROM_UPSTREAM.md](CHANGES_FROM_UPSTREAM.md) for the file inventory
relative to original KG2RAG. [Model setup](model/readme.md) lists required models.

## Recorded development results

These runs use the same fixed 50 questions; scores are percentages.

| Method | Answer EM | Answer F1 | Evidence precision | Evidence recall | Evidence F1 | Mean sentences |
|---|---:|---:|---:|---:|---:|---:|
| Original KG²RAG | 32.0 | 52.6 | 21.3 | 83.0 | 33.4 | 9.58 |
| SEM, threshold 0.5 | 28.0 | 43.2 | 36.7 | 58.2 | 38.1 | 5.38 |
| SEM, threshold 0.3 | 36.0 | 52.1 | 22.3 | 77.9 | 33.7 | 8.76 |
| CROSS, raw-score threshold 0.5 | 38.0 | 56.7 | 43.0 | 68.3 | 47.8 | 4.90 |
| SEM 0.3 → CROSS 0.5 | 36.0 | 53.6 | 42.5 | 66.6 | 47.0 | 4.82 |
| Joint rank fusion, cap 10, CROSS weight 0.7 | 34.0 | 53.5 | 21.9 | 83.7 | 34.2 | 9.20 |
| Joint rank fusion, cap 5, CROSS weight 0.7 | 36.0 | 55.7 | 37.0 | 73.7 | 48.5 | 4.74 |

All seven runs have 50 predictions and no recorded failures. SEM 0.5 selects fewer
sentences but loses supporting facts and answer F1. SEM 0.3 recovers much of that
loss and roughly matches baseline answer F1 with fewer sentences. CROSS improves answer and evidence F1 in this development sample with fewer
sentences, but loses recall. None of these findings establish a reliable
held-out gain; tokens, latency, and repeated-run
variation were not measured in these historical runs. Source fingerprints differ
between the original baseline and the SEM implementation; review those changes
and use matched fresh runs when testing new code.

Small experiment records and predictions are trackable under `output/<name>/`.
Graphs, weights, checkpoints, logs, and local source snapshots stay ignored.
Environment snapshots describe the machine that ran an experiment; they are not
portable dependency lockfiles.

![Original KG²RAG framework](framework.jpg)

## Citation
If you find the repository helpful, please cite the following paper.
```bibtex
@inproceedings{KG2RAG,
  title = {Knowledge Graph-Guided Retrieval Augmented Generation},
  author = {Zhu, Xiangrong and 
            Xie, Yuexiang and 
            Liu, Yi and 
            Li, Yaliang and 
            Hu, Wei},
  booktitle = {NAACL},
  year = {2025}
}
```

The combined SEM→CROSS run scored below CROSS alone in this single development
comparison and changed final evidence for only 3/50 questions. The other 47 had
identical ordered evidence, so answer-score differences cannot be attributed
solely to filtering. See [combined results](output/sem_cross50/README.md).

Joint selection at cap 10 retains close to baseline context volume and gold-support
recall, but scores below CROSS alone on answer/evidence F1 in this sample. See
[joint results](output/joint50/README.md) for configuration, runtime, and limitations.

Reducing the joint cap to five improves its development tradeoff: answer F1 is
55.7%, evidence F1 48.5%, and mean context 4.74 sentences. CROSS retains slightly
higher answer F1/precision, while joint cap 5 retains more gold support. See the
[cap-five evidence review](output/joint50_k5/README.md), including concrete missed
answer facts and observed formatting defects in sampled cached graphs. These
findings do not establish held-out performance or causal effects.


Native score reuse is available as `--method reuse`: it combines existing dense
retrieval similarities with the original BGE group scores and caches exact BGE
inputs across runs. No auxiliary MiniLM encoders are loaded in this mode.
See [WORKFLOW.md](WORKFLOW.md#native-score-reuse-prepared-full-comparison-pending)
for selection semantics, prepared experiments, and the one-question validation.
Full native-score comparison results are still pending.


No method is expected to improve every metric automatically. The path-based
experiments trade evidence precision for answer F1/recall on this sample; choose
future objectives explicitly and judge them on matched held-out data.
