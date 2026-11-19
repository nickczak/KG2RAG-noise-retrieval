# KG²RAG retrieval-noise experiments

This project extends [nju-websoft/KG2RAG](https://github.com/nju-websoft/KG2RAG),
the implementation of Zhu et al., *Knowledge Graph-Guided Retrieval Augmented
Generation* (NAACL 2025). The original framework and published results belong to
that work. This fork investigates sentence filtering after graph expansion.

The distractor runner supports the original pipeline (`baseline`) and a pretrained
bi-encoder semantic filter (`sem`). CROSS sentence filtering and automated graph
recovery are not implemented yet. The fullwiki runner has not been extended.

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
checkpoint behavior. The [original walkthrough](output/smoke/README.md) is retained.

## Recorded development results

These runs use the same fixed 50 questions; scores are percentages.

| Method | Answer EM | Answer F1 | Evidence precision | Evidence recall | Evidence F1 | Mean sentences |
|---|---:|---:|---:|---:|---:|---:|
| Original KG²RAG | 32.0 | 52.6 | 21.3 | 83.0 | 33.4 | 9.58 |
| SEM, threshold 0.5 | 28.0 | 43.2 | 36.7 | 58.2 | 38.1 | 5.38 |
| SEM, threshold 0.3 | 36.0 | 52.1 | 22.3 | 77.9 | 33.7 | 8.76 |

All three runs have 50 predictions and no recorded failures. SEM 0.5 selects fewer
sentences but loses supporting facts and answer F1. SEM 0.3 recovers much of that
loss and roughly matches baseline answer F1 with fewer sentences. This is not a
reliable overall gain or a held-out result; tokens, latency, and repeated-run
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
