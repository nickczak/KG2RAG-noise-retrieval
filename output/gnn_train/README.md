# Matryoshka GNN training artifacts

`hotpot_train.json` is the official HotpotQA distractor training split, downloaded
locally for this experiment. It is newline-delimited JSON and is ignored by Git.
The trained `.pt` checkpoint is also local/ignored; retain its provenance and
experiment outputs when reporting results.

Train with:

```bash
.venv/bin/python code/train_matryoshka_gnn.py \
  --data output/gnn_train/hotpot_train.json \
  --output output/gnn_train/matryoshka_gnn.pt \
  --max-questions 10000 --epochs 3 --device cuda
```

The prototype uses MiniLM sentence vectors, shared-title links for training
graphs, and shared extracted-KG endpoints at inference. This graph-construction
mismatch must be addressed before interpreting a quality change as evidence
that GNN message passing or Matryoshka training helped.
