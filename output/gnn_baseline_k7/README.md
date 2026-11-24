# gnn_baseline_k7

Matched original KG2RAG selector with 7 initial dense seeds and the existing
graph/BGE selector capped at 7 sentences. On the 50-question held-out set:

- Answer EM/F1: 48.00 / 63.75
- Supporting-fact precision/recall/F1: 28.61 / 78.70 / 41.26
- Mean selected sentences: 6.68
- Prediction runtime: 633 seconds

Compare with `gnn_mrl_heldout`. This is the matched control for that experiment.

1. `./experiment graphs gnn_baseline_k7`
2. `./experiment run gnn_baseline_k7 --smoke`
3. `./experiment evaluate gnn_baseline_k7 --smoke`
4. `./experiment run gnn_baseline_k7`
5. `./experiment evaluate gnn_baseline_k7`

Append `--resume` to a run command after interruption. Read WORKFLOW.md before changing settings or comparing results.
