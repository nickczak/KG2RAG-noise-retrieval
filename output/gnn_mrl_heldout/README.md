# gnn_mrl_heldout

Trained Matryoshka GNN candidate selector, using the same 50 questions, 7
initial dense seeds, and 7-sentence maximum as `gnn_baseline_k7`.

- Answer EM/F1: 44.00 / 65.23
- Supporting-fact precision/recall/F1: 27.43 / 79.10 / 40.13
- Mean selected sentences: 7.00
- Prediction runtime: 114 seconds

Compared with the matched baseline, answer F1 rises by 1.48 points, while answer
EM falls by 4 points and supporting-fact F1 falls by 1.13 points. The GNN keeps
slightly more sentences and has lower evidence precision. This is a mixed result,
not an overall improvement. The mean candidate graph has 10.14 nodes and 34.8
directed links (density 0.34). Training used title-mention links while inference
used extracted KG endpoints, so the graph message-passing comparison is not yet
controlled. Per-candidate traces are in `gnn_mrl.json`.

1. `./experiment graphs gnn_mrl_heldout`
2. `./experiment run gnn_mrl_heldout --smoke`
3. `./experiment evaluate gnn_mrl_heldout --smoke`
4. `./experiment run gnn_mrl_heldout`
5. `./experiment evaluate gnn_mrl_heldout`

Append `--resume` to a run command after interruption. Read WORKFLOW.md before changing settings or comparing results.
