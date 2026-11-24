# gnn_mrl_smoke

This run uses a prototype checkpoint trained on 200 HotpotQA train questions
for one epoch. One-question smoke completed successfully. It returned the
correct answer but only 0.5 supporting-fact recall with 7 selected sentences.
This is an integration check, not a quality result. Train the 10,000-question
checkpoint described in `WORKFLOW.md` before running a full held-out comparison.

1. `./experiment graphs gnn_mrl_smoke`
2. `./experiment run gnn_mrl_smoke --smoke`
3. `./experiment evaluate gnn_mrl_smoke --smoke`
4. `./experiment run gnn_mrl_smoke`
5. `./experiment evaluate gnn_mrl_smoke`

Append `--resume` to a run command after interruption. Read WORKFLOW.md before changing settings or comparing results.
