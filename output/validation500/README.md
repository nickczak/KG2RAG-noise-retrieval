# 500-question tuning sample

`hotpot.json` contains 500 questions sampled from the HotpotQA distractor
development set with seed `20261121`. `selection.json` records the sampled IDs
and source. The sample excludes IDs used by the earlier `baseline50`,
`heldout50`, and smoke runs, plus the runner's prompt examples.

The baseline and path experiments use the same five dense seeds and four-sentence
evidence cap:

- `../validation500_baseline_k5_cap4/`: original KG2RAG graph/BGE selector,
  capped at four final evidence sentences.
- `../validation500_path_k5_cap4/`: path selector, rank fusion weight 0.7,
  capped at four final evidence sentences.

This folder's `validation500` name is retained for compatibility with the active
extraction and run commands, but this sample is now the **tuning/development set**.
Use its results to choose parameters only. The project's final report must use a
separate 500-question held-out sample, selected after tuning and not inspected
during method development.

Both experiments use the same questions and shared `output/validation500/kgs`
directory. Successful answer runs write `usage.json` with per-question Ollama
prompt/completion/total tokens, query latency, full sample pipeline latency, and
Ollama-reported generation time. Extraction token use is not included.
Graph extraction saves complete paragraph files as it proceeds; rerun the
extract command after interruption and it will reuse valid files. Prediction
runs checkpoint per question; resume an interrupted prediction with `--resume`.
There are 4,938 unique paragraph titles and about 20,418 unique paragraph
sentences, so graph extraction may take longer than one night. The sequence is
restartable and runs smoke checks before processing all 500 questions.

Start the full sequence in the background and follow its log:

```bash
nohup bash scripts/run_validation500.sh \
  > output/validation500/overnight.log 2>&1 < /dev/null &
tail -f output/validation500/overnight.log
```

Or run the stages in the foreground:

```bash
./experiment extract validation500_baseline_k5_cap4
./experiment graphs validation500_path_k5_cap4

./experiment run validation500_baseline_k5_cap4 --smoke
./experiment evaluate validation500_baseline_k5_cap4 --smoke
./experiment run validation500_path_k5_cap4 --smoke
./experiment evaluate validation500_path_k5_cap4 --smoke

./experiment run validation500_baseline_k5_cap4
./experiment evaluate validation500_baseline_k5_cap4
./experiment run validation500_path_k5_cap4
./experiment evaluate validation500_path_k5_cap4
./experiment compare validation500_baseline_k5_cap4 validation500_path_k5_cap4
```

Keep the machine awake and Ollama available while commands run. Extraction and
two 500-question answer runs may take longer than one night. Progress is saved;
see `WORKFLOW.md` for recovery commands and run status.
