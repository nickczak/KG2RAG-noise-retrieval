#!/usr/bin/env bash
set -Eeuo pipefail

baseline=validation500_baseline_k5_cap4
path=validation500_path_k5_cap4

# Extraction is incremental: valid paragraph graphs are reused on restart.
./experiment extract "$baseline"
./experiment graphs "$path"

run_smoke() {
  local name="$1"
  local checkpoint="output/$name/smoke/predictions.json.checkpoint.json"
  if [[ -f "$checkpoint" ]]; then
    ./experiment run "$name" --smoke --resume
  else
    ./experiment run "$name" --smoke
  fi
  ./experiment evaluate "$name" --smoke
}

run_full() {
  local name="$1"
  local checkpoint="output/$name/predictions.json.checkpoint.json"
  if [[ -f "$checkpoint" ]]; then
    ./experiment run "$name" --resume
  else
    ./experiment run "$name"
  fi
  ./experiment evaluate "$name"
}

# Check both chains on one item before spending time on all 500 questions.
run_smoke "$baseline"
run_smoke "$path"

# Run the matched baseline and proposed path selector on the same sample/KGs.
run_full "$baseline"
run_full "$path"
./experiment compare "$baseline" "$path"
