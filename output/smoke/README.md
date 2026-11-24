# Original one-question workflow check

This folder is a historical smoke record. For current setup, graph extraction,
new runs, and evaluation use [WORKFLOW.md](../../WORKFLOW.md). Temporary CPU runner
copies and source-string path replacement have been replaced by the runner and
extractor CLIs. Generated smoke data/graphs/results are local and ignored.

The original question asks which government position Shirley Temple held.
The gold answer was `Chief of Protocol`; the model returned `Diplomat.`. All
three annotated supporting facts were selected among ten sentences: evidence
precision 0.30, recall 1.00, and F1 about 0.462. This verifies execution, not
answer accuracy. This question also appears in the runner's example prompt;
the fixed 50-question development set excludes it and the other prompt examples.

The environment was repaired with NumPy 1.26.4 and PyArrow 25.0.0; current direct
dependencies live in root `requirements.txt`. Ollama uses GPU resources separately
from CPU PyTorch/BGE. Inspect loaded placement with `ollama ps`.

Current one-question checks use a new named experiment and isolated smoke folder:

```bash
./experiment new baseline_smoke_next --method baseline
./experiment graphs baseline_smoke_next
./experiment run baseline_smoke_next --smoke
./experiment evaluate baseline_smoke_next --smoke
```

The workflow selects its first development question, records provenance, verifies
graph coverage, checkpoints predictions, and validates results before evaluation.
Use `./experiment extract baseline_smoke_next` only if graphs are missing. Reuse
valid graph caches between selection methods; use a new experiment name after
source/settings changes. The completed 50-question comparisons and prepared
native-score experiments are described in [../README.md](../README.md).
