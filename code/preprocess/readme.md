# Data Preprocessing

This folder contains scripts for preprocessing three datasets: HotpotQA, MuSiQue, and TriviaQA. Each script is used to extract knowledge graphs (KGs) from the datasets. Below are the details of each script and how they are used:

## HotpotQA Dataset

For named experiments, use `./experiment graphs RUN_NAME` to check the shared
cache and `./experiment extract RUN_NAME` to generate missing graphs. See
[WORKFLOW.md](../../WORKFLOW.md).

Or run directly from the repository root:

```bash
.venv/bin/python -u code/preprocess/hotpot_extraction.py \
  --data_path output/baseline50/hotpot_50.json \
  --kg_dir output/baseline50/kgs \
  --model_name llama3:8b
```

This script extracts triplets from the HotpotQA dataset. The main steps are as follows:
1. Load the dataset file `hotpot_dev_distractor_v1.json`.
2. Use the `llama3:8b` model to extract triplets sentence by sentence.
3. Save the extracted triplets to the specified output directory.

The CLI is import-safe, creates directories, shows sentence progress, checks
filename collisions, and atomically writes each paragraph. Rerunning skips saved
paragraph JSON files. Triple extraction prompts and parsing retain the original
behavior. Generation is expensive; reuse these graphs across filtering variants.



## MuSiQue Dataset

`> python musique_extraction.py`

This script extracts triplets from the MuSiQue dataset. The main steps are as follows:
1. Load the dataset file `musique_ans_v1.0_dev.jsonl`.
2. Use the `llama3:8b` model to extract triplets from each paragraph.
3. Save the extracted triplets to the specified output directory.

## TriviaQA Dataset

`> python trivia_extraction.py`

This script extracts triplets from the TriviaQA dataset. The main steps are as follows:
1. Load the dataset file `trivia.json`.
2. Use the `llama3:8b` model to extract triplets from each context paragraph.
3. Save the extracted triplets to the specified output directory.
