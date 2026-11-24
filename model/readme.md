# Models used by this fork

Install the Python environment using [WORKFLOW.md](../WORKFLOW.md). The default
configuration uses Ollama for generation/initial embeddings and CPU PyTorch for
reranking; these device choices are independent.

```bash
ollama pull llama3:8b
ollama pull mxbai-embed-large
ollama list
```

Keep Ollama serving during extraction and prediction. Use `ollama ps` while a
model is loaded to inspect its CPU/GPU placement.

| Model | Role | Methods |
|---|---|---|
| `llama3:8b` | Triple extraction and answer generation | All |
| `mxbai-embed-large` | Initial RAG embeddings; native similarities | All |
| `BAAI/bge-reranker-large` | Original graph-group relevance scoring | All |
| `sentence-transformers/all-MiniLM-L6-v2` | Additional semantic scorer | `sem`, `sem_cross`, `joint` |
| `cross-encoder/ms-marco-MiniLM-L6-v2` | Additional sentence relevance scorer | `cross`, `sem_cross`, `joint` |

Python model weights download through Hugging Face when first needed and are
cached locally. A baseline run without score caching can download BGE first.
Cache-enabled runs require a complete downloaded checkpoint so its files can be
fingerprinted. `reuse` loads neither auxiliary MiniLM model; its BGE wrapper loads
weights only when an exact score is missing. Downloaded weights are not committed.
