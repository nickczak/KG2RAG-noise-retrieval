"""Reuse native retrieval vectors and cache exact BGE question/text scores."""
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sqlite3
import time
from typing import Any

import numpy as np
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.postprocessor.types import BaseNodePostprocessor


def pinned_reranker(model_name):
    """Resolve an already downloaded checkpoint and fingerprint its actual files.

    Passing the resolved directory to FlagReranker keeps tokenizer and weights on
    the same snapshot. No cache is shared merely because two model names match.
    """
    path = Path(model_name)
    if not path.is_dir():
        from huggingface_hub import try_to_load_from_cache
        config = try_to_load_from_cache(model_name, 'config.json')
        if not isinstance(config, str):
            raise ValueError('BGE is not downloaded yet. Run an ordinary baseline smoke test first.')
        path = Path(config).parent
    path = path.absolute()
    files = sorted(file for file in path.rglob('*') if file.is_file())
    if not any(file.suffix in ('.bin', '.safetensors') for file in files):
        raise ValueError(f'No downloaded model weights in {path}; load the reranker once before caching')
    digest = hashlib.sha256()
    for file in files:
        digest.update(str(file.relative_to(path)).encode())
        digest.update(b'\0')
        with file.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
    identity = {
        'checkpoint_sha256': digest.hexdigest(),
        'libraries': {package: version(package) for package in
                      ['FlagEmbedding', 'transformers', 'torch', 'numpy']},
        'cache_schema': 1,
    }
    return str(path), identity


class CachedReranker:
    """Lazy BGE scorer; only unseen exact model/settings/question/text keys run."""

    def __init__(self, database, factory, identity, settings):
        self.factory = factory
        self.model = None
        self.identity = identity
        self.settings = settings
        self.namespace = hashlib.sha256(json.dumps(
            {'identity': identity, 'settings': settings}, sort_keys=True, allow_nan=False).encode()).hexdigest()
        database = Path(database)
        database.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(database, timeout=30)
        self.connection.execute('PRAGMA journal_mode=WAL')
        self.connection.execute('CREATE TABLE IF NOT EXISTS scores '
                                '(namespace TEXT, key TEXT, score REAL NOT NULL, PRIMARY KEY(namespace, key))')
        self.stats = {'requested_pairs': 0, 'cache_hits': 0, 'model_pairs': 0,
                      'batch_duplicates_avoided': 0, 'model_calls': 0, 'model_seconds': 0.0}

    def compute_score(self, sentence_pairs, **kwargs):
        single = len(sentence_pairs) == 2 and all(isinstance(item, str) for item in sentence_pairs)
        pairs = [tuple(sentence_pairs)] if single else [tuple(pair) for pair in sentence_pairs]
        if any(len(pair) != 2 or any(not isinstance(text, str) for text in pair) for pair in pairs):
            raise ValueError('Expected question/text string pairs')
        keys = [hashlib.sha256(json.dumps([pair, kwargs], sort_keys=True, allow_nan=False).encode()).hexdigest()
                for pair in pairs]
        values, missing = {}, {}
        self.stats['requested_pairs'] += len(pairs)
        for key, pair in zip(keys, pairs):
            row = self.connection.execute('SELECT score FROM scores WHERE namespace=? AND key=?',
                                          (self.namespace, key)).fetchone()
            if row is not None:
                if not np.isfinite(row[0]):
                    raise ValueError('Invalid cached BGE score')
                values[key] = row[0]
                self.stats['cache_hits'] += 1
            elif key in missing:
                self.stats['batch_duplicates_avoided'] += 1
            else:
                missing[key] = pair
        if missing:
            if self.model is None:
                self.model = self.factory()
            start = time.perf_counter()
            self.stats['model_calls'] += 1
            self.stats['model_pairs'] += len(missing)
            try:
                result = np.asarray(self.model.compute_score(list(missing.values()), **kwargs),
                                    dtype=float).reshape(-1)
            finally:
                self.stats['model_seconds'] += time.perf_counter() - start
            if len(result) != len(missing) or not np.all(np.isfinite(result)):
                raise ValueError('BGE returned invalid scores; nothing from this batch was cached')
            for key, score in zip(missing, result):
                values[key] = float(score)
            with self.connection:
                self.connection.executemany('INSERT OR REPLACE INTO scores VALUES (?, ?, ?)',
                    [(self.namespace, key, values[key]) for key in missing])
        scores = [values[key] for key in keys]
        return scores[0] if single else scores

    def close(self):
        self.connection.close()


class NativeScoreStore:
    """True cosine scores are separate from propagated graph expansion scores."""

    def __init__(self):
        self.scores = {}
        self.seed_ids = set()
        self.query_embedding = None
        self.seed_scores_reused = 0
        self.expanded_vectors_reused = 0

    def record(self, nodes, query_bundle):
        self.scores.clear()
        self.seed_ids.clear()
        self.expanded_vectors_reused = 0
        self.query_embedding = np.asarray(query_bundle.embedding, dtype=float)
        if self.query_embedding.ndim != 1 or not np.all(np.isfinite(self.query_embedding)):
            raise ValueError('Native score reuse needs the dense query embedding from retrieval')
        for node in nodes:
            if node.score is None or not np.isfinite(node.score):
                raise ValueError('Expected native retrieval similarity for every seed')
            self.scores[node.node.id_] = float(node.score)
            self.seed_ids.add(node.node.id_)
        self.seed_scores_reused = len(self.scores)

    def add_expanded(self, nodes, vector_store):
        query_norm = np.linalg.norm(self.query_embedding)
        if not np.isfinite(query_norm) or query_norm == 0:
            raise ValueError('Invalid query vector for cosine similarity')
        for node in nodes:
            if node.node.id_ in self.scores:
                continue
            vector = np.asarray(vector_store.get(node.node.id_), dtype=float)
            norm = np.linalg.norm(vector)
            if vector.shape != self.query_embedding.shape or not np.all(np.isfinite(vector)) or norm == 0:
                raise ValueError('Missing or invalid stored sentence embedding')
            self.scores[node.node.id_] = float(np.dot(self.query_embedding, vector) / (query_norm * norm))
            self.expanded_vectors_reused += 1


class NativeScoreRetriever(BaseRetriever):
    """Record the original top-k query result; do not change seed/tie selection."""

    def __init__(self, retriever, store):
        super().__init__(callback_manager=getattr(retriever, 'callback_manager', None))
        self.retriever = retriever
        self.store = store

    def _retrieve(self, query_bundle):
        nodes = self.retriever.retrieve(query_bundle)
        self.store.record(nodes, query_bundle)
        return nodes

    async def _aretrieve(self, query_bundle):
        nodes = await self.retriever.aretrieve(query_bundle)
        self.store.record(nodes, query_bundle)
        return nodes


class NativeSemanticScorePostprocessor(BaseNodePostprocessor):
    """Use vectors already in VectorStoreIndex; no additional embedding model."""

    store: Any
    vector_store: Any

    @classmethod
    def class_name(cls):
        return 'NativeSemanticScorePostprocessor'

    def _postprocess_nodes(self, nodes, query_bundle=None):
        if not nodes:
            return []
        self.store.add_expanded(nodes, self.vector_store)
        return nodes
