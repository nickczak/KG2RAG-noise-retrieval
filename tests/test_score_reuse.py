"""Native semantic/BGE reuse must not invent or transfer scores across inputs."""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from llama_index.core import VectorStoreIndex
from llama_index.core.embeddings import MockEmbedding
from llama_index.core.retrievers import VectorIndexRetriever

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
from util.score_reuse import (CachedReranker, NativeScoreStore, NativeScoreRetriever,
                             NativeSemanticScorePostprocessor, pinned_reranker)
from util.kg_post_processor import GraphFilterPostProcessor
import kg_rag_distractor as runner


class CacheTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def cache(self, factory, identity=None, settings=None):
        cache = CachedReranker(self.root / 'scores.sqlite', factory,
                                identity or {'weights': 'one'}, settings or {'fp16': False})
        self.addCleanup(cache.close)
        return cache

    def test_duplicate_inputs_order_and_persistent_hits_do_not_reload_model(self):
        model = Mock()
        model.compute_score.return_value = [1., 2.]
        factory = Mock(return_value=model)
        cache = self.cache(factory)
        self.assertEqual(cache.compute_score([('q', 'a'), ('q', 'b'), ('q', 'a')]), [1, 2, 1])
        model.compute_score.assert_called_once_with([('q', 'a'), ('q', 'b')])
        self.assertEqual(cache.stats['model_pairs'], 2)
        self.assertEqual(cache.stats['batch_duplicates_avoided'], 1)
        unused_factory = Mock(side_effect=AssertionError('Cache hit must not load model'))
        warm = self.cache(unused_factory)
        self.assertEqual(warm.compute_score([('q', 'b'), ('q', 'a')]), [2, 1])
        self.assertEqual(warm.stats['cache_hits'], 2)
        self.assertEqual(warm.stats['model_calls'], 0)
        unused_factory.assert_not_called()

    def test_changed_model_settings_question_text_and_call_options_miss(self):
        model = Mock()
        model.compute_score.side_effect = lambda pairs, **kwargs: [3.] * len(pairs)
        cache = self.cache(lambda: model)
        cache.compute_score([('q', 'text')])
        cache.compute_score([('q2', 'text'), ('q', 'text ')])
        cache.compute_score([('q', 'text')], normalize=True)
        changed_model = self.cache(lambda: model, identity={'weights': 'two'})
        changed_model.compute_score([('q', 'text')])
        changed_settings = self.cache(lambda: model, settings={'fp16': True})
        changed_settings.compute_score([('q', 'text')])
        self.assertEqual(model.compute_score.call_count, 5)

    def test_single_pair_empty_and_nonfinite_scores(self):
        model = Mock()
        model.compute_score.return_value = 4.
        cache = self.cache(lambda: model)
        self.assertEqual(cache.compute_score([]), [])
        self.assertIsNone(cache.model)
        self.assertEqual(cache.compute_score(('q', 'a')), 4.)
        self.assertEqual(cache.compute_score([('q', 'a')]), [4.])
        model.compute_score.return_value = [np.nan]
        with self.assertRaises(ValueError):
            cache.compute_score([('q', 'bad')])
        model.compute_score.return_value = [8.]
        self.assertEqual(cache.compute_score([('q', 'bad')]), [8.])

    def test_checkpoint_fingerprint_changes_when_weights_change(self):
        model = self.root / 'model'
        model.mkdir()
        (model / 'config.json').write_text('{}')
        weight = model / 'model.safetensors'
        weight.write_bytes(b'first')
        with patch('util.score_reuse.version', return_value='test'):
            path, first = pinned_reranker(str(model))
            weight.write_bytes(b'second')
            _, second = pinned_reranker(str(model))
        self.assertEqual(Path(path), model)
        self.assertNotEqual(first['checkpoint_sha256'], second['checkpoint_sha256'])


class NativeSemanticTests(unittest.TestCase):
    def test_real_index_reuses_query_and_text_embeddings_without_new_forward_calls(self):
        class CountingEmbedding(MockEmbedding):
            query_calls: int = 0
            text_calls: int = 0

            def _get_query_embedding(self, query):
                self.query_calls += 1
                return super()._get_query_embedding(query)

            def _get_text_embedding(self, text):
                self.text_calls += 1
                return super()._get_text_embedding(text)

        embedding = CountingEmbedding(embed_dim=2)
        texts = [TextNode(id_='a', text='one'), TextNode(id_='b', text='two')]
        index = VectorStoreIndex(texts, embed_model=embedding)
        store = NativeScoreStore()
        retriever = NativeScoreRetriever(VectorIndexRetriever(index, similarity_top_k=1), store)
        bundle = QueryBundle('question')
        seed = retriever.retrieve(bundle)
        other = next(node for node in texts if node.id_ != seed[0].node.id_)
        expanded = seed + [NodeWithScore(node=other, score=123.)]
        NativeSemanticScorePostprocessor(store=store, vector_store=index.vector_store).postprocess_nodes(expanded, bundle)
        self.assertEqual(embedding.query_calls, 1)
        self.assertEqual(embedding.text_calls, 2)
        self.assertEqual(store.expanded_vectors_reused, 1)
        self.assertAlmostEqual(store.scores[other.id_], 1.0)
        self.assertEqual(expanded[1].score, 123.)

    def test_seed_scores_and_query_vector_reused_expansion_heuristic_is_not_cosine(self):
        seed = NodeWithScore(node=TextNode(id_='seed', text='seed'), score=0.8)
        extra = NodeWithScore(node=TextNode(id_='extra', text='extra'), score=123.)
        base = Mock(callback_manager=None)
        def retrieve(bundle):
            bundle.embedding = [1., 0.]
            return [seed]
        base.retrieve.side_effect = retrieve
        store = NativeScoreStore()
        wrapped = NativeScoreRetriever(base, store)
        bundle = QueryBundle('question')
        self.assertEqual(wrapped.retrieve(bundle), [seed])
        base.retrieve.assert_called_once_with(bundle)
        vector_store = Mock()
        vector_store.get.return_value = [0., 1.]
        processor = NativeSemanticScorePostprocessor(store=store, vector_store=vector_store)
        self.assertIs(processor.store, store)
        self.assertEqual(processor.postprocess_nodes([seed, extra], bundle), [seed, extra])
        self.assertEqual(store.scores, {'seed': 0.8, 'extra': 0.0})
        self.assertEqual(extra.score, 123.)
        vector_store.get.assert_called_once_with('extra')
        processor.postprocess_nodes([seed, extra], bundle)
        vector_store.get.assert_called_once()
        self.assertEqual(store.expanded_vectors_reused, 1)
        self.assertEqual(store.seed_scores_reused, 1)


class NativeGroupTests(unittest.TestCase):
    def setUp(self):
        self.nodes = [NodeWithScore(node=TextNode(id_=name, text=name), score=0.1)
                      for name in ['A', 'B', 'C']]
        self.store = NativeScoreStore()
        self.store.scores = {'A': 0.1, 'B': 0.2, 'C': 0.9}

    def processor(self, reranker, topk=2):
        return GraphFilterPostProcessor(dataset='hotpotqa', topk=topk, use_tpt=False,
                   ents=set(), doc2kg={}, chunks_index={}, reranker=reranker,
                   native_scores=self.store, cross_weight=0.7)

    def test_groups_are_selected_whole_and_not_scored_again_as_sentences(self):
        model = Mock()
        model.compute_score.return_value = [8., 3.]
        processor = self.processor(model)
        kept = processor._select_native(self.nodes, QueryBundle('q'), [['A', 'B', 'A'], ['C']], ['ABA', 'C'])
        model.compute_score.assert_called_once_with([('q', 'ABA'), ('q', 'C')])
        self.assertEqual(kept, self.nodes[:2])
        self.assertAlmostEqual(processor.reuse_audit['group_scores'][0]['semantic_mean'], 0.15)
        self.assertEqual(processor.reuse_audit['selection_unit'], 'complete_groups')
        self.assertEqual(processor.reuse_audit['sentence_scores'], [])

    def test_oversized_group_falls_back_to_genuine_individual_scores(self):
        model = Mock()
        model.compute_score.side_effect = [[100.], [1., 5., 3.]]
        processor = self.processor(model)
        kept = processor._select_native(self.nodes, QueryBundle('q'), [['A', 'B', 'C']], ['ABC'])
        self.assertEqual(kept, self.nodes[1:])
        self.assertEqual(model.compute_score.call_count, 2)
        self.assertEqual(processor.reuse_audit['selection_unit'], 'sentence_fallback')
        self.assertEqual([row['bge_score'] for row in processor.reuse_audit['sentence_scores']], [1, 5, 3])

    def test_missing_semantic_scores_are_not_replaced_with_graph_weights(self):
        processor = self.processor(Mock(compute_score=Mock(return_value=[3.])))
        self.store.scores.clear()
        with self.assertRaises(ValueError):
            processor._select_native(self.nodes, QueryBundle('q'), [['A']], ['A'])

    def test_pipeline_has_no_second_semantic_or_cross_model(self):
        args = argparse.Namespace(method='reuse')
        markers = [object(), object(), object()]
        processors = runner.build_postprocessors(args, *markers, native_scores=self.store, vector_store=Mock())
        self.assertEqual(len(processors), 4)
        self.assertIsInstance(processors[1], NativeSemanticScorePostprocessor)
        self.assertEqual([processors[0], *processors[2:]], markers)

    def test_original_graph_filter_cached_and_uncached_results_match(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Mock()
            model.compute_score.return_value = [1., 2., 3.]
            cache = CachedReranker(Path(directory) / 'scores.sqlite', lambda: model, {'weights': 'one'}, {})
            try:
                nodes = [NodeWithScore(node=TextNode(id_=f'Title##{i}', text=str(i)), score=0.1)
                         for i in range(3)]
                kwargs = dict(dataset='hotpotqa', topk=2, use_tpt=False,
                              ents={'Title'}, doc2kg={}, chunks_index={})
                plain = GraphFilterPostProcessor(**kwargs, reranker=model).postprocess_nodes(nodes, QueryBundle('q'))
                first = GraphFilterPostProcessor(**kwargs, reranker=cache).postprocess_nodes(nodes, QueryBundle('q'))
                second = GraphFilterPostProcessor(**kwargs, reranker=cache).postprocess_nodes(nodes, QueryBundle('q'))
                self.assertEqual(plain, first)
                self.assertEqual(first, second)
                self.assertEqual(cache.stats['model_calls'], 1)
                self.assertEqual(cache.stats['cache_hits'], 3)
            finally:
                cache.close()


if __name__ == '__main__':
    unittest.main()
