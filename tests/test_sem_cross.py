"""Verify the actual SEM → CROSS cascade without weights or Ollama."""
import argparse
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
import kg_rag_distractor as runner
from util.kg_post_processor import SemanticFilterPostProcessor, CrossEncoderFilterPostProcessor


class CombinedFilterTests(unittest.TestCase):
    def setUp(self):
        self.args = argparse.Namespace(method='sem_cross', sem_threshold=0.3, cross_threshold=0.5)
        self.sem = SentenceTransformer(modules=[], device='cpu')
        self.cross = Mock(spec=CrossEncoder)
        self.sentinels = [object(), object(), object()]

    def processors(self):
        return runner.build_postprocessors(self.args, *self.sentinels, self.sem, self.cross)

    def test_pipeline_order_and_cross_only_scores_sem_survivors(self):
        nodes = [NodeWithScore(node=TextNode(id_=str(i), text=f'sentence {i}'), score=0.1)
                 for i in range(4)]
        processors = self.processors()
        self.assertIs(processors[0], self.sentinels[0])
        self.assertIsInstance(processors[1], SemanticFilterPostProcessor)
        self.assertIsInstance(processors[2], CrossEncoderFilterPostProcessor)
        self.assertEqual(processors[3:], self.sentinels[1:])
        scores = np.array([0.1, 0.8, 0.4, 0.9])
        vectors = np.column_stack([scores, np.sqrt(1 - scores ** 2)])
        self.cross.predict.return_value = np.array([-2., 2., 3.])
        with patch.object(self.sem, 'encode', side_effect=[np.array([1., 0.]), vectors]):
            kept = processors[1].postprocess_nodes(nodes, QueryBundle('question'))
        kept = processors[2].postprocess_nodes(kept, QueryBundle('question'))
        self.cross.predict.assert_called_once_with([
            ('question', 'sentence 1'), ('question', 'sentence 2'), ('question', 'sentence 3')])
        self.assertEqual([node.node.id_ for node in kept], ['2', '3'])
        self.assertIs(kept[0], nodes[2])
        self.assertEqual(kept[0].score, 0.1)

    def test_both_fallbacks_keep_one_node_and_never_restore_sem_rejects(self):
        nodes = [NodeWithScore(node=TextNode(id_=str(i), text=f'sentence {i}'), score=0.1)
                 for i in range(2)]
        processors = self.processors()
        self.cross.predict.return_value = np.array([-4.])
        with patch.object(self.sem, 'encode', side_effect=[np.array([1., 0.]), np.array([[0., 1.], [0.2, 0.98]])]):
            kept = processors[1].postprocess_nodes(nodes, QueryBundle('question'))
        kept = processors[2].postprocess_nodes(kept, QueryBundle('question'))
        self.assertEqual([node.node.id_ for node in kept], ['1'])
        self.cross.predict.assert_called_once_with([('question', 'sentence 1')])

    def test_baseline_has_no_added_filters(self):
        self.args.method = 'baseline'
        self.assertEqual(self.processors(), self.sentinels)


if __name__ == '__main__':
    unittest.main()
