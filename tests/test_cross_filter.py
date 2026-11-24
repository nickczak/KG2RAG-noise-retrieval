"""Fixed relevance scores test CROSS selection without fetching a model."""
from pathlib import Path
import sys
import unittest

import numpy as np
from sentence_transformers import CrossEncoder
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
from util.kg_post_processor import CrossEncoderFilterPostProcessor


class FixedCrossEncoder(CrossEncoder):
    def __init__(self, scores):
        # Intentionally skip weight loading; this tests the filter, not the model.
        self.scores = np.array(scores)
        self.pairs = None

    def predict(self, pairs):
        self.pairs = pairs
        return self.scores


class CrossFilterTests(unittest.TestCase):
    def select(self, scores, threshold=0.5):
        nodes = [NodeWithScore(node=TextNode(id_=str(i), text=f'text {i}'), score=0.1)
                 for i in range(len(scores))]
        encoder = FixedCrossEncoder(scores)
        processor = CrossEncoderFilterPostProcessor(encoder=encoder, threshold=threshold)
        result = processor.postprocess_nodes(nodes, QueryBundle('question'))
        self.assertEqual(encoder.pairs, [('question', node.node.text) for node in nodes] if nodes else None)
        return nodes, result

    def test_equality_and_original_objects_scores_are_preserved(self):
        nodes, result = self.select([-4, 0.5, 8])
        self.assertEqual([node.node.id_ for node in result], ['1', '2'])
        self.assertIs(result[0], nodes[1])
        self.assertEqual(result[1].score, 0.1)

    def test_fallback_and_single_candidate(self):
        _, result = self.select([-4, -2, -5])
        self.assertEqual(result[0].node.id_, '1')
        _, result = self.select([-4])
        self.assertEqual(result[0].node.id_, '0')

    def test_empty_input(self):
        _, result = self.select([])
        self.assertEqual(result, [])


if __name__ == '__main__':
    unittest.main()
