"""Exercise SEM selection with fixed vectors; no model download or server."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
from sentence_transformers import SentenceTransformer
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
from util.kg_post_processor import SemanticFilterPostProcessor


class SemanticFilterTests(unittest.TestCase):
    def setUp(self):
        # Empty module list creates a valid encoder object without fetching weights.
        self.encoder = SentenceTransformer(modules=[], device='cpu')
        self.nodes = [NodeWithScore(node=TextNode(id_=str(i), text=f'sentence {i}'),
                                    score=score)
                      for i, score in enumerate([0.1, 0.9, 0.3])]
        self.processor = SemanticFilterPostProcessor(encoder=self.encoder, threshold=0.5)

    def select(self, scores):
        # Dotting each row with [1, 0] gives the chosen normalized cosine score.
        vectors = np.array([[score, np.sqrt(1 - score ** 2)] for score in scores])
        with patch.object(self.encoder, 'encode', side_effect=[np.array([1., 0.]), vectors]) as encode:
            result = self.processor.postprocess_nodes(self.nodes, QueryBundle('question'))
            self.assertTrue(all(call.kwargs['normalize_embeddings'] for call in encode.call_args_list))
            return result

    def test_threshold_includes_equality_and_preserves_nodes_and_scores(self):
        result = self.select([0.5, 0.2, 0.8])
        self.assertEqual([node.node.id_ for node in result], ['0', '2'])
        self.assertIs(result[0], self.nodes[0])
        self.assertEqual([node.score for node in result], [0.1, 0.3])

    def test_fallback_uses_semantic_score_not_original_retrieval_score(self):
        result = self.select([0.1, 0.2, 0.4])
        self.assertEqual([node.node.id_ for node in result], ['2'])

    def test_empty_input_does_not_call_encoder(self):
        with patch.object(self.encoder, 'encode') as encode:
            self.assertEqual(self.processor.postprocess_nodes([], QueryBundle('question')), [])
            encode.assert_not_called()


if __name__ == '__main__':
    unittest.main()
