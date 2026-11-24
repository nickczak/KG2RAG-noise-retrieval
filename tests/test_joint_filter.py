"""Joint selection uses both rankings before discarding any sentence."""
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
import kg_rag_distractor as runner
from util.kg_post_processor import select_joint_chunks, _rank_percentiles, JointFilterPostProcessor


class JointSelectionTests(unittest.TestCase):
    def setUp(self):
        self.nodes = [NodeWithScore(node=TextNode(id_=str(i), text=f'text {i}'), score=0.1)
                      for i in range(3)]

    def test_strong_cross_match_is_protected_despite_low_sem_score(self):
        kept = select_joint_chunks(self.nodes, [0.9, 0.8, -0.2], [-5, -2, 8], keep_k=2)
        self.assertEqual([node.node.id_ for node in kept], ['1', '2'])
        self.assertIs(kept[1], self.nodes[2])
        self.assertEqual(kept[1].score, 0.1)

    def test_fusion_changes_other_slots_but_always_reserves_cross_anchor(self):
        low_cross_weight = select_joint_chunks(self.nodes, [3, 2, 1], [1, 2, 3], 2, 0.1)
        high_cross_weight = select_joint_chunks(self.nodes, [3, 2, 1], [1, 2, 3], 2, 0.9)
        self.assertEqual([node.node.id_ for node in low_cross_weight], ['0', '2'])
        self.assertEqual([node.node.id_ for node in high_cross_weight], ['1', '2'])

    def test_raw_score_scale_does_not_change_selection(self):
        first = select_joint_chunks(self.nodes, [0.8, 0.6, 0.1], [-1, 0, 10], 2)
        second = select_joint_chunks(self.nodes, [80, 60, 10], [-100, 0, 1000], 2)
        self.assertEqual([n.node.id_ for n in first], [n.node.id_ for n in second])

    def test_average_ties_empty_single_and_budget(self):
        np.testing.assert_allclose(_rank_percentiles(np.array([8., 8., 1.])), [0.75, 0.75, 0])
        self.assertEqual(select_joint_chunks([], [], []), [])
        self.assertEqual(select_joint_chunks(self.nodes[:1], [0.1], -5, 1), self.nodes[:1])
        self.assertEqual(select_joint_chunks(self.nodes, [0, 0, 0], [0, 0, 0], 1), self.nodes[:1])
        self.assertEqual(select_joint_chunks(self.nodes, [3, 1, 2], [1, 2, 3], 10), self.nodes)

    def test_invalid_scores_and_parameters_fail_explicitly(self):
        for sem, cross, k, weight in [([1], [1, 2, 3], 2, 0.7),
                                     ([1, np.nan, 3], [1, 2, 3], 2, 0.7),
                                     ([1, 2, 3], [1, np.inf, 3], 2, 0.7),
                                     ([1, 2, 3], [1, 2, 3], 0, 0.7),
                                     ([1, 2, 3], [1, 2, 3], 2, 1.1)]:
            with self.assertRaises(ValueError):
                select_joint_chunks(self.nodes, sem, cross, k, weight)

    def test_processor_scores_every_candidate_and_pipeline_uses_one_joint_stage(self):
        sem = SentenceTransformer(modules=[], device='cpu')
        cross = Mock(spec=CrossEncoder)
        cross.predict.return_value = np.array([-5., -2., 8.])
        args = Mock(method='joint', joint_keep_k=2, cross_weight=0.7)
        expansion, graph, presentation = object(), object(), object()
        processors = runner.build_postprocessors(args, expansion, graph, presentation, sem, cross)
        self.assertEqual(len(processors), 4)
        self.assertIsInstance(processors[1], JointFilterPostProcessor)
        self.assertEqual([processors[0], *processors[2:]], [expansion, graph, presentation])
        with patch.object(sem, 'encode', side_effect=[np.array([1., 0.]),
                          np.array([[0.9, 0.4], [0.8, 0.6], [-0.2, 0.98]])]):
            kept = processors[1].postprocess_nodes(self.nodes, QueryBundle('question'))
        cross.predict.assert_called_once_with([('question', n.node.text) for n in self.nodes])
        self.assertEqual([n.node.id_ for n in kept], ['1', '2'])
        cross.predict.reset_mock()
        self.assertEqual(processors[1].postprocess_nodes([], QueryBundle('question')), [])
        cross.predict.assert_not_called()
        with self.assertRaises(ValueError):
            processors[1]._postprocess_nodes(self.nodes)


if __name__ == '__main__':
    unittest.main()
