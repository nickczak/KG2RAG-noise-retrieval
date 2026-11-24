"""Checkpoint behavior without model downloads or Ollama calls."""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
import kg_rag_distractor as runner


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.args = argparse.Namespace(
            dataset='hotpotqa', result_path=str(Path(self.directory.name) / 'pred.json'),
            model_name='llama3:8b', embed_model_name='mxbai-embed-large',
            reranker='test', reranker_device='cpu', reranker_fp16=False,
            top_k=10, use_tpt=False, resume=False, temperature=0.0, final_keep_k=None,
            method='baseline', sem_model=None, sem_threshold=None,
            cross_model=None, cross_threshold=None)
        self.data = [{'_id': 'a'}, {'_id': 'b'}]

    def state(self):
        return runner.load_checkpoint(self.args, self.data, {})

    def test_interruption_resume_and_single_model_load(self):
        state = self.state()
        with patch.object(runner, 'init_model'), patch.object(runner, 'FlagReranker') as model:
            with patch.object(runner, 'process_sample', side_effect=[('a', 'answer', []), KeyboardInterrupt]):
                with self.assertRaises(KeyboardInterrupt):
                    runner.kgrag_distractor_predict(self.args, self.data, {}, state)
            model.assert_called_once()
        self.assertEqual(json.loads(Path(self.args.result_path).read_text())['answer'], {'a': 'answer'})
        self.args.resume = True
        with patch.object(runner, 'init_model'), patch.object(runner, 'FlagReranker') as model:
            with patch.object(runner, 'process_sample', return_value=('b', 'second', [])) as process:
                runner.kgrag_distractor_predict(self.args, self.data, {}, self.state())
                self.assertEqual(process.call_args.args[1]['_id'], 'b')
                process.assert_called_once()
            model.assert_called_once()
        with patch.object(runner, 'FlagReranker') as model:
            runner.kgrag_distractor_predict(self.args, self.data, {}, self.state())
            model.assert_not_called()

    def test_failures_are_retried_and_settings_must_match(self):
        with patch.object(runner, 'init_model'), patch.object(runner, 'FlagReranker'):
            with patch.object(runner, 'process_sample', side_effect=[ValueError('offline'), ('b', 'answer', [])]):
                with self.assertRaises(RuntimeError):
                    runner.kgrag_distractor_predict(self.args, self.data, {}, self.state())
        self.args.resume = True
        state = self.state()
        self.assertIn('a', state['errors'])
        self.assertNotIn('a', state['prediction']['answer'])
        with patch.object(runner, 'init_model'), patch.object(runner, 'FlagReranker'):
            with patch.object(runner, 'process_sample', return_value=('a', 'recovered', [])) as process:
                runner.kgrag_distractor_predict(self.args, self.data, {}, state)
                process.assert_called_once()
        self.assertFalse(self.state()['errors'])
        self.args.top_k = 5
        with self.assertRaises(ValueError):
            self.state()

    def test_no_accidental_overwrite_or_unverified_resume(self):
        Path(self.args.result_path).write_text('{}')
        with self.assertRaises(FileExistsError):
            self.state()
        self.args.resume = True
        with self.assertRaises(ValueError):
            self.state()

    def test_changed_data_or_graph_is_rejected(self):
        runner.save_checkpoint(self.args, self.data, self.state())
        self.args.resume = True
        with self.assertRaises(ValueError):
            runner.load_checkpoint(self.args, [{'_id': 'different'}], {})
        with self.assertRaises(ValueError):
            runner.load_checkpoint(self.args, self.data, {'changed': {}})

    def test_sem_settings_must_match_on_resume(self):
        self.args.method = 'sem'
        self.args.sem_model = 'test-encoder'
        self.args.sem_threshold = 0.5
        runner.save_checkpoint(self.args, self.data, self.state())
        self.args.resume = True
        self.state()
        for attribute, value in [('sem_threshold', 0.6),
                                 ('sem_model', 'other-encoder'),
                                 ('method', 'baseline')]:
            with self.subTest(attribute=attribute):
                original = getattr(self.args, attribute)
                setattr(self.args, attribute, value)
                with self.assertRaises(ValueError):
                    self.state()
                setattr(self.args, attribute, original)

    def test_sem_encoder_loaded_once_and_passed_to_each_question(self):
        self.args.method = 'sem'
        self.args.sem_model = 'test-encoder'
        self.args.sem_threshold = 0.5
        with patch.object(runner, 'init_model'), patch.object(runner, 'FlagReranker'):
            with patch.object(runner, 'SentenceTransformer') as encoder:
                with patch.object(runner, 'process_sample',
                                  side_effect=[('a', 'one', []), ('b', 'two', [])]) as process:
                    runner.kgrag_distractor_predict(self.args, self.data, {}, self.state())
                    encoder.assert_called_once_with('test-encoder', device='cpu')
                    self.assertEqual(process.call_count, 2)
                    for call in process.call_args_list:
                        self.assertIs(call.args[4], encoder.return_value)

    def test_cross_encoder_loaded_once_and_settings_guard_resume(self):
        self.args.method = 'cross'
        self.args.cross_model = 'cross-encoder/ms-marco-MiniLM-L6-v2'
        self.args.cross_threshold = 0.5
        with patch.object(runner, 'init_model'), patch.object(runner, 'FlagReranker'):
            with patch.object(runner, 'CrossEncoder') as encoder:
                with patch.object(runner, 'process_sample',
                                  side_effect=[('a', 'one', []), ('b', 'two', [])]) as process:
                    runner.kgrag_distractor_predict(self.args, self.data, {}, self.state())
                    encoder.assert_called_once_with(self.args.cross_model, device='cpu')
                    for call in process.call_args_list:
                        self.assertIs(call.args[4], encoder.return_value)
        self.args.resume = True
        self.state()
        for key, changed in [('cross_model', 'other'), ('cross_threshold', 0.3)]:
            old = getattr(self.args, key)
            setattr(self.args, key, changed)
            with self.assertRaises(ValueError):
                self.state()
            setattr(self.args, key, old)

    def test_combined_loads_each_encoder_once_and_checks_both_settings(self):
        self.args.method = 'sem_cross'
        self.args.sem_model, self.args.sem_threshold = 'sem', 0.3
        self.args.cross_model, self.args.cross_threshold = 'cross', 0.5
        with patch.object(runner, 'init_model'), patch.object(runner, 'FlagReranker'):
            with patch.object(runner, 'SentenceTransformer') as sem, patch.object(runner, 'CrossEncoder') as cross:
                with patch.object(runner, 'process_sample', side_effect=[('a', 'one', []), ('b', 'two', [])]) as process:
                    runner.kgrag_distractor_predict(self.args, self.data, {}, self.state())
                    sem.assert_called_once_with('sem', device='cpu')
                    cross.assert_called_once_with('cross', device='cpu')
                    self.assertEqual(process.call_count, 2)
                    for call in process.call_args_list:
                        self.assertIs(call.args[4], sem.return_value)
                        self.assertIs(call.args[5], cross.return_value)
        self.args.resume = True
        self.state()
        for key, changed in [('sem_model', 'other'), ('sem_threshold', 0.1),
                             ('cross_model', 'other'), ('cross_threshold', 1.0)]:
            old = getattr(self.args, key)
            setattr(self.args, key, changed)
            with self.assertRaises(ValueError):
                self.state()
            setattr(self.args, key, old)

    def test_joint_loads_both_models_and_guards_all_result_affecting_settings(self):
        self.args.method = 'joint'
        self.args.sem_model, self.args.cross_model = 'sem', 'cross'
        self.args.joint_keep_k, self.args.cross_weight = 10, 0.7
        with patch.object(runner, 'init_model'), patch.object(runner, 'FlagReranker'):
            with patch.object(runner, 'SentenceTransformer') as sem, patch.object(runner, 'CrossEncoder') as cross:
                with patch.object(runner, 'process_sample', side_effect=[('a', 'one', []), ('b', 'two', [])]) as process:
                    runner.kgrag_distractor_predict(self.args, self.data, {}, self.state())
                    sem.assert_called_once_with('sem', device='cpu')
                    cross.assert_called_once_with('cross', device='cpu')
                    for call in process.call_args_list:
                        self.assertIs(call.args[4], sem.return_value)
                        self.assertIs(call.args[5], cross.return_value)
        self.args.resume = True
        config = self.state()['config']
        self.assertIsNone(config['sem_threshold'])
        self.assertIsNone(config['cross_threshold'])
        for key, changed in [('sem_model', 'other'), ('cross_model', 'other'),
                             ('joint_keep_k', 5), ('cross_weight', 0.8)]:
            old = getattr(self.args, key)
            setattr(self.args, key, changed)
            with self.assertRaises(ValueError):
                self.state()
            setattr(self.args, key, old)

    def test_native_reuse_uses_only_shared_bge_and_records_audit(self):
        self.args.method = 'reuse'
        self.args.joint_keep_k, self.args.cross_weight = 5, 0.7
        self.args.native_reranker_identity = {'weights': 'one'}
        shared = object()
        def process(args, sample, kg, reranker, encoder, cross_encoder, score_audit,
                    usage_audit, usage_handler):
            self.assertIs(reranker, shared)
            self.assertIsNone(encoder)
            self.assertIsNone(cross_encoder)
            score_audit.update(seed_scores_reused=1)
            return sample['_id'], 'answer', []
        with patch.object(runner, 'init_model'), patch.object(runner, 'FlagReranker') as bge:
            with patch.object(runner, 'SentenceTransformer') as sem, patch.object(runner, 'CrossEncoder') as cross:
                with patch.object(runner, 'process_sample', side_effect=process):
                    state = self.state()
                    runner.kgrag_distractor_predict(self.args, self.data, {}, state, shared)
                sem.assert_not_called()
                cross.assert_not_called()
                bge.assert_not_called()
        self.assertEqual(state['score_reuse']['a']['seed_scores_reused'], 1)
        self.args.resume = True
        self.state()
        self.args.native_reranker_identity = {'weights': 'two'}
        with self.assertRaises(ValueError):
            self.state()


if __name__ == '__main__':
    unittest.main()
