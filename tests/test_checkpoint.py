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
            top_k=10, use_tpt=False, resume=False,
            method='baseline', sem_model=None, sem_threshold=None)
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


if __name__ == '__main__':
    unittest.main()
