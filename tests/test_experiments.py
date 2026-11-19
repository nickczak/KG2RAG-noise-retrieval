"""Exercise experiment orchestration without downloading or calling models."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import ujson

REPOSITORY = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('experiments', REPOSITORY / 'scripts/experiments.py')
workflow = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workflow)


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.addCleanup(patch.stopall)
        patch.object(workflow, 'ROOT', self.root).start()
        patch.object(workflow, 'OUTPUT', self.root / 'output').start()
        self.data = [{'_id': 'a', 'question': 'Question?', 'answer': 'answer',
                      'supporting_facts': [['Title', 0]],
                      'context': [['Title', ['answer sentence']]]}]
        workflow.write(self.root / 'data.json', self.data)
        workflow.write(self.root / 'kgs/Title.json', {'0': [['a', 'relation', 'b']]})
        target = self.root / 'code/util/hotpot_evaluate_v1.py'
        target.parent.mkdir(parents=True)
        shutil.copyfile(REPOSITORY / 'code/util/hotpot_evaluate_v1.py', target)

    def new(self, name='trial'):
        workflow.new(argparse.Namespace(name=name, method='sem', data='data.json',
                     kgs='kgs', threshold=0.3, top_k=10, model='llama3:8b',
                     embedding='mxbai-embed-large', reranker='BAAI/bge-reranker-large',
                     sem_model='encoder'))
        return self.root / 'output' / name

    def complete(self, name='trial'):
        path = self.new(name)
        config = {key: value for key, value in workflow.settings(name).items()
                  if key not in ('data_path', 'kg_dir')}
        config.update(dataset='hotpotqa', reranker_fp16=False, use_tpt=False,
                      data_hash=hashlib.sha256(ujson.dumps(self.data, sort_keys=True).encode()).hexdigest(),
                      kg_hash='graph', source_hash='source')
        prediction = {'answer': {'a': 'answer'}, 'sp': {'a': [['Title', 0]]}}
        workflow.write(path / 'predictions.json', prediction)
        workflow.write(path / 'predictions.json.checkpoint.json',
                       {'config': config, 'prediction': prediction, 'errors': {}})
        return path

    def test_new_refuses_overwrite_and_paths(self):
        self.new()
        with self.assertRaises(FileExistsError):
            self.new()
        with self.assertRaises(ValueError):
            workflow.folder('../trial')

    def test_prepare_is_reproducible_and_excludes_prompt_examples(self):
        data = self.data + [{**self.data[0], '_id': 'b', 'question': next(iter(workflow.EXAMPLES))}]
        workflow.write(self.root / 'all.json', data)
        workflow.prepare(argparse.Namespace(name='subset', size=1, seed=42, source='all.json'))
        selected = workflow.read(self.root / 'output/subset/hotpot.json')
        self.assertEqual(selected, self.data)

    def test_graph_coverage_and_malformed_triples(self):
        self.new()
        config = workflow.settings('trial')
        self.assertEqual(workflow.graph_coverage(config), [])
        workflow.write(self.root / 'kgs/Title.json', {'0': [['invalid']]})
        with self.assertRaises(ValueError):
            workflow.graph_coverage(config)

    def test_run_builds_sem_command_without_model_calls(self):
        path = self.new()
        with patch.object(workflow, 'record_environment'), patch.object(workflow, 'timed_call') as call:
            workflow.run(argparse.Namespace(name='trial', smoke=True, resume=False))
        command = call.call_args.args[0]
        self.assertEqual(command[command.index('--sem_threshold') + 1], '0.3')
        self.assertEqual(command[command.index('--method') + 1], 'sem')
        self.assertIn(str(path / 'smoke/predictions.json'), command)
        self.assertEqual(workflow.read(path / 'smoke/hotpot_one.json'), self.data)

    def test_incomplete_evaluation_does_not_publish_metrics(self):
        path = self.complete()
        workflow.write(path / 'predictions.json', {'answer': {}, 'sp': {}})
        with self.assertRaises(ValueError):
            workflow.evaluate(argparse.Namespace(name='trial', smoke=False))
        self.assertFalse((path / 'metrics.json').exists())

    def test_evaluation_recomputes_metrics_and_validates_threshold(self):
        path = self.complete()
        workflow.evaluate(argparse.Namespace(name='trial', smoke=False))
        self.assertEqual(workflow.read(path / 'metrics.json')['em'], 1)
        plan = workflow.read(path / 'experiment.json')
        plan['sem_threshold'] = 0.5
        workflow.write(path / 'experiment.json', plan)
        with self.assertRaises(ValueError):
            workflow.completed('trial')

    def test_completed_records_work_without_ignored_checkpoint_after_clone(self):
        path = self.complete()
        state = workflow.read(path / 'predictions.json.checkpoint.json')
        workflow.write(path / 'config.json', state['config'])
        (path / 'predictions.json.checkpoint.json').unlink()
        _, _, metrics, average = workflow.completed('trial')
        self.assertEqual(metrics['f1'], 1)
        self.assertEqual(average, 1)

    def test_wrong_dataset_and_evidence_are_rejected(self):
        path = self.complete()
        workflow.write(self.root / 'data.json', [{**self.data[0], 'answer': 'changed'}])
        with self.assertRaises(ValueError):
            workflow.completed('trial')
        workflow.write(self.root / 'data.json', self.data)
        state = workflow.read(path / 'predictions.json.checkpoint.json')
        state['prediction']['sp']['a'] = [['Title', 99]]
        workflow.write(path / 'predictions.json', state['prediction'])
        workflow.write(path / 'predictions.json.checkpoint.json', state)
        with self.assertRaises(ValueError):
            workflow.completed('trial')

    def test_compare_rejects_different_graph_fingerprints(self):
        self.complete('one')
        path = self.complete('two')
        state = workflow.read(path / 'predictions.json.checkpoint.json')
        state['config']['kg_hash'] = 'different'
        workflow.write(path / 'predictions.json.checkpoint.json', state)
        with self.assertRaises(ValueError):
            workflow.compare(argparse.Namespace(names=['one', 'two']))

    def test_same_output_lock_cannot_be_acquired_twice(self):
        path = self.new()
        with workflow.lock(path):
            with self.assertRaises(ValueError):
                with workflow.lock(path):
                    self.fail('Second writer acquired the same lock')

    def test_environment_capture_saves_source_and_is_not_overwritten(self):
        path = self.new()
        source = self.root / 'code/kg_rag_distractor.py'
        source.write_text('# fixture runner\n')
        response = SimpleNamespace(stdout='fixture output\n', stderr='')
        with patch.object(workflow.subprocess, 'run', return_value=response) as run:
            workflow.record_environment(path, ['python', 'fixture'])
            calls = run.call_count
            workflow.record_environment(path, ['python', 'changed'])
            self.assertEqual(run.call_count, calls)
        self.assertEqual((path / 'source/code/kg_rag_distractor.py').read_text(), source.read_text())
        self.assertEqual(workflow.read(path / 'provenance.json')['command'], ['python', 'fixture'])

    def test_failed_attempt_records_exit_status(self):
        path = self.new()
        with patch.object(workflow.subprocess, 'call', return_value=1):
            with self.assertRaises(ValueError):
                workflow.timed_call(['fixture'], path, 'prediction')
        history = workflow.read(path / 'timing.json')
        self.assertEqual(history[0]['exit_code'], 1)
        self.assertEqual(history[0]['stage'], 'prediction')
        self.assertGreaterEqual(history[0]['seconds'], 0)


if __name__ == '__main__':
    unittest.main()
