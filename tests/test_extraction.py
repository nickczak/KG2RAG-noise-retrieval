"""Test extraction caching/interruptions with a mocked LLM."""
import argparse
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('hotpot_extraction', ROOT / 'code/preprocess/hotpot_extraction.py')
extraction = importlib.util.module_from_spec(spec)
spec.loader.exec_module(extraction)


class ExtractionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.data = [{'context': [['Title', ['first', 'second']]]}]
        self.data_path = self.root / 'data.json'
        self.data_path.write_text(json.dumps(self.data))
        self.args = argparse.Namespace(data_path=str(self.data_path),
                     kg_dir=str(self.root / 'kgs'), model_name='test', timeout=120)

    def test_saved_paragraph_is_reused_without_extraction(self):
        with patch.object(extraction, 'Ollama'), patch.object(extraction, 'extract_triplets',
                 return_value=[['a', 'rel', 'b']]) as extract:
            extraction.main(self.args)
            self.assertEqual(extract.call_count, 2)
            self.assertEqual(extract.call_args_list[1].args[1], 'Title: second')
            extraction.main(self.args)
            self.assertEqual(extract.call_count, 2)
        saved = json.loads((self.root / 'kgs/Title.json').read_text())
        self.assertEqual(set(saved), {'0', '1'})

    def test_interruption_does_not_publish_partial_paragraph(self):
        with patch.object(extraction, 'Ollama'), patch.object(extraction, 'extract_triplets',
                 side_effect=[[['a', 'rel', 'b']], KeyboardInterrupt]):
            with self.assertRaises(KeyboardInterrupt):
                extraction.main(self.args)
        self.assertFalse((self.root / 'kgs/Title.json').exists())

    def test_filename_collision_is_rejected_before_model_loading(self):
        self.data_path.write_text(json.dumps([{'context': [['A/B', ['first']], ['A_B', ['second']]]}]))
        with patch.object(extraction, 'Ollama') as model:
            with self.assertRaises(ValueError):
                extraction.main(self.args)
            model.assert_not_called()


if __name__ == '__main__':
    unittest.main()
