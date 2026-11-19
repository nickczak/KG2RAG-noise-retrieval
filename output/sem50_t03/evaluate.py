"""Verify completed predictions, save metrics/config, and compare development runs."""
import ast
import contextlib
import importlib.util
import io
import json
from pathlib import Path

out = Path(__file__).resolve().parent
import os
os.chdir(out.parents[1])
data = json.loads(Path('output/baseline50/hotpot_50.json').read_text())
pred = json.loads((out / 'predictions.json').read_text())
state = json.loads((out / 'predictions.json.checkpoint.json').read_text())
assert state['config']['method'] == 'sem'
assert state['config']['sem_threshold'] == 0.3
ids = {sample['_id'] for sample in data}
assert set(pred['answer']) == set(pred['sp']) == ids, 'Incomplete prediction IDs'
assert not state['errors'], 'Resolve failed questions first'
assert pred == state['prediction'], 'Prediction export differs from checkpoint'
spec = importlib.util.spec_from_file_location('hotpot_eval', 'code/util/hotpot_evaluate_v1.py')
evaluator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluator)
stream = io.StringIO()
with contextlib.redirect_stdout(stream):
    evaluator.eval(str(out / 'predictions.json'), 'output/baseline50/hotpot_50.json')
metrics = ast.literal_eval(stream.getvalue().strip())
(out / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
(out / 'config.json').write_text(json.dumps(state['config'], indent=2) + '\n')
print(json.dumps(metrics, indent=2))
print('\nComparison (percent; evidence count is sentences):')
for label, folder in [('Baseline', 'baseline50'), ('SEM 0.5', 'sem50'), ('SEM 0.3', 'sem50_t03')]:
    directory = Path('output') / folder
    values = json.loads((directory / 'metrics.json').read_text())
    prediction = json.loads((directory / 'predictions.json').read_text())
    average = sum(map(len, prediction['sp'].values())) / len(data)
    print(f"{label}: answer F1={100*values['f1']:.2f}, "
          f"support precision={100*values['sp_prec']:.2f}, "
          f"support recall={100*values['sp_recall']:.2f}, "
          f"support F1={100*values['sp_f1']:.2f}, sentences={average:.2f}")
