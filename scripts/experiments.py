"""Named HotpotQA experiments. Model calls happen only in run/extract commands."""
import argparse
import ast
import contextlib
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import random
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'output'
EXAMPLES = {
    'Were Scott Derrickson and Ed Wood of the same nationality?',
    'Who was born earlier, Emma Bull or Virginia Woolf?',
    'The arena where the Lewiston Maineiacs played their home games can seat how many people?',
    'What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?',
}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def folder(name):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', name):
        raise ValueError('Use a run name containing only letters, digits, underscores, or hyphens')
    path = OUTPUT / name
    if path.is_symlink():
        raise ValueError('Experiment folders must not be symbolic links')
    return path


def absolute(path):
    return ROOT / path


def load_data(path):
    data = read(path)
    if not isinstance(data, list) or not data:
        raise ValueError('Expected a nonempty HotpotQA JSON array')
    ids = [sample['_id'] for sample in data]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate question IDs')
    return data


def settings(name):
    path = folder(name)
    if not (path / 'experiment.json').exists():
        raise ValueError(f'{name} is a historical/manual run. Use new with a fresh name before running code.')
    config = read(path / 'experiment.json')
    if config['method'] not in ('baseline', 'sem', 'cross', 'sem_cross', 'joint', 'reuse', 'path_reuse'):
        raise ValueError('Unsupported method; implement and test it before adding it here')
    if config['method'] in ('sem', 'sem_cross') and (not math.isfinite(config['sem_threshold']) or not -1 <= config['sem_threshold'] <= 1):
        raise ValueError('SEM threshold must be between -1 and 1')
    if config['method'] in ('cross', 'sem_cross') and not math.isfinite(config['cross_threshold']):
        raise ValueError('CROSS threshold must be a finite relevance score')
    if config['top_k'] < 1:
        raise ValueError('top_k must be positive')
    if config['method'] == 'baseline' and config.get('final_keep_k') is not None:
        if type(config['final_keep_k']) is not int or config['final_keep_k'] < 1:
            raise ValueError('final_keep_k must be a positive integer')
    if config['method'] in ('joint', 'reuse', 'path_reuse'):
        if type(config['joint_keep_k']) is not int or config['joint_keep_k'] < 1:
            raise ValueError('joint_keep_k must be a positive integer')
        if not math.isfinite(config['cross_weight']) or not 0 <= config['cross_weight'] <= 1:
            raise ValueError('cross_weight must be finite and between 0 and 1')
    return config


def prepare(args):
    path = folder(args.name)
    if path.exists():
        raise FileExistsError(f'{path} already exists; choose a new name')
    if args.size < 1:
        raise ValueError('size must be positive')
    data = load_data(absolute(args.source))
    eligible = [sample for sample in data if sample['question'].strip() not in EXAMPLES]
    selected = random.Random(args.seed).sample(eligible, args.size)
    write(path / 'hotpot.json', selected)
    write(path / 'selection.json', {
        'source': args.source, 'seed': args.seed, 'size': args.size,
        'excluded_questions': sorted(EXAMPLES),
        'question_ids': [sample['_id'] for sample in selected],
    })
    print(f'Saved {len(selected)} questions to {path / "hotpot.json"}')
    print('Use this dataset path with new --data; reserve other IDs for held-out evaluation.')


def new(args):
    path = folder(args.name)
    if path.exists():
        raise FileExistsError('Choose a new name; existing runs are never overwritten')
    load_data(absolute(args.data))
    if args.method in ('joint', 'reuse', 'path_reuse') and any(value is not None for value in
            [args.threshold, getattr(args, 'sem_threshold', None), getattr(args, 'cross_threshold', None)]):
        raise ValueError('This selector uses --joint-keep-k, not score cutoffs')
    joint_keep_k = getattr(args, 'joint_keep_k', 10)
    cross_weight = getattr(args, 'cross_weight', 0.7)
    if type(joint_keep_k) is not int or joint_keep_k < 1 or not math.isfinite(cross_weight) or not 0 <= cross_weight <= 1:
        raise ValueError('Require joint_keep_k > 0 and cross_weight between 0 and 1')
    if args.method == 'sem_cross' and args.threshold is not None:
        raise ValueError('For sem_cross use --sem-threshold and --cross-threshold, not --threshold')
    threshold = args.threshold if args.threshold is not None else 0.5
    sem_threshold = getattr(args, 'sem_threshold', None)
    cross_threshold = getattr(args, 'cross_threshold', None)
    if sem_threshold is None:
        sem_threshold = 0.3 if args.method == 'sem_cross' else threshold
    if cross_threshold is None:
        cross_threshold = threshold
    if args.top_k < 1 or not all(map(math.isfinite, [sem_threshold, cross_threshold])):
        raise ValueError('Require top_k > 0 and finite thresholds')
    final_keep_k = getattr(args, 'final_keep_k', None)
    if final_keep_k is not None and (args.method != 'baseline' or type(final_keep_k) is not int or final_keep_k < 1):
        raise ValueError('--final-keep-k is a positive baseline-only evidence cap')
    if not math.isfinite(args.temperature) or args.temperature < 0:
        raise ValueError('temperature must be finite and nonnegative')
    if args.method in ('sem', 'sem_cross') and not -1 <= sem_threshold <= 1:
        raise ValueError('SEM threshold must be between -1 and 1')
    write(path / 'experiment.json', {
        'method': args.method, 'data_path': args.data, 'kg_dir': args.kgs,
        'model_name': args.model, 'embed_model_name': args.embedding,
        'temperature': args.temperature,
        'reranker': args.reranker, 'reranker_device': 'cpu',
        'sem_model': None if args.method in ('reuse', 'path_reuse') else args.sem_model,
        'sem_threshold': None if args.method in ('joint', 'reuse', 'path_reuse') else sem_threshold,
        'cross_model': None if args.method in ('reuse', 'path_reuse') else getattr(args, 'cross_model', 'cross-encoder/ms-marco-MiniLM-L6-v2'),
        'cross_threshold': None if args.method in ('joint', 'reuse', 'path_reuse') else cross_threshold,
        'joint_keep_k': joint_keep_k, 'cross_weight': cross_weight,
        'final_keep_k': final_keep_k if args.method == 'baseline' else None,
        'score_cache_dir': (getattr(args, 'score_cache', None) or 'output/score_cache')
                           if args.method in ('reuse', 'path_reuse') else getattr(args, 'score_cache', None),
        'top_k': args.top_k,
    })
    (path / 'README.md').write_text(
        f'# {args.name}\n\nPrepared experiment; no results yet.\n\n'
        f'1. `./experiment graphs {args.name}`\n'
        f'2. `./experiment run {args.name} --smoke`\n'
        f'3. `./experiment evaluate {args.name} --smoke`\n'
        f'4. `./experiment run {args.name}`\n'
        f'5. `./experiment evaluate {args.name}`\n\n'
        'Append `--resume` to a run command after interruption. '
        'Read WORKFLOW.md before changing settings or comparing results.\n')
    print(f'Created {path}. Next: ./experiment graphs {args.name}')


def graph_coverage(config):
    data = load_data(absolute(config['data_path']))
    titles = sorted({title for sample in data for title, _ in sample['context']})
    filenames = [title.replace('/', '_') + '.json' for title in titles]
    if len(set(filenames)) != len(filenames):
        raise ValueError('Distinct paragraph titles collide on graph filenames')
    missing, empty = [], []
    for title, filename in zip(titles, filenames):
        path = absolute(config['kg_dir']) / filename
        if not path.exists():
            missing.append(title)
            continue
        graph = read(path)
        if not isinstance(graph, dict):
            raise ValueError(f'Expected a sentence-indexed graph: {path}')
        for triples in graph.values():
            if not isinstance(triples, list) or any(
                not isinstance(triple, list) or len(triple) != 3 or
                any(not isinstance(part, str) for part in triple) for triple in triples
            ):
                raise ValueError(f'Malformed triples: {path}')
        if not any(graph.values()):
            empty.append(title)
    print(f'Graph files: {len(titles)-len(missing)}/{len(titles)}; empty: {len(empty)}')
    return missing


def graphs(args):
    missing = graph_coverage(settings(args.name))
    if missing:
        raise ValueError(f'{len(missing)} graphs missing. Run: ./experiment extract {args.name}')


@contextlib.contextmanager
def lock(path):
    path.mkdir(parents=True, exist_ok=True)
    with (path / '.run.lock').open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError(f'Another process is using {path}') from None
        yield


def timed_call(command, path, stage):
    started = datetime.now(timezone.utc).isoformat()
    clock = time.perf_counter()
    code = 130
    try:
        code = subprocess.call(command, cwd=ROOT)
    finally:
        file = path / 'timing.json'
        history = read(file) if file.exists() else []
        history.append({'stage': stage, 'started_utc': started,
                        'seconds': time.perf_counter() - clock, 'exit_code': code})
        write(file, history)
    if code:
        raise ValueError(f'{stage} exited with status {code}; inspect output before resuming')


def extract(args):
    config = settings(args.name)
    # Shared graph cache needs its own lock, independent of the run directory.
    path = folder(args.name)
    with lock(absolute(config['kg_dir'])):
        timed_call([sys.executable, '-u', 'code/preprocess/hotpot_extraction.py',
                    '--data_path', str(absolute(config['data_path'])),
                    '--kg_dir', str(absolute(config['kg_dir'])),
                    '--model_name', config['model_name']], path, 'extraction')
    graphs(args)


def record_environment(path, command):
    # Save once at first launch; snapshot source even if it is not committed yet.
    if (path / 'provenance.json').exists():
        return
    commands = {'environment.txt': [sys.executable, '-m', 'pip', 'freeze', '--local'],
                'commit.txt': ['git', 'rev-parse', 'HEAD'],
                'git-status.txt': ['git', 'status', '--short']}
    for name, cmd in commands.items():
        result = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=True)
        (path / name).write_text(result.stdout)
    for name, cmd in [('ollama-version.txt', ['ollama', '--version']),
                      ('ollama-models.txt', ['ollama', 'list'])]:
        result = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, timeout=30, check=True)
        (path / name).write_text('\n'.join(line.rstrip() for line in
                                          (result.stdout + result.stderr).splitlines()) + '\n')
    hashes = {}
    for source in [ROOT / 'code/kg_rag_distractor.py', *sorted((ROOT / 'code/util').glob('*.py'))]:
        relative = source.relative_to(ROOT)
        snapshot = path / 'source' / relative
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(source.read_bytes())
        hashes[str(relative)] = hashlib.sha256(source.read_bytes()).hexdigest()
    write(path / 'provenance.json', {'python': sys.executable, 'command': command,
                                    'created_utc': datetime.now(timezone.utc).isoformat(),
                                    'source_sha256': hashes})


def run(args):
    config = settings(args.name)
    directory = folder(args.name)
    path = directory / 'smoke' if args.smoke else directory
    with lock(path):
        data_path = absolute(config['data_path'])
        if args.smoke:
            one = load_data(data_path)[:1]
            data_path = path / 'hotpot_one.json'
            if data_path.exists() and read(data_path) != one:
                raise ValueError('Smoke question changed; create a new experiment')
            write(data_path, one)
        check_config = {**config, 'data_path': str(data_path)}
        if graph_coverage(check_config):
            raise ValueError(f'Missing graphs; run ./experiment extract {args.name} first')
        result = path / 'predictions.json'
        if not args.resume and (result.exists() or Path(str(result) + '.checkpoint.json').exists()):
            raise ValueError('Output exists; use --resume or create a new experiment')
        if args.resume and not Path(str(result) + '.checkpoint.json').exists():
            raise ValueError('No checkpoint to resume')
        command = [sys.executable, '-u', 'code/kg_rag_distractor.py', '--dataset', 'hotpotqa',
                   '--data_path', str(data_path), '--kg_dir', str(absolute(config['kg_dir'])),
                   '--result_path', str(result)]
        for key in ['model_name', 'embed_model_name', 'reranker', 'reranker_device',
                    'method', 'top_k']:
            command.extend(['--' + key, str(config[key])])
        command.extend(['--temperature', str(config.get('temperature', 0.0))])
        if config['method'] in ('sem', 'sem_cross', 'joint'):
            command.extend(['--sem_model', config['sem_model']])
        if config['method'] in ('sem', 'sem_cross'):
            command.extend(['--sem_threshold', str(config['sem_threshold'])])
        if config['method'] in ('cross', 'sem_cross'):
            for key in ['cross_model', 'cross_threshold']:
                command.extend(['--' + key, str(config[key])])
        if config['method'] == 'joint':
            for key in ['cross_model', 'joint_keep_k', 'cross_weight']:
                command.extend(['--' + key, str(config[key])])
        if config['method'] == 'baseline' and config.get('final_keep_k') is not None:
            command.extend(['--final_keep_k', str(config['final_keep_k'])])
        if config['method'] in ('reuse', 'path_reuse'):
            for key in ['joint_keep_k', 'cross_weight']:
                command.extend(['--' + key, str(config[key])])
        if config.get('score_cache_dir'):
            command.extend(['--score_cache_dir', str(absolute(config['score_cache_dir']))])
        if args.resume:
            command.append('--resume')
        record_environment(path, command)
        timed_call(command, path, 'prediction')
    print(f'Next: ./experiment evaluate {args.name}' + (' --smoke' if args.smoke else ''))


def completed(name, smoke=False):
    directory = folder(name)
    plan = read(directory / 'experiment.json') if (directory / 'experiment.json').exists() else None
    path = directory / 'smoke' if smoke else directory
    data_path = path / 'hotpot_one.json' if smoke else absolute(
        plan['data_path'] if plan else 'output/baseline50/hotpot_50.json')
    data = load_data(data_path)
    prediction = read(path / 'predictions.json')
    ids = {sample['_id'] for sample in data}
    if set(prediction['answer']) != ids or set(prediction['sp']) != ids:
        raise ValueError(f'{name}: prediction IDs are incomplete or contain extra IDs')
    checkpoint = path / 'predictions.json.checkpoint.json'
    if checkpoint.exists():
        state = read(checkpoint)
        if state['errors'] or state['prediction'] != prediction:
            raise ValueError(f'{name}: failed questions or prediction/checkpoint mismatch')
        config = state['config']
    else:
        config = read(path / 'config.json')
    # Match the runner's ujson serialization when checking historical fingerprints.
    import ujson
    digest = hashlib.sha256(ujson.dumps(data, sort_keys=True).encode()).hexdigest()
    if config['data_hash'] != digest:
        raise ValueError(f'{name}: dataset content differs from the completed run')
    if plan:
        for key in ['method', 'model_name', 'embed_model_name', 'reranker', 'reranker_device', 'top_k']:
            if config[key] != plan[key]:
                raise ValueError(f'{name}: plan differs from actual run: {key}')
        if 'temperature' in plan and config.get('temperature') != plan['temperature']:
            raise ValueError(f'{name}: plan differs from actual run: temperature')
        if plan['method'] == 'baseline' and config.get('final_keep_k') != plan.get('final_keep_k'):
            raise ValueError(f'{name}: baseline evidence cap differs from the plan')
        if plan['method'] in ('sem', 'cross', 'sem_cross'):
            filters = ['sem', 'cross'] if plan['method'] == 'sem_cross' else [plan['method']]
            keys = [method + suffix for method in filters for suffix in ['_model', '_threshold']]
            if any(config[key] != plan[key] for key in keys):
                raise ValueError(f'{name}: filter plan differs from actual run')
        if plan['method'] == 'joint' and any(config[key] != plan[key] for key in
                ['sem_model', 'cross_model', 'joint_keep_k', 'cross_weight']):
            raise ValueError(f'{name}: joint selector plan differs from actual run')
        if plan['method'] in ('reuse', 'path_reuse'):
            if not config.get('native_reranker_identity'):
                raise ValueError(f'{name}: missing native scorer fingerprint')
            if any(config[key] != plan[key] for key in ['joint_keep_k', 'cross_weight']):
                raise ValueError(f'{name}: native selector plan differs from actual run')
    contexts = {sample['_id']: sample['context'] for sample in data}
    for qid in ids:
        if not isinstance(prediction['answer'][qid], str) or not prediction['answer'][qid].strip():
            raise ValueError(f'{qid}: empty or invalid answer')
        facts = prediction['sp'][qid]
        if not isinstance(facts, list):
            raise ValueError(f'{qid}: evidence must be a list')
        for fact in facts:
            if not isinstance(fact, list) or len(fact) != 2:
                raise ValueError(f'{qid}: malformed evidence location')
            title, index = fact
            paragraphs = dict(contexts[qid])
            if title not in paragraphs or type(index) is not int or not 0 <= index < len(paragraphs[title]):
                raise ValueError(f'{qid}: evidence location outside candidate context')
        if len({tuple(fact) for fact in facts}) != len(facts):
            raise ValueError(f'{qid}: duplicate evidence locations')
    spec = importlib.util.spec_from_file_location('hotpot_eval', ROOT / 'code/util/hotpot_evaluate_v1.py')
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        evaluator.eval(str(path / 'predictions.json'), str(data_path))
    metrics = ast.literal_eval(stream.getvalue().strip())
    return path, config, metrics, sum(map(len, prediction['sp'].values())) / len(data)


def evaluate(args):
    directory = folder(args.name)
    if not directory.exists():
        raise FileNotFoundError(f'No experiment folder: {directory}')
    with lock(directory / 'smoke' if args.smoke else directory):
        path, config, metrics, average = completed(args.name, args.smoke)
        write(path / 'metrics.json', metrics)
        write(path / 'config.json', config)
        usage_path = path / 'usage.json'
        usage = read(usage_path) if usage_path.exists() else None
        write(path / 'summary.json', {'mean_evidence_sentences': average,
                                      'metrics': metrics, 'usage': usage})
    print(json.dumps(metrics, indent=2))
    print(f'Mean selected sentences: {average:.2f}')
    if usage:
        token_stats = usage['per_question_mean_std']
        print('Mean tokens per question: '
              f'prompt {token_stats["prompt_tokens"]["mean"]:.1f}, '
              f'completion {token_stats["completion_tokens"]["mean"]:.1f}, '
              f'total {token_stats["total_tokens"]["mean"]:.1f}')
        print('Mean latency per question: '
              f'query {token_stats["query_latency_seconds"]["mean"]:.2f}s, '
              f'Ollama generation {token_stats["ollama_generation_seconds"]["mean"]:.2f}s')


def compare(args):
    records = [(name, completed(name)) for name in args.names]
    first = records[0][1][1]

    def evidence_cap(record):
        if record['method'] in ('reuse', 'path_reuse'):
            return record['joint_keep_k']
        return record.get('final_keep_k') or record['top_k']

    # Evidence/source changes are reported, not silently treated as the same run.
    common = ['data_hash', 'kg_hash', 'model_name', 'embed_model_name', 'reranker',
              'reranker_device', 'reranker_fp16', 'top_k', 'temperature', 'use_tpt']
    for name, (_, config, _, _) in records[1:]:
        differing = [key for key in common if config.get(key) != first.get(key)]
        if differing:
            raise ValueError(f'{name}: incomparable settings: {", ".join(differing)}')
        if (config.get('native_reranker_identity') and first.get('native_reranker_identity') and
                config['native_reranker_identity'] != first['native_reranker_identity']):
            raise ValueError(f'{name}: native BGE model fingerprints/settings differ')
        if evidence_cap(config) != evidence_cap(first):
            raise ValueError(f'{name}: final evidence caps differ')
    print('Run                 Method       Seeds  Cap  Answer EM  Answer F1  SP prec  SP recall  SP F1  Sentences')
    for name, (_, config, metrics, average) in records:
        print(f'{name:20} {config["method"]:12} {config["top_k"]:5} {evidence_cap(config):4} '
              f'{100*metrics["em"]:8.2f} {100*metrics["f1"]:10.2f} '
              f'{100*metrics["sp_prec"]:8.2f} {100*metrics["sp_recall"]:10.2f} '
              f'{100*metrics["sp_f1"]:6.2f} {average:10.2f}')
    if len({record[1][1]['source_hash'] for record in records}) > 1:
        print('NOTE: source fingerprints differ; inspect changes before attributing effects to a method.')
    print('Mean per-question usage (± standard deviation):')
    print('Run                 Prompt tokens  Completion tokens  Total tokens  Query seconds  Ollama seconds')
    for name, (path, _, _, _) in records:
        usage_path = path / 'usage.json'
        if not usage_path.exists():
            print(f'{name:20} usage not recorded')
            continue
        stats = read(usage_path)['per_question_mean_std']
        def shown(key):
            row = stats[key]
            return 'n/a' if row['mean'] is None else f'{row["mean"]:.1f}±{row["std"]:.1f}'
        print(f'{name:20} {shown("prompt_tokens"):>13} {shown("completion_tokens"):>19} '
              f'{shown("total_tokens"):>13} {shown("query_latency_seconds"):>14} '
              f'{shown("ollama_generation_seconds"):>15}')
    print('Scores are percentages. Sentence counts are not token counts; these are development results.')


def status(args):
    path = folder(args.name)
    if not path.exists():
        raise FileNotFoundError(f'No experiment folder: {path}')
    plan = read(path / 'experiment.json') if (path / 'experiment.json').exists() else None
    data = load_data(absolute(plan['data_path'] if plan else 'output/baseline50/hotpot_50.json'))
    checkpoint = path / 'predictions.json.checkpoint.json'
    if checkpoint.exists():
        state = read(checkpoint)
        print(f'{args.name}: {len(state["prediction"]["answer"])}/{len(data)} successful; '
              f'{len(state["errors"])} recorded failures')
    elif (path / 'predictions.json').exists():
        completed(args.name)
        print(f'{args.name}: {len(data)}/{len(data)} prediction IDs verified; no local checkpoint')
    else:
        print(f'{args.name}: prepared, no full prediction output yet')


def check(args):
    # Parsing does not create bytecode or import the extraction script's model code.
    for path in [*sorted((ROOT / 'code').rglob('*.py')),
                 *sorted((ROOT / 'scripts').glob('*.py')), *sorted((ROOT / 'tests').glob('*.py'))]:
        ast.parse(path.read_text(), filename=str(path))
    result = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests',
                             '-p', args.pattern, '-v'], cwd=ROOT, capture_output=True, text=True)
    if result.returncode:
        print(result.stdout, end='')
        print(result.stderr, end='', file=sys.stderr)
        raise ValueError('Offline tests failed')
    summary = re.search(r'Ran (\d+) tests? in', result.stderr)
    if not summary or int(summary.group(1)) == 0:
        raise ValueError('No tests matched the requested pattern')
    print(f'{summary.group(1)} offline tests passed.')
    subprocess.run(['git', 'diff', '--check'], cwd=ROOT, check=True)
    subprocess.run(['git', 'diff', '--cached', '--check'], cwd=ROOT, check=True)
    print('Syntax, offline tests, and whitespace checks passed. No models downloaded or run.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('check', help='Run offline checks')
    p.add_argument('--pattern', default='test*.py', help='unittest filename pattern')
    p.set_defaults(action=check)
    p = commands.add_parser('prepare', help='Sample a fixed development dataset')
    p.add_argument('name'); p.add_argument('--size', type=int, default=50)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--source', default='data/hotpotqa/hotpot_dev_distractor_v1.json')
    p.set_defaults(action=prepare)
    p = commands.add_parser('new', help='Create a named experiment without running models')
    p.add_argument('name'); p.add_argument('--method', choices=['baseline', 'sem', 'cross', 'sem_cross', 'joint', 'reuse', 'path_reuse'], default='baseline')
    p.add_argument('--score-cache', default=None, help='Exact-input BGE cache directory (reuse default: output/score_cache)')
    p.add_argument('--joint-keep-k', type=int, default=10, help='Maximum evidence sentences retained by the selector')
    p.add_argument('--final-keep-k', type=int, default=None,
                   help='Baseline final evidence sentence cap (default: top-k)')
    p.add_argument('--cross-weight', type=float, default=0.7, help='CROSS share of fused rank score; semantic rank receives the remainder')
    p.add_argument('--threshold', type=float, default=None, help='Single-filter cutoff (default 0.5); use separate cutoffs for sem_cross')
    p.add_argument('--sem-threshold', type=float, default=None, help='SEM cosine cutoff (combined default 0.3)')
    p.add_argument('--cross-threshold', type=float, default=None, help='CROSS raw-score cutoff (default 0.5)')
    p.add_argument('--data', default='output/baseline50/hotpot_50.json')
    p.add_argument('--kgs', default='output/baseline50/kgs')
    p.add_argument('--model', default='llama3:8b'); p.add_argument('--embedding', default='mxbai-embed-large')
    p.add_argument('--reranker', default='BAAI/bge-reranker-large')
    p.add_argument('--temperature', type=float, default=0.0,
                   help='Ollama generation temperature; default 0 for repeatable comparisons')
    p.add_argument('--sem-model', default='sentence-transformers/all-MiniLM-L6-v2')
    p.add_argument('--cross-model', default='cross-encoder/ms-marco-MiniLM-L6-v2')
    p.add_argument('--top-k', type=int, default=10); p.set_defaults(action=new)
    for command, action in [('graphs', graphs), ('extract', extract), ('run', run),
                            ('evaluate', evaluate), ('status', status)]:
        p = commands.add_parser(command)
        p.add_argument('name'); p.set_defaults(action=action)
        if command in ('run', 'evaluate'):
            p.add_argument('--smoke', action='store_true')
        if command == 'run':
            p.add_argument('--resume', action='store_true')
    p = commands.add_parser('compare', help='Validate and compare completed runs')
    p.add_argument('names', nargs='+'); p.set_defaults(action=compare)
    args = parser.parse_args()
    try:
        args.action(args)
    except (ValueError, KeyError, OSError, subprocess.SubprocessError) as error:
        parser.exit(1, f'Error: {error}\n')
    except KeyboardInterrupt:
        parser.exit(130, 'Interrupted. Reuse the same run command with --resume.\n')


if __name__ == '__main__':
    main()
