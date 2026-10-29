"""Validate saved predictions, run verifiers, and summarize complete results."""

import json
import os
from pathlib import Path
import subprocess
import sys

from .grading import extract_final_answer
from .suite import digest, load_suite, read_jsonl, write_json


def validate_predictions(suite: Path, run_dir: Path):
    manifest, questions, suite_hash = load_suite(suite)
    run = json.loads((run_dir / 'run.json').read_text())
    if run.get('status') not in ('generated', 'graded', 'grading_failed'):
        raise ValueError('Generation did not complete; partial runs cannot receive benchmark scores')
    if run['suite_sha256'] != suite_hash:
        raise ValueError('Predictions belong to a different suite')
    predictions_path = run_dir / 'predictions.jsonl'
    if digest(predictions_path.read_bytes()) != run['predictions_sha256']:
        raise ValueError('Predictions changed after generation')
    predictions = read_jsonl(predictions_path)
    by_id = {row['task_id']: row for row in predictions}
    if len(predictions) != len(by_id) or set(by_id) != {q['task_id'] for q in questions}:
        raise ValueError('Missing, duplicate, or extra prediction IDs')
    for question in questions:
        record = by_id[question['task_id']]
        if any(record[key] != question[key] for key in ('benchmark', 'instruction_sha256', 'prompt')) or record['suite_sha256'] != suite_hash:
            raise ValueError('Prediction question/provenance mismatch')
        if record['answer'] != extract_final_answer(record['response']):
            raise ValueError('Saved answer does not match response extraction')
    return manifest, questions, run, by_id


def summarize(manifest: dict, questions: list[dict], predictions: dict, graded: dict,
              backend: str) -> dict:
    rows, scores = [], {}
    for question in questions:
        task_id = question['task_id']
        result = graded.get(task_id, {'error': 'No verifier result'})
        prediction = predictions[task_id]
        rows.append({'task_id': task_id, 'benchmark': question['benchmark'],
                     'finish_reason': prediction['finish_reason'],
                     'output_tokens': prediction.get('output_tokens'),
                     'answer': prediction['answer'], **result})
    for name, expected in manifest['counts'].items():
        subset = [row for row in rows if row['benchmark'] == name]
        valid = [row for row in subset if 'error' not in row and row.get('reward') in (0.0, 1.0)]
        errors = len(subset) - len(valid)
        scores[name] = {
            'expected': expected, 'graded': len(valid), 'errors': errors,
            'correct': sum(row['reward'] == 1.0 for row in valid),
            'accuracy': sum(row['reward'] for row in valid) / expected if errors == 0 else None,
            'no_final_answer': sum(row['answer'] is None for row in subset),
            'truncated': sum(row['finish_reason'] == 'length' for row in subset),
            'mean_output_tokens': sum(row.get('output_tokens') or 0 for row in subset) / expected,
        }
    return {'status': 'complete' if all(row['errors'] == 0 for row in scores.values()) else 'incomplete',
            'grading_backend': backend, 'subset': manifest['subset'],
            'protocol': manifest['protocol'], 'benchmarks': scores, 'results': rows}


def read_harbor_job(job: Path, expected_ids: set[str]) -> dict:
    graded = {}
    for result_path in job.glob('*/result.json'):
        result = json.loads(result_path.read_text())
        # Harbor records the task.toml name, including our local/ namespace.
        task_id = result['task_name'].removeprefix('local/')
        if task_id not in expected_ids or task_id in graded:
            raise ValueError(f'Unexpected or duplicate Harbor trial: {task_id}')
        if result.get('exception_info'):
            graded[task_id] = {'error': result['exception_info']['exception_message']}
        else:
            details = result_path.parent / 'verifier/grading.json'
            if not details.exists():
                graded[task_id] = {'error': 'Harbor did not return verifier diagnostics'}
                continue
            detail = json.loads(details.read_text())
            reward = ((result.get('verifier_result') or {}).get('rewards') or {}).get('reward')
            if reward not in (0.0, 1.0) or detail.get('reward') != reward:
                graded[task_id] = {'error': 'Harbor reward missing or disagrees with grader diagnostics'}
            else:
                graded[task_id] = detail
    return graded


def grade_run(suite: Path, run_dir: Path, backend: str = 'harbor', concurrency: int = 2) -> dict:
    manifest, questions, run, predictions = validate_predictions(suite, run_dir)
    destination = run_dir / 'grading'
    if destination.exists():
        raise FileExistsError('Grading already attempted. Preserve this run; copy it to a new directory to retry.')
    destination.mkdir()
    graded = {}
    try:
        if backend == 'local':
            for question in questions:
                task_id, benchmark = question['task_id'], question['benchmark']
                task = suite / 'tasks' / benchmark / task_id
                trial = destination / task_id
                trial.mkdir()
                (trial / 'answer.txt').write_text(predictions[task_id]['answer'] or '', encoding='utf-8')
                command = [sys.executable, str(task / 'tests/grade.py'),
                           '--answer', str(trial / 'answer.txt'), '--expected', str(task / 'tests/reference.json'),
                           '--reward', str(trial / 'reward.txt'), '--details', str(trial / 'grading.json')]
                try:
                    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
                    (trial / 'verifier.log').write_text(result.stdout + result.stderr)
                    if result.returncode:
                        raise RuntimeError(f'Verifier exited {result.returncode}; see {trial}/verifier.log')
                    graded[task_id] = json.loads((trial / 'grading.json').read_text())
                except (OSError, RuntimeError, subprocess.TimeoutExpired, ValueError) as error:
                    graded[task_id] = {'error': str(error)}
        elif backend == 'harbor':
            # Inherit the source path so custom agents also work from a source checkout.
            env = os.environ.copy()
            source = str(Path(__file__).resolve().parents[1])
            env['PYTHONPATH'] = source + os.pathsep + env.get('PYTHONPATH', '')
            for benchmark in manifest['counts']:
                command = [sys.executable, '-m', 'harbor.cli.main', 'run',
                           '-p', str((suite / 'tasks' / benchmark).resolve()),
                           '-a', 'math_eval.replay:SavedPredictionAgent',
                           '--ak', 'predictions_file=' + str((run_dir / 'predictions.jsonl').resolve()),
                           '-m', run['model']['label'], '-n', str(concurrency), '-k', '1',
                           '--jobs-dir', str(destination.resolve()), '--job-name', benchmark]
                print(f'Harbor grading: {benchmark}', flush=True)
                with (destination / f'{benchmark}.log').open('w') as log:
                    process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env)
                job = destination / benchmark
                expected_ids = {q['task_id'] for q in questions if q['benchmark'] == benchmark}
                graded.update(read_harbor_job(job, expected_ids))
                if process.returncode:
                    for task_id in expected_ids:
                        graded[task_id] = {'error': f'Harbor job exited {process.returncode}; see {benchmark}.log'}
                    print(f'Harbor exited {process.returncode}; see {destination}/{benchmark}.log', file=sys.stderr)
        else:
            raise ValueError(f'Unknown backend: {backend}')
        summary = summarize(manifest, questions, predictions, graded, backend)
        summary.update(model=run['model'], run_id=run['run_id'], suite_sha256=run['suite_sha256'],
                       generation_kind=run['generation_kind'])
        write_json(run_dir / 'summary.json', summary)
        run['status'] = 'graded' if summary['status'] == 'complete' else 'grading_failed'
        write_json(run_dir / 'run.json', run)
        for benchmark, score in summary['benchmarks'].items():
            print(f"{benchmark}: {score['correct']}/{score['expected']}, accuracy={score['accuracy']}, errors={score['errors']}")
        if summary['status'] != 'complete':
            raise RuntimeError(f'Grading incomplete; inspect {run_dir}/summary.json')
        return summary
    except BaseException:
        run['status'] = 'grading_failed'
        write_json(run_dir / 'run.json', run)
        raise
