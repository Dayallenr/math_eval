"""Behavioral checks for dataset isolation, provenance, and incomplete runs."""

import asyncio
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from math_eval.grading import compare, extract_final_answer, grade_file
from math_eval.inference import generate_run, identify_model
from math_eval.results import grade_run, read_harbor_job, summarize, validate_predictions
from math_eval.suite import digest, load_suite, normalize, prepare, read_jsonl, SOURCES, write_json


SETTINGS = dict(batch_size=2, max_new_tokens=16, max_input_tokens=512,
                temperature=0.0, top_p=1.0, seed=42, device='cpu', dtype='auto', enable_thinking=False)


def fixtures(root):
    root.mkdir()
    records = {
        'math500': [{'unique_id': 'test/algebra/1.json', 'problem': 'Compute one half.', 'answer': r'\frac{1}{2}', 'solution': 'SECRET GOLD SOLUTION'}],
        'aime2026': [{'problem_idx': 1, 'problem': 'Compute two plus three.', 'answer': 5}],
        'hmmt2026': [{'problem_idx': 1, 'problem': 'Compute negative one half.', 'answer': r'-\frac{1}{2}'}],
        'olympiadbench': [{'id': 1606, 'question': 'Find the requested pair.', 'final_answer': ['(2,3)'], 'language': 'English', 'subject': 'Math', 'modality': 'Text-only', 'question_type': 'Open-ended', 'context': None, 'solution': ['SECRET GOLD SOLUTION']}],
        'olymmath': [{'unique_id': 'OlymMATH-EASY-0-EN', 'problem': 'Compute the square root of four.', 'answer': '2'}],
    }
    for key, rows in records.items():
        (root / f'{key}.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    return root


class FakeGenerator:
    """Explicit fixture, never exposed as an evaluation CLI backend."""
    metadata = {'kind': 'test_fixture'}

    def __init__(self, identity, settings):
        pass

    def generate(self, prompts):
        answers = ['0.5' if 'one half.' in p and 'negative' not in p else '-0.5' if 'negative' in p else '5' if 'two plus' in p else '(2,3)' if 'pair' in p else '2' for p in prompts]
        return [{'response': r'\boxed{' + a + '}', 'rendered_prompt': p,
                 'input_tokens': 10, 'output_tokens': 4, 'finish_reason': 'eos'} for p, a in zip(prompts, answers)]


class ExtractionTests(unittest.TestCase):
    def test_nested_braces_reasoning_and_multiple_boxes(self):
        self.assertEqual(extract_final_answer(r'\boxed{99} then \boxed{\frac{1}{2}}'), r'\frac{1}{2}')
        self.assertEqual(extract_final_answer(r'<think>\boxed{99}</think>Final: \boxed{5}'), '5')
        self.assertIsNone(extract_final_answer(r'<think>\boxed{99}'))
        self.assertIsNone(extract_final_answer(r'<think>\boxed{99}</think>No final answer'))
        self.assertIsNone(extract_final_answer(r'\boxed{5} later \boxed{6'))
        self.assertIsNone(extract_final_answer('I tried 12 and 15.'))
        self.assertEqual(extract_final_answer(' 005 '), '005')
        self.assertEqual(extract_final_answer(r'\boxed{\{1,2\}}'), r'\{1,2\}')

    def test_broken_reference_is_an_error_even_with_no_submission(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_json(root / 'reference.json', {'kind': 'integer', 'answer': '1000'})
            with self.assertRaises(ValueError):
                grade_file(root / 'missing.txt', root / 'reference.json')

    def test_symbolic_equivalence_and_integer_contract(self):
        self.assertEqual(compare('005', {'kind': 'integer', 'answer': '5'})['reward'], 1.0)
        self.assertEqual(compare('5.0', {'kind': 'integer', 'answer': '5'})['reward'], 0.0)
        for pred, gold in [('0.5', r'\frac{1}{2}'), (r'\sqrt{8}', r'2\sqrt{2}'), ('q+p', 'p+q')]:
            with self.subTest(pred=pred):
                self.assertEqual(compare(pred, {'kind': 'math', 'answer': gold})['reward'], 1.0)
        self.assertEqual(compare('3', {'kind': 'math', 'answer': '2'})['reward'], 0.0)
        self.assertEqual(compare('69', {'kind': 'math', 'answer': '$69$,$84$'})['reward'], 0.0)
        self.assertEqual(compare('69,84', {'kind': 'math', 'answer': '$69$,$84$'})['reward'], 1.0)
        with self.assertRaises(ValueError):
            compare('5', {'kind': 'integer', 'answer': '1000'})


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.suite = self.root / 'suite'
        prepare(self.suite, list(SOURCES), local_sources=fixtures(self.root / 'sources'))
        self.model = self.root / 'model'
        self.model.mkdir()
        (self.model / 'config.json').write_text('{}')
        (self.model / 'model.safetensors').write_bytes(b'fixture-not-real-weights')
        write_json(self.model / 'export_manifest.json', {'step': 100, 'training_run': 'grpo-test'})

    def generate(self):
        run = generate_run(self.suite, str(self.model), self.root / 'runs', SETTINGS, generator_factory=FakeGenerator)
        info = json.loads((run / 'run.json').read_text())
        info['generation_kind'] = 'test_fixture'
        write_json(run / 'run.json', info)
        return run

    def test_suite_has_no_gold_in_model_inputs_and_is_portable(self):
        manifest, questions, _ = load_suite(self.suite)
        self.assertEqual(manifest['question_count'], 5)
        self.assertTrue(manifest['subset'])
        self.assertIsNone(manifest['sources']['math500']['revision'])
        text = (self.suite / 'questions.jsonl').read_text()
        self.assertNotIn('SECRET GOLD SOLUTION', text)
        self.assertNotIn('"answer":', text)
        for q in questions:
            self.assertNotIn('/app/answer.txt', q['prompt'])
            self.assertEqual(q['prompt'].split('\n\n', 1)[1], q['question'])
        # No directory traversal through upstream IDs.
        rows = normalize('math500', [{'unique_id': '../../bad', 'problem': r'Keep \frac{1}{2} exact.', 'answer': '1'}])
        self.assertNotIn('/', rows[0]['task_id'])
        with self.assertRaises(FileExistsError):
            prepare(self.suite, list(SOURCES), local_sources=self.root / 'sources')

    def test_detect_suite_edits_and_extra_tasks(self):
        path = self.suite / 'questions.jsonl'
        original = path.read_bytes()
        path.write_bytes(original + b'\n')
        with self.assertRaises(ValueError):
            load_suite(self.suite)
        path.write_bytes(original)
        (self.suite / 'tasks/extra.txt').write_text('extra')
        with self.assertRaises(ValueError):
            load_suite(self.suite)

    def test_model_identity_and_reject_raw_adapters(self):
        identity = identify_model(str(self.model))
        self.assertEqual(identity['label'], 'grpo-test / step 100')
        (self.model / 'model.safetensors').rename(self.model / 'adapter_model.safetensors')
        with self.assertRaises(ValueError):
            identify_model(str(self.model))
        with self.assertRaises(FileNotFoundError):
            identify_model(str(self.root / 'missing'))

    def test_one_model_load_batched_generation_and_local_verifiers(self):
        with patch.object(FakeGenerator, '__init__', return_value=None) as init:
            run = self.generate()
            self.assertEqual(init.call_count, 1)
        summary = grade_run(self.suite, run, backend='local')
        self.assertEqual(summary['status'], 'complete')
        self.assertTrue(all(s['accuracy'] == 1.0 for s in summary['benchmarks'].values()))
        with self.assertRaises(FileExistsError):
            grade_run(self.suite, run, backend='local')

    def test_reject_missing_duplicate_and_modified_predictions(self):
        run = self.generate()
        predictions_path = run / 'predictions.jsonl'
        original = predictions_path.read_text()
        predictions_path.write_text(original + '\n')
        with self.assertRaisesRegex(ValueError, 'changed'):
            validate_predictions(self.suite, run)
        for rows in (read_jsonl(predictions_path)[:-1], read_jsonl(predictions_path) + [read_jsonl(predictions_path)[0]]):
            predictions_path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
            metadata = json.loads((run / 'run.json').read_text())
            metadata['predictions_sha256'] = digest(predictions_path.read_bytes())
            write_json(run / 'run.json', metadata)
            with self.assertRaises(ValueError):
                validate_predictions(self.suite, run)

    def test_generation_failure_stays_incomplete(self):
        class Broken(FakeGenerator):
            def generate(self, prompts):
                raise RuntimeError('GPU fixture failure')
        with self.assertRaises(RuntimeError):
            generate_run(self.suite, str(self.model), self.root / 'runs', SETTINGS, generator_factory=Broken)
        run = next((self.root / 'runs').iterdir())
        self.assertEqual(json.loads((run / 'run.json').read_text())['status'], 'generation_failed')
        with self.assertRaises(ValueError):
            validate_predictions(self.suite, run)

    def test_verifier_errors_do_not_become_accuracy_zero(self):
        run = self.generate()
        manifest, questions, _, predictions = validate_predictions(self.suite, run)
        result = summarize(manifest, questions, predictions, {}, 'harbor')
        self.assertEqual(result['status'], 'incomplete')
        self.assertTrue(all(s['accuracy'] is None for s in result['benchmarks'].values()))

    def test_harbor_namespaced_ids_and_reward_validation(self):
        trial = self.root / 'harbor/trial'
        (trial / 'verifier').mkdir(parents=True)
        write_json(trial / 'result.json', {'task_name': 'local/aime2026-001',
                   'exception_info': None, 'verifier_result': {'rewards': {'reward': 1.0}}})
        write_json(trial / 'verifier/grading.json', {'reward': 1.0, 'reason': 'correct'})
        result = read_harbor_job(trial.parent, {'aime2026-001'})
        self.assertEqual(result['aime2026-001']['reward'], 1.0)
        write_json(trial / 'verifier/grading.json', {'reward': 0.0})
        self.assertIn('error', read_harbor_job(trial.parent, {'aime2026-001'})['aime2026-001'])
        with self.assertRaises(ValueError):
            read_harbor_job(trial.parent, {'another-task'})

    def test_replay_only_transports_saved_prediction(self):
        from math_eval.replay import SavedPredictionAgent
        from harbor.models.agent.context import AgentContext
        run = self.generate()
        question = load_suite(self.suite)[1][0]
        instruction = self.suite / 'tasks' / question['benchmark'] / question['task_id'] / 'instruction.md'
        class Environment:
            async def upload_file(self, source, target):
                self.answer = Path(source).read_text()
                self.target = target
        env, context = Environment(), AgentContext()
        agent = SavedPredictionAgent(logs_dir=self.root / 'logs', predictions_file=str(run / 'predictions.jsonl'))
        asyncio.run(agent.run(instruction.read_text(), env, context))
        self.assertEqual(env.answer, '0.5')
        self.assertEqual(env.target, '/app/answer.txt')
        with self.assertRaises(ValueError):
            asyncio.run(agent.run('different question', env, context))


if __name__ == '__main__':
    unittest.main()
