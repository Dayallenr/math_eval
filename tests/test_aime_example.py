"""Keep the small teaching example's integer grader checked independently."""

import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    'aime_example_grader', ROOT / 'tasks/aime-example/tests/grade.py',
)
grader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(grader)


class ExampleGraderTests(unittest.TestCase):
    def test_integer_contract(self):
        for text, expected in [('5', 5), ('005\n', 5), (' 0\n', 0), ('999', 999)]:
            with self.subTest(text=text):
                self.assertEqual(grader.parse_answer(text), expected)
        for text in ['', '1000', '-1', '+5', '5.0', '5 6', 'The answer is 5', r'\boxed{5}', '５']:
            with self.subTest(text=text):
                self.assertIsNone(grader.parse_answer(text))

    def test_correct_incorrect_missing_and_malformed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            answer, expected = root / 'answer.txt', root / 'expected.txt'
            expected.write_text('5\n')
            self.assertEqual(grader.grade_submission(answer, expected)['reason'], 'missing_answer')
            for text, reward, reason in [('005\n', 1.0, 'correct'), ('6', 0.0, 'incorrect_answer'),
                                        ('5\n6', 0.0, 'invalid_format'), ('5' * 129, 0.0, 'invalid_format')]:
                answer.write_text(text)
                result = grader.grade_submission(answer, expected)
                self.assertEqual((result['reward'], result['reason']), (reward, reason))
            answer.write_bytes(b'\xff')
            self.assertEqual(grader.grade_submission(answer, expected)['reason'], 'invalid_format')

    def test_bad_reference_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            expected = root / 'expected.txt'
            expected.write_text('not a reference')
            with self.assertRaises(ValueError):
                grader.grade_submission(root / 'missing.txt', expected)


if __name__ == '__main__':
    unittest.main()
