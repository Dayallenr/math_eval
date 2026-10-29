"""Shared final-answer verifier; copied into each generated Harbor task."""

import argparse
import json
from pathlib import Path
import re


def extract_final_answer(response: str) -> str | None:
    """Use the last balanced box in the final response, ignoring reasoning tags.

    Never mine arbitrary numbers from a chain of thought. An unfinished final box
    is a formatting failure, even if an earlier intermediate box was complete.
    """
    if '</think>' in response:
        response = response.rsplit('</think>', 1)[1]
    elif '<think>' in response:
        return None
    matches = list(re.finditer(r'\\(?:boxed|fbox)\s*\{', response))
    if matches:
        start = matches[-1].end()
        depth, backslashes = 1, 0
        for pos in range(start, len(response)):
            char = response[pos]
            # Escaped braces are mathematical symbols, not grouping delimiters.
            if char == '\\':
                backslashes += 1
                continue
            escaped = backslashes % 2
            backslashes = 0
            if escaped:
                continue
            if char == '{':
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0:
                    answer = response[start:pos].strip()
                    return answer or None
        return None
    # A bare numeric answer is unambiguous. Reasoning prose is never a fallback.
    text = response.strip()
    return text if re.fullmatch(r'-?[0-9]+(?:\.[0-9]+)?', text) else None


def clean_math(text: str) -> str:
    return (text.strip().replace('$', '').replace(r'\(', '').replace(r'\)', '')
            .replace(r'\[', '').replace(r'\]', ''))


def validate_reference(reference: dict) -> None:
    if reference.get('kind') not in ('integer', 'math') or not isinstance(reference.get('answer'), str) or not reference['answer'].strip():
        raise ValueError('Invalid reference file')
    if reference['kind'] == 'integer' and re.fullmatch(r'[0-9]{1,3}', reference['answer'].strip()) is None:
        raise ValueError('AIME reference must be an integer in 0..999')


def compare(answer: str, reference: dict) -> dict:
    validate_reference(reference)
    gold = reference['answer']
    kind = reference['kind']
    if kind == 'integer':
        if re.fullmatch(r'[0-9]{1,3}', gold.strip()) is None:
            raise ValueError('AIME reference must be an integer in 0..999')
        if re.fullmatch(r'[0-9]{1,3}', answer.strip()) is None:
            return {'reward': 0.0, 'reason': 'invalid_format'}
        correct = int(answer) == int(gold)
        method = 'integer'
    elif kind == 'math':
        # Import even for exact matches: a broken grader installation is an error.
        from math_verify import LatexExtractionConfig, parse, verify
        prediction, expected = clean_math(answer), clean_math(gold)
        if prediction == expected:
            correct, method = True, 'exact'
        else:
            config = [LatexExtractionConfig()]
            parsed_gold = parse(r'\boxed{' + expected + '}', extraction_config=config,
                                fallback_mode='no_fallback', parsing_timeout=5)
            parsed_answer = parse(r'\boxed{' + prediction + '}', extraction_config=config,
                                  fallback_mode='no_fallback', parsing_timeout=5)
            correct = bool(parsed_gold and parsed_answer and verify(
                parsed_gold, parsed_answer, strict=True, timeout_seconds=5,
            ))
            method = 'math_verify' if parsed_gold and parsed_answer else 'unparseable'
    else:
        raise ValueError(f'Unknown answer kind: {kind}')
    return {'reward': float(correct), 'reason': 'correct' if correct else 'incorrect_answer',
            'method': method, 'answer': answer}


def grade_file(answer_path: Path, reference_path: Path) -> dict:
    reference = json.loads(reference_path.read_text(encoding='utf-8'))
    # Validate reference structure even when the model wrote no answer.
    validate_reference(reference)
    if not answer_path.exists():
        return {'reward': 0.0, 'reason': 'missing_answer'}
    if answer_path.stat().st_size > 16384:
        return {'reward': 0.0, 'reason': 'invalid_format'}
    try:
        answer = answer_path.read_text(encoding='utf-8').strip()
    except UnicodeDecodeError:
        return {'reward': 0.0, 'reason': 'invalid_format'}
    if not answer:
        return {'reward': 0.0, 'reason': 'no_final_answer'}
    return compare(answer, reference)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--answer', type=Path, default=Path('/app/answer.txt'))
    parser.add_argument('--expected', type=Path, default=Path(__file__).with_name('reference.json'))
    parser.add_argument('--reward', type=Path, default=Path('/logs/verifier/reward.txt'))
    parser.add_argument('--details', type=Path, default=Path('/logs/verifier/grading.json'))
    args = parser.parse_args()
    args.reward.unlink(missing_ok=True)
    args.details.unlink(missing_ok=True)
    result = grade_file(args.answer, args.expected)
    for path in (args.reward, args.details):
        path.parent.mkdir(parents=True, exist_ok=True)
    args.reward.write_text(f"{result['reward']}\n", encoding='utf-8')
    args.details.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
