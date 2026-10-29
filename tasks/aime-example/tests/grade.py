"""Grade one AIME final-answer file. Runs in the container using stdlib only."""

import argparse
import json
from pathlib import Path
import re


def parse_answer(text: str) -> int | None:
    """Accept one ASCII integer, 0..999; whitespace and leading zeros are OK."""
    text = text.strip()
    if re.fullmatch(r"[0-9]{1,3}", text) is None:
        return None
    return int(text)


def grade_submission(answer_path: Path, expected_path: Path) -> dict:
    # An invalid/missing reference is a broken task, not an incorrect model answer.
    expected = parse_answer(expected_path.read_text(encoding="utf-8"))
    if expected is None:
        raise ValueError(f"Invalid reference answer: {expected_path}")

    if not answer_path.exists():
        return {"reward": 0.0, "reason": "missing_answer"}
    if answer_path.stat().st_size > 128:
        return {"reward": 0.0, "reason": "invalid_format"}
    try:
        answer = parse_answer(answer_path.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        answer = None
    if answer is None:
        return {"reward": 0.0, "reason": "invalid_format"}
    correct = answer == expected
    return {
        "reward": float(correct),
        "reason": "correct" if correct else "incorrect_answer",
        "parsed_answer": answer,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answer", type=Path, default=Path("/app/answer.txt"))
    parser.add_argument(
        "--expected", type=Path,
        default=Path(__file__).resolve().with_name("expected_answer.txt"),
    )
    parser.add_argument(
        "--reward", type=Path, default=Path("/logs/verifier/reward.txt"),
    )
    parser.add_argument(
        "--details", type=Path, default=Path("/logs/verifier/grading.json"),
    )
    args = parser.parse_args()
    # Avoid leaving an old successful reward behind if grading fails this time.
    args.reward.unlink(missing_ok=True)
    args.details.unlink(missing_ok=True)
    result = grade_submission(args.answer, args.expected)
    args.reward.parent.mkdir(parents=True, exist_ok=True)
    args.details.parent.mkdir(parents=True, exist_ok=True)
    args.reward.write_text(f"{result['reward']}\n", encoding="utf-8")
    args.details.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))
    # A completed grading of an incorrect response is not an infrastructure error.


if __name__ == "__main__":
    main()
