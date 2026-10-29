"""Prepare once; evaluate any exported checkpoint with the same fixed suite."""

import argparse
import math
from pathlib import Path

from .inference import generate_run
from .results import grade_run
from .suite import SOURCES, prepare


def positive_int(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError('Must be positive')
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare', help='Download pinned datasets and generate Harbor tasks')
    prep.add_argument('--output', type=Path, required=True)
    prep.add_argument('--benchmarks', choices=list(SOURCES), nargs='+', default=list(SOURCES))
    prep.add_argument('--limit', type=positive_int, help='First N per benchmark: subset smoke test, not a full score')
    prep.add_argument('--local-sources', type=Path, help='Offline fixtures: BENCHMARK.jsonl; never claims Hub provenance')
    for name in ('generate', 'evaluate'):
        cmd = sub.add_parser(name, help='Generate model responses' if name == 'generate' else 'Generate and grade with Harbor')
        cmd.add_argument('--suite', type=Path, required=True)
        cmd.add_argument('--model', required=True, help='Merged/exported HF model directory, or Hub repo ID')
        cmd.add_argument('--revision', help='Hub model commit/tag; resolved to exact commit')
        cmd.add_argument('--output-root', type=Path, default=Path('runs'))
        cmd.add_argument('--batch-size', type=positive_int, default=1)
        cmd.add_argument('--max-new-tokens', type=positive_int, default=8192)
        cmd.add_argument('--max-input-tokens', type=positive_int, default=8192)
        cmd.add_argument('--temperature', type=float, default=0.0)
        cmd.add_argument('--top-p', type=float, default=1.0)
        cmd.add_argument('--seed', type=int, default=42)
        cmd.add_argument('--enable-thinking', action=argparse.BooleanOptionalAction, default=False)
        cmd.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
        cmd.add_argument('--dtype', choices=('auto', 'bfloat16', 'float16', 'float32'), default='auto')
        if name == 'evaluate':
            cmd.add_argument('--grading-backend', choices=('harbor', 'local'), default='harbor')
            cmd.add_argument('--concurrency', type=positive_int, default=2)
    grade = sub.add_parser('grade', help='Grade saved predictions, without a GPU or endpoint')
    grade.add_argument('--suite', type=Path, required=True)
    grade.add_argument('--run', type=Path, required=True)
    grade.add_argument('--backend', choices=('harbor', 'local'), default='harbor')
    grade.add_argument('--concurrency', type=positive_int, default=2)
    args = parser.parse_args()
    try:
        if args.command == 'prepare':
            manifest = prepare(args.output, args.benchmarks, args.limit, args.local_sources)
            print(f"Prepared {manifest['question_count']} questions: {manifest['counts']}")
            print(f"Subset: {manifest['subset']}; suite: {args.output}")
        elif args.command == 'grade':
            grade_run(args.suite.resolve(), args.run.resolve(), args.backend, args.concurrency)
        else:
            if not math.isfinite(args.temperature) or not math.isfinite(args.top_p) or args.temperature < 0 or not 0 < args.top_p <= 1:
                raise ValueError('Temperature must be >=0 and top_p must be in (0,1]')
            settings = {key: getattr(args, key) for key in (
                'batch_size', 'max_new_tokens', 'max_input_tokens', 'temperature', 'top_p',
                'seed', 'enable_thinking', 'device', 'dtype')}
            run = generate_run(args.suite.resolve(), args.model, args.output_root.resolve(), settings, args.revision)
            if args.command == 'evaluate':
                grade_run(args.suite.resolve(), run, args.grading_backend, args.concurrency)
    except (ValueError, OSError, RuntimeError, ImportError) as error:
        parser.exit(1, f'Error: {error}\n')


if __name__ == '__main__':
    main()
