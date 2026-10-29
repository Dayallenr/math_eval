"""Pinned benchmark adapters and self-contained Harbor task generation."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile

from . import grading

PROMPT_VERSION = 'boxed-v1'
INSTRUCTION = r'Return your final response within \boxed{}. Place all requested answers in one final box. Put any units outside the box.'


@dataclass(frozen=True)
class Source:
    dataset: str
    revision: str
    files: tuple[str, ...]
    count: int
    selection: str


SOURCES = {
    'math500': Source('HuggingFaceH4/MATH-500', '6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be',
                      ('test.jsonl',), 500, 'test'),
    'aime2026': Source('MathArena/aime_2026', 'd61f486805306e6e6b5913504c07763a99e1b112',
                       ('data/train-00000-of-00001.parquet',), 30, 'train'),
    'hmmt2026': Source('MathArena/hmmt_feb_2026', '589a53fe152febc0f4f99f3e561007bf8da8e664',
                       ('data/train-00000-of-00001.parquet',), 33, 'train'),
    'olympiadbench': Source('Hothan/OlympiadBench', '91184b52131e7fc9455fef848035173aea8cc01a',
                            ('OlympiadBench/OE_TO_maths_en_COMP/OE_TO_maths_en_COMP.parquet',),
                            674, 'OE_TO_maths_en_COMP / train'),
    'olymmath': Source('RUC-AIBOX/OlymMATH', '2c6532ea2cf929ac1c421532af5951553eaee727',
                       ('data/OlymMATH-EN-EASY.jsonl', 'data/OlymMATH-EN-HARD.jsonl'),
                       200, 'English EASY + HARD; excludes Chinese and Lean'),
}


def digest(data: bytes | str) -> str:
    return hashlib.sha256(data.encode() if isinstance(data, str) else data).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def write_json(path: Path, data: dict) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def normalize(benchmark: str, rows: list[dict]) -> list[dict]:
    """Only allowlist question fields; never interpolate solutions into prompts."""
    problems = []
    seen = set()
    for row in rows:
        if benchmark in ('aime2026', 'hmmt2026'):
            source_id = str(row['problem_idx'])
            if re.fullmatch(r'[1-9][0-9]*', source_id) is None:
                raise ValueError(f'Invalid source index: {source_id}')
            if int(source_id) > SOURCES[benchmark].count:
                raise ValueError(f'Source index out of range: {source_id}')
            question, answer = row['problem'], row['answer']
            sort_key = f'{int(source_id):03d}'
        elif benchmark == 'olympiadbench':
            source_id, question = str(row['id']), row['question']
            if re.fullmatch(r'[1-9][0-9]*', source_id) is None:
                raise ValueError('Invalid OlympiadBench source ID')
            if row.get('language') != 'English' or row.get('subject') != 'Math' or row.get('modality') != 'Text-only':
                raise ValueError('Unexpected OlympiadBench language/subject/modality')
            if row.get('question_type') != 'Open-ended' or row.get('context'):
                raise ValueError('Unexpected OlympiadBench type/context; review adapter')
            answers = row['final_answer']
            if not isinstance(answers, list) or len(answers) != 1 or not isinstance(answers[0], str):
                raise ValueError('Expected one annotated final_answer string; review adapter')
            answer = answers[0]
            sort_key = source_id.zfill(6)
        else:
            source_id, question, answer = row['unique_id'], row['problem'], row['answer']
            if not isinstance(source_id, str) or not source_id.strip():
                raise ValueError('Invalid source unique_id')
            sort_key = source_id
        if source_id in seen:
            raise ValueError(f'Duplicate source ID in {benchmark}: {source_id}')
        seen.add(source_id)
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f'Invalid question: {source_id}')
        if isinstance(answer, bool) or not isinstance(answer, (int, str)) or not str(answer).strip():
            raise ValueError(f'Invalid reference: {source_id}')
        kind = 'integer' if benchmark == 'aime2026' else 'math'
        if kind == 'integer' and re.fullmatch(r'[0-9]{1,3}', str(answer)) is None:
            raise ValueError('AIME reference outside 0..999')
        metadata = {key: row[key] for key in ('subject', 'level', 'subfield', 'answer_type', 'is_multiple_answer', 'unit') if row.get(key) is not None}
        # Hash IDs that contain slashes; directory names never contain source text.
        suffix = source_id.zfill(3) if benchmark in ('aime2026', 'hmmt2026', 'olympiadbench') else digest(source_id)[:16]
        problems.append({'task_id': f'{benchmark}-{suffix}', 'source_id': source_id,
                         'benchmark': benchmark, 'question': question, 'kind': kind,
                         'metadata': metadata, '_answer': str(answer), '_sort': sort_key})
    if not problems:
        raise ValueError(f'Empty benchmark: {benchmark}')
    return sorted(problems, key=lambda row: row['_sort'])


DOCKERFILE = '''FROM python:3.12-slim
RUN pip install --no-cache-dir "math-verify[antlr4_13_2]==0.9.0" "latex2sympy2-extended==1.11.0" "sympy==1.14.0"
WORKDIR /app
'''
TEST_SH = '''#!/bin/bash
set -eu
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec python "${SCRIPT_DIR}/grade.py" "$@"
'''


def prepare(output: Path, benchmarks: list[str], limit: int | None = None,
            local_sources: Path | None = None) -> dict:
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite suite: {output}')
    if limit is not None and limit < 1:
        raise ValueError('Limit must be positive')
    if not benchmarks or len(set(benchmarks)) != len(benchmarks):
        raise ValueError('Select distinct benchmarks')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.suite-', dir=output.parent) as tmp:
        stage = Path(tmp) / 'build'
        stage.mkdir()
        public, sources, selected_counts = [], {}, {}
        for benchmark in benchmarks:
            spec = SOURCES[benchmark]
            rows, files = [], []
            if local_sources:
                path = local_sources / f'{benchmark}.jsonl'
                rows = read_jsonl(path)
                files.append({'file': path.name, 'sha256': digest(path.read_bytes())})
            else:
                from huggingface_hub import hf_hub_download
                for filename in spec.files:
                    path = Path(hf_hub_download(repo_id=spec.dataset, repo_type='dataset',
                                               revision=spec.revision, filename=filename))
                    if path.suffix == '.jsonl':
                        rows.extend(read_jsonl(path))
                    else:
                        import pyarrow.parquet as pq
                        rows.extend(pq.read_table(path).to_pylist())
                    files.append({'file': filename, 'sha256': digest(path.read_bytes())})
                if len(rows) != spec.count:
                    raise ValueError(f'{benchmark}: expected {spec.count} rows, got {len(rows)}')
            problems = normalize(benchmark, rows)
            selected = problems if limit is None else problems[:limit]
            selected_counts[benchmark] = len(selected)
            sources[benchmark] = {
                'kind': 'local_jsonl' if local_sources else 'huggingface',
                'dataset': spec.dataset, 'revision': None if local_sources else spec.revision,
                'selection': spec.selection, 'source_rows': len(rows), 'expected_full_count': spec.count,
                'files': files,
            }
            for problem in selected:
                task_id = problem['task_id']
                task = stage / 'tasks' / benchmark / task_id
                for child in ('environment', 'tests', 'solution'):
                    (task / child).mkdir(parents=True, exist_ok=True)
                prompt = INSTRUCTION + '\n\n' + problem['question']
                instruction = f'Task ID: {task_id}\n\n{prompt}\n\nWrite only the extracted final answer to /app/answer.txt.\n'
                (task / 'instruction.md').write_text(instruction, encoding='utf-8')
                (task / 'environment/Dockerfile').write_text(DOCKERFILE, encoding='utf-8')
                (task / 'tests/test.sh').write_text(TEST_SH, encoding='utf-8')
                (task / 'tests/test.sh').chmod(0o755)
                shutil.copyfile(grading.__file__, task / 'tests/grade.py')
                write_json(task / 'tests/reference.json', {'kind': problem['kind'], 'answer': problem['_answer']})
                # Oracle transports a JSON string safely; no source answer becomes shell code.
                oracle = '#!/bin/sh\nset -eu\npython - <<\'PY_ORACLE\'\nimport json\nfrom pathlib import Path\nanswer = json.loads(' + repr(json.dumps(problem['_answer'])) + ')\nPath("/app/answer.txt").write_text(answer, encoding="utf-8")\nPY_ORACLE\n'
                (task / 'solution/solve.sh').write_text(oracle, encoding='utf-8')
                (task / 'solution/solve.sh').chmod(0o755)
                (task / 'task.toml').write_text(
                    f'schema_version = "1.4"\n[task]\nname = "local/{task_id}"\n'
                    f'[metadata]\nbenchmark = "{benchmark}"\n'
                    '[agent]\ntimeout_sec = 120.0\n[verifier]\ntimeout_sec = 30.0\n'
                    '[environment]\nbuild_timeout_sec = 600.0\n', encoding='utf-8')
                public.append({key: problem[key] for key in ('task_id', 'source_id', 'benchmark', 'question', 'kind', 'metadata')} | {
                    'prompt': prompt, 'instruction_sha256': digest(instruction),
                })
        (stage / 'questions.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in public), encoding='utf-8')
        # Covers gold, grader, prompt, and environment; tampering changes the suite identity.
        hashes = {str(path.relative_to(stage)): digest(path.read_bytes()) for path in sorted(stage.rglob('*')) if path.is_file()}
        manifest = {'schema_version': 1, 'prompt_version': PROMPT_VERSION, 'sources': sources,
                    'adapter_sha256': digest(Path(__file__).read_bytes()),
                    'counts': selected_counts, 'question_count': len(public),
                    'subset': bool(local_sources) or any(selected_counts[key] != SOURCES[key].count for key in benchmarks),
                    'protocol': 'single-response boxed-v1; integer AIME; Math-Verify 0.9.0 elsewhere; custom protocol, not paper reproduction',
                    'file_sha256': hashes}
        write_json(stage / 'suite.json', manifest)
        stage.rename(output)
    return manifest


def load_suite(root: Path) -> tuple[dict, list[dict], str]:
    manifest_path = root / 'suite.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('schema_version') != 1:
        raise ValueError('Unsupported suite schema')
    actual_files = {str(path.relative_to(root)) for path in root.rglob('*') if path.is_file() and path.name != 'suite.json'}
    if actual_files != set(manifest['file_sha256']):
        raise ValueError('Suite files added or removed; regenerate the suite')
    for filename, expected in manifest['file_sha256'].items():
        path = (root / filename).resolve()
        if not path.is_relative_to(root.resolve()) or digest(path.read_bytes()) != expected:
            raise ValueError(f'Suite file changed or invalid: {filename}')
    questions = read_jsonl(root / 'questions.jsonl')
    ids = [row['task_id'] for row in questions]
    if len(ids) != len(set(ids)) or len(ids) != manifest['question_count']:
        raise ValueError('Suite question count/IDs mismatch')
    actual = {name: sum(row['benchmark'] == name for row in questions) for name in manifest['counts']}
    if actual != manifest['counts']:
        raise ValueError('Suite benchmark counts mismatch')
    return manifest, questions, digest(manifest_path.read_bytes())
