"""Standard-library-only file-transfer helper, sent to a temporary CPU pod."""

import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import sys
import tarfile
import time


def relative(value):
    path = PurePosixPath(value)
    if path.is_absolute() or '..' in path.parts or str(path) == '.':
        raise ValueError('Unsafe transfer path')
    return path


def request_digest(request):
    return hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()


def evaluation_directory(request, root):
    directory = root / str(relative(request['config']['output_relative_path'])) / str(relative(request['evaluation_id']))
    if not directory.resolve().is_relative_to(root.resolve()):
        raise ValueError('Evaluation directory escapes output storage')
    return directory


def check_request(request, directory, initialize=False):
    target = directory / 'request.json'
    if target.exists():
        if json.loads(target.read_text()) != request:
            raise ValueError('Stored evaluation belongs to another request')
    elif initialize:
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError('Preserve existing output without a matching request')
        directory.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(request, indent=2) + '\n')
    else:
        raise FileNotFoundError('Stored request is missing')


def generation_path(request, directory):
    pointer = json.loads((directory / 'generation.json').read_text())
    if pointer['request_sha256'] != request_digest(request):
        raise ValueError('Generation pointer belongs to another request')
    path = relative(pointer['run_relative_path'])
    if len(path.parts) != 2 or path.parts[0] != 'runs':
        raise ValueError('Invalid generation run directory')
    run = directory / str(path)
    if not run.resolve().is_relative_to(directory.resolve()):
        raise ValueError('Generation directory escapes evaluation')
    return run


def prepare(request, phase, output=Path('/results'), suites=Path('/suites'), scratch=Path('/tmp/transfer')):
    scratch.mkdir(parents=True, exist_ok=True)
    directory = evaluation_directory(request, output)
    check_request(request, directory, initialize=phase == 'stage')
    if phase == 'stage':
        archive = suites / str(relative(request['config']['suite_archive']))
        shutil.copyfile(archive, scratch / 'suite.tgz')
        actual = hashlib.sha256((scratch / 'suite.tgz').read_bytes()).hexdigest()
        if actual != request['config']['archive_sha256']:
            raise ValueError('Stored benchmark archive fingerprint differs')
    elif phase == 'download':
        run = generation_path(request, directory)
        with tarfile.open(scratch / 'generation.tgz', 'w:gz') as archive:
            archive.add(directory / 'generation.json', arcname='generation.json')
            for name in ('run.json', 'predictions.jsonl'):
                archive.add(run / name, arcname=name)
    elif phase != 'upload':
        raise ValueError('Unknown transfer phase')
    (scratch / 'ready').touch()


def apply_upload(request, output=Path('/results'), scratch=Path('/tmp/transfer')):
    directory = evaluation_directory(request, output)
    check_request(request, directory)
    run = generation_path(request, directory)
    names = {'run.json', 'summary.json', 'grading-diagnostics.tgz', 'report.json', 'benchmarks.csv', 'pipeline.json'}
    contents = {}
    with tarfile.open(scratch / 'upload.tgz', 'r:gz') as archive:
        for member in archive.getmembers():
            if member.name not in names or member.name in contents or not member.isfile():
                raise ValueError('Unexpected grading archive member')
            contents[member.name] = archive.extractfile(member).read()
    if set(contents) != names:
        raise ValueError('Incomplete grading archive')
    report, summary = (json.loads(contents[name]) for name in ('report.json', 'summary.json'))
    pipeline = json.loads(contents['pipeline.json'])
    if (report['evaluation_id'] != request['evaluation_id'] or report['status'] != 'complete' or
            report['request_sha256'] != request_digest(request) or summary['status'] != 'complete' or
            pipeline['request_sha256'] != request_digest(request) or pipeline['status'] != 'complete'):
        raise ValueError('Invalid or incomplete report')
    original, updated = json.loads((run / 'run.json').read_text()), json.loads(contents['run.json'])
    if ({k: v for k, v in original.items() if k != 'status'} !=
            {k: v for k, v in updated.items() if k != 'status'} or updated['status'] != 'graded'):
        raise ValueError('Uploaded grading changed the generation identity')
    if (report['suite_sha256'] != original['suite_sha256'] or
            summary['suite_sha256'] != original['suite_sha256'] or
            report['model'] != original['model'] or summary['model'] != original['model']):
        raise ValueError('Report does not match the generated predictions')
    destinations = {name: (run if name in ('run.json', 'summary.json', 'grading-diagnostics.tgz') else directory) / name
                    for name in names}
    # Check every destination before writing; retries may repeat identical data.
    for name, target in destinations.items():
        if name != 'run.json' and target.exists() and target.read_bytes() != contents[name]:
            raise FileExistsError('Preserve different existing results: ' + str(target))
    for name in sorted(names - {'report.json'}) + ['report.json']:
        target = destinations[name]
        temporary = target.with_name(target.name + '.tmp')
        temporary.write_bytes(contents[name])
        temporary.replace(target)
    print('MATH_EVAL_UPLOAD_COMPLETE', flush=True)


def main():
    request, phase = json.loads(sys.argv[1]), sys.argv[2]
    if phase == 'apply':
        apply_upload(request)
        return
    prepare(request, phase)
    print('MATH_EVAL_TRANSFER_READY', flush=True)
    deadline = time.monotonic() + 1200
    while not Path('/tmp/transfer/done').exists():
        if time.monotonic() >= deadline:
            raise TimeoutError('File transfer did not finish in twenty minutes')
        time.sleep(1)


if __name__ == '__main__':
    main()
