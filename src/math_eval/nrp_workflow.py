"""Cluster-side controller and workers for an entire checkpoint evaluation."""

from contextlib import contextmanager
import csv
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import ssl
import tarfile
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from .inference import generate_run
from .nrp import dns_name, job_for, request_hash, validate_config
from .results import grade_run
from .suite import load_suite


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def validate_request(request):
    if request.get('schema_version') != 1:
        raise ValueError('Unsupported evaluation request')
    validate_config(request['config'])
    dns_name(request['evaluation_id'], maximum=48)
    dns_name(request['model_pvc'])
    from .nrp import relative_path
    request['model_relative_path'] = relative_path(request['model_relative_path'])
    if type(request['pilot']) is not bool:
        raise ValueError('pilot must be a boolean')
    return request


@contextmanager
def staged_suite(request, suite_mount=Path('/suites')):
    """Read one pinned archive from Ceph; hash/check task files on container disk."""
    config = request['config']
    name = config['pilot_suite'] if request['pilot'] else config['full_suite']
    fingerprint = config['pilot_sha256'] if request['pilot'] else config['full_sha256']
    archive = Path(suite_mount) / config['suite_archive']
    with tempfile.TemporaryDirectory(prefix='math-eval-suite-') as temporary:
        local_archive = Path(temporary) / 'suite.tgz'
        shutil.copyfile(archive, local_archive)
        if hashlib.sha256(local_archive.read_bytes()).hexdigest() != config['archive_sha256']:
            raise ValueError('Stored benchmark archive fingerprint differs from NRP config')
        with tarfile.open(local_archive, 'r:gz') as packed:
            members = []
            for member in packed.getmembers():
                path = PurePosixPath(member.name)
                if path.is_absolute() or '..' in path.parts or not (member.isfile() or member.isdir()):
                    raise ValueError('Invalid benchmark archive member')
                if path == PurePosixPath(name) or PurePosixPath(name) in path.parents:
                    members.append(member)
            packed.extractall(temporary, members=members, filter='data')
        suite = Path(temporary) / name
        if load_suite(suite)[2] != fingerprint:
            raise ValueError('Extracted suite fingerprint differs from NRP config')
        yield suite


def read_request(directory):
    return validate_request(json.loads((Path(directory) / 'request.json').read_text()))


def worker_generate(directory, suite_mount=Path('/suites'), model_mount=Path('/models'), generator_factory=None):
    directory = Path(directory).resolve()
    request = read_request(directory)
    config = request['config']
    with staged_suite(request, suite_mount) as suite:
        kwargs = {'generator_factory': generator_factory} if generator_factory is not None else {}
        run = generate_run(suite, str(Path(model_mount) / request['model_relative_path']),
                           directory / 'runs', config['settings'], **kwargs)
    pointer = {'run_relative_path': str(run.relative_to(directory)), 'request_sha256': request_hash(request)}
    atomic_json(directory / 'generation.json', pointer)
    print('Generation saved:', run, flush=True)
    return run


def run_path(directory, request):
    pointer = json.loads((directory / 'generation.json').read_text())
    if pointer['request_sha256'] != request_hash(request):
        raise ValueError('Generation pointer belongs to another evaluation request')
    relative = PurePosixPath(pointer['run_relative_path'])
    if relative.is_absolute() or '..' in relative.parts or relative.parts[:1] != ('runs',):
        raise ValueError('Invalid generation pointer')
    result = (directory / str(relative)).resolve()
    if not result.is_relative_to(directory.resolve()):
        raise ValueError('Generation pointer escapes evaluation directory')
    return result


def configure_incluster_kubectl(namespace, directory):
    host = os.environ['KUBERNETES_SERVICE_HOST']
    if ':' in host and not host.startswith('['):
        host = '[' + host + ']'
    credentials = '/var/run/secrets/kubernetes.io/serviceaccount/'
    config = {'apiVersion': 'v1', 'kind': 'Config',
              'clusters': [{'name': 'nrp', 'cluster': {
                  'server': 'https://' + host + ':' + os.environ.get('KUBERNETES_SERVICE_PORT_HTTPS', '443'),
                  'certificate-authority': credentials + 'ca.crt'}}],
              'users': [{'name': 'worker', 'user': {'tokenFile': credentials + 'token'}}],
              'contexts': [{'name': 'nrp', 'context': {'cluster': 'nrp', 'user': 'worker', 'namespace': namespace}}],
              'current-context': 'nrp'}
    path = directory / 'kubeconfig.json'
    atomic_json(path, config)
    os.environ['KUBECONFIG'] = str(path)


def harbor_environment_kwargs(request):
    from .suite import DOCKERFILE
    config = request['config']
    image_map = {hashlib.sha256(DOCKERFILE.encode()).hexdigest(): config['task_image'], **config['task_images']}
    uid = os.environ.get('MATH_EVAL_PARENT_UID')
    if not uid:
        raise RuntimeError('Harbor worker needs its parent Job UID for task-pod cleanup')
    return {'namespace': config['namespace'], 'image_map': image_map,
            'image_pull_secret': config['image_pull_secret'], 'evaluation_id': request['evaluation_id'],
            'owner_job': request['evaluation_id'] + '-grade', 'owner_uid': uid,
            'sandbox_timeout': config['sandbox_timeout']}


def worker_grade(directory, suite_mount=Path('/suites')):
    directory = Path(directory).resolve()
    request = read_request(directory)
    run = run_path(directory, request)
    if (run / 'grading-diagnostics.tgz').exists() or (run / 'summary.json').exists():
        raise FileExistsError('Grading already attempted; preserve this evaluation and use a new evaluation ID')
    with staged_suite(request, suite_mount) as suite, tempfile.TemporaryDirectory(prefix='math-eval-grading-') as temporary:
        scratch = Path(temporary) / 'run'
        scratch.mkdir()
        for name in ('run.json', 'predictions.jsonl'):
            shutil.copyfile(run / name, scratch / name)
        try:
            configure_incluster_kubectl(request['config']['namespace'], Path(temporary))
            summary = grade_run(suite, scratch, backend='harbor', concurrency=request['config']['grading_concurrency'],
                                environment_import='math_eval.harbor_nrp:NRPEnvironment',
                                environment_kwargs=harbor_environment_kwargs(request))
        finally:
            # Preserve diagnostics on both successful grading and verifier errors.
            # Archive thousands of small files before writing to shared storage.
            if (scratch / 'grading').exists():
                archive = Path(temporary) / 'grading-diagnostics.tgz'
                with tarfile.open(archive, 'w:gz') as packed:
                    packed.add(scratch / 'grading', arcname='grading')
                shutil.copyfile(archive, run / archive.name)
            for name in ('run.json', 'summary.json'):
                if (scratch / name).exists():
                    shutil.copyfile(scratch / name, run / name)
    report = {key: summary[key] for key in ('status', 'grading_backend', 'subset', 'protocol', 'benchmarks',
                                            'model', 'run_id', 'suite_sha256', 'generation_kind')}
    report['harbor_environment'] = summary['harbor_environment']
    report.update(evaluation_id=request['evaluation_id'], output_pvc=request['config']['output_pvc'],
                  run_relative_path=str(run.relative_to(directory)),
                  evaluation_relative_path=request['config']['output_relative_path'] + '/' + request['evaluation_id'])
    atomic_json(directory / 'report.json', report)
    with (directory / 'benchmarks.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['benchmark', 'accuracy', 'correct', 'expected', 'errors', 'truncated'])
        writer.writeheader()
        for benchmark, score in summary['benchmarks'].items():
            writer.writerow({'benchmark': benchmark, **{key: score[key] for key in writer.fieldnames[1:]}})
    print('Report saved:', directory / 'report.json', flush=True)
    return report


class KubernetesError(RuntimeError):
    def __init__(self, status, message):
        self.status = status
        super().__init__(f'Kubernetes API {status}: {message}')


class KubernetesJobs:
    """Use the projected service-account token and CA; no embedded kubeconfig."""
    def __init__(self, namespace):
        self.namespace = namespace
        host = os.environ['KUBERNETES_SERVICE_HOST']
        if ':' in host and not host.startswith('['):
            host = '[' + host + ']'
        port = os.environ.get('KUBERNETES_SERVICE_PORT_HTTPS', '443')
        self.base = 'https://' + host + ':' + port + '/apis/batch/v1/namespaces/' + quote(namespace, safe='') + '/jobs'
        self.credentials = Path('/var/run/secrets/kubernetes.io/serviceaccount')
        self.ssl = ssl.create_default_context(cafile=str(self.credentials / 'ca.crt'))

    def call(self, method, suffix='', body=None):
        for attempt in range(4):
            token = (self.credentials / 'token').read_text().strip()  # Reread rotated tokens.
            request = Request(self.base + suffix, method=method,
                              data=json.dumps(body).encode() if body is not None else None,
                              headers={'Authorization': 'Bearer ' + token, 'Content-Type': 'application/json'})
            try:
                with urlopen(request, context=self.ssl, timeout=30) as response:
                    return json.load(response)
            except HTTPError as error:
                if error.code in (429, 500, 502, 503, 504) and attempt < 3:
                    time.sleep(2 ** attempt)
                    continue
                detail = error.read().decode(errors='replace')[:1000]
                raise KubernetesError(error.code, detail) from error
            except (URLError, TimeoutError, OSError) as error:
                if attempt == 3:
                    raise RuntimeError(f'Kubernetes API connection failed: {type(error).__name__}') from error
                time.sleep(2 ** attempt)

    def get(self, name):
        return self.call('GET', '/' + quote(name, safe=''))

    def create(self, job):
        return self.call('POST', body=job)


def ensure_job(api, document):
    """A lost POST response or existing matching Job never launches a second GPU run."""
    name = document['metadata']['name']
    try:
        existing = api.get(name)
    except KubernetesError as error:
        if error.status != 404:
            raise
        try:
            return api.create(document)
        except KubernetesError as conflict:
            if conflict.status != 409:
                raise
            existing = api.get(name)
    expected = document['metadata']['annotations']['math-eval/request-sha256']
    if (existing['metadata'].get('annotations', {}).get('math-eval/request-sha256') != expected or
            existing['metadata'].get('labels', {}) != document['metadata']['labels']):
        raise ValueError('Existing Job does not match this evaluation request: ' + name)
    return existing


def wait_for_job(api, name, interval, sleep=time.sleep):
    while True:
        job = api.get(name)
        for condition in job.get('status', {}).get('conditions', []):
            if condition.get('status') != 'True':
                continue
            if condition['type'] == 'Failed':
                raise RuntimeError(f'{name} failed: {condition.get("reason", "")} {condition.get("message", "")}')
            if condition['type'] == 'Complete':
                return job
        sleep(interval)


def coordinate(request, output_mount=Path('/results'), api=None, sleep=time.sleep):
    request = validate_request(request)
    config = request['config']
    directory = (Path(output_mount) / config['output_relative_path'] / request['evaluation_id']).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    request_file = directory / 'request.json'
    if request_file.exists() and json.loads(request_file.read_text()) != request:
        raise ValueError('Evaluation directory already contains another request')
    if (directory / 'pipeline.json').exists():
        raise FileExistsError('This controller was already started; inspect its saved state or submit a new evaluation ID')
    atomic_json(request_file, request)
    state = {'evaluation_id': request['evaluation_id'], 'status': 'starting', 'grading_backend': 'harbor',
             'request_sha256': request_hash(request), 'jobs': {}}
    atomic_json(directory / 'pipeline.json', state)
    try:
        api = api or KubernetesJobs(config['namespace'])
        owner = api.get(request['evaluation_id'])['metadata']['uid']
        for role, phase in [('generate', 'generating'), ('grade', 'grading')]:
            document = job_for(request, role, owner_uid=owner)
            state.update(status=phase)
            state['jobs'][role] = document['metadata']['name']
            atomic_json(directory / 'pipeline.json', state)
            ensure_job(api, document)
            print('Waiting for ' + role + ' Job: ' + document['metadata']['name'], flush=True)
            wait_for_job(api, document['metadata']['name'], config['poll_seconds'], sleep)
            if role == 'generate':
                # Require the generation pointer before launching a grading worker.
                run_path(directory, request)
        report = json.loads((directory / 'report.json').read_text())
        if report['status'] != 'complete' or report['evaluation_id'] != request['evaluation_id']:
            raise ValueError('Grading finished without a complete matching report')
        state.update(status='complete', run_relative_path=report['run_relative_path'])
        atomic_json(directory / 'pipeline.json', state)
        for benchmark, score in report['benchmarks'].items():
            print(f"{benchmark}: {score['correct']}/{score['expected']} ({100 * score['accuracy']:.2f}%)", flush=True)
        print('Results directory: ' + str(directory), flush=True)
        print('MATH_EVAL_REPORT=' + json.dumps(report), flush=True)
        return report
    except BaseException as error:
        state.update(status='failed', error=f'{type(error).__name__}: {error}')
        atomic_json(directory / 'pipeline.json', state)
        raise
