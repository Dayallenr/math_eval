"""Coordinate NRP generation and Harbor grading with the laptop's kubectl login."""

from contextlib import contextmanager
from copy import deepcopy
import csv
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import time
from uuid import uuid4

from . import nrp
from .nrp_workflow import atomic_json, request_hash, staged_suite, validate_request
from .results import grade_run, validate_predictions
from .suite import DOCKERFILE, load_suite


TRANSFER = Path(__file__).with_name('_transfer.py').read_text()


def reuse_pvc_volumes(pod):
    """Mount each claim once, preserving read-only access at each mount path."""
    claims, aliases, readonly, volumes = {}, {}, {}, []
    for volume in pod.get('volumes', []):
        source = volume.get('persistentVolumeClaim')
        if source is None:
            volumes.append(volume)
            continue
        name, claim = volume['name'], source['claimName']
        readonly[name] = source.get('readOnly', False)
        if claim in claims:
            canonical = claims[claim]
            aliases[name] = canonical['name']
            canonical['persistentVolumeClaim']['readOnly'] = (
                canonical['persistentVolumeClaim'].get('readOnly', False) and readonly[name])
        else:
            claims[claim] = volume
            volumes.append(volume)
    for container in pod.get('containers', []) + pod.get('initContainers', []):
        for mount in container.get('volumeMounts', []):
            if readonly.get(mount['name']):
                mount['readOnly'] = True
            mount['name'] = aliases.get(mount['name'], mount['name'])
    pod['volumes'] = volumes
    return pod


def generation_document(request):
    document = nrp.job_for(request, 'generate')
    reuse_pvc_volumes(document['spec']['template']['spec'])
    return document


class UserCluster:
    """All API operations use the user's kubeconfig; credentials never enter pods."""

    def __init__(self, config):
        self.config = config

    def call(self, arguments, document=None, timeout=60):
        command = ['kubectl', '--context', self.config['context'], '-n', self.config['namespace'],
                   '--request-timeout=' + ('10m' if timeout > 60 else '30s'), *arguments]
        result = subprocess.run(command, input=json.dumps(document) if document is not None else None,
                                capture_output=True, text=True, timeout=timeout)
        if result.returncode:
            raise RuntimeError(result.stderr.strip() or result.stdout.strip())
        return result.stdout

    def generation_job(self, request):
        document = generation_document(request)
        name = document['metadata']['name']
        existing = self.call(['get', 'job', name, '--ignore-not-found', '-o', 'json'])
        if not existing:
            try:
                return json.loads(self.call(['create', '-f', '-', '-o', 'json'], document))
            except (RuntimeError, subprocess.TimeoutExpired):
                # A lost create response must not launch a second GPU job.
                existing = self.call(['get', 'job', name, '--ignore-not-found', '-o', 'json'])
                if not existing:
                    raise
        existing = json.loads(existing)
        if (existing['metadata'].get('annotations', {}).get('math-eval/request-sha256') != request_hash(request) or
                existing['metadata'].get('labels') != document['metadata']['labels']):
            raise ValueError('Existing generation job belongs to a different request')
        return existing


def transfer_pod(request, phase, name=None):
    config = request['config']
    name = name or request['evaluation_id'] + '-xfer-' + uuid4().hex[:6]
    mounts = [{'name': 'output', 'mountPath': '/results', 'readOnly': phase == 'download'}]
    volumes = [{'name': 'output', 'persistentVolumeClaim': {'claimName': config['output_pvc'],
                                                           'readOnly': phase == 'download'}}]
    if phase == 'stage':
        mounts.append({'name': 'suite', 'mountPath': '/suites', 'readOnly': True})
        volumes.append({'name': 'suite', 'persistentVolumeClaim': {'claimName': config['suite_pvc'], 'readOnly': True}})
    resources = {'cpu': '1', 'memory': '512Mi', 'ephemeral-storage': '2Gi'}
    pod = {'restartPolicy': 'Never', 'automountServiceAccountToken': False,
           'activeDeadlineSeconds': 1800, 'nodeSelector': {'kubernetes.io/arch': 'amd64'},
           'affinity': {'nodeAffinity': {'requiredDuringSchedulingIgnoredDuringExecution': {
               'nodeSelectorTerms': [{'matchExpressions': [{
                   'key': 'feature.node.kubernetes.io/pci-10de.present', 'operator': 'NotIn', 'values': ['true'],
               }]}]}}},
           'containers': [{'name': 'transfer', 'image': config['task_image'],
                           'command': ['python', '-u', '-c', TRANSFER, json.dumps(request), phase],
                           'resources': {'requests': resources, 'limits': resources.copy()},
                           'volumeMounts': mounts}], 'volumes': volumes}
    if config['image_pull_secret']:
        pod['imagePullSecrets'] = [{'name': config['image_pull_secret']}]
    reuse_pvc_volumes(pod)
    return {'apiVersion': 'v1', 'kind': 'Pod', 'metadata': {
        'name': name, 'namespace': config['namespace'],
        'labels': {'app.kubernetes.io/name': 'math-eval', 'math-eval/evaluation': request['evaluation_id'],
                   'math-eval/role': 'laptop-transfer'},
        'annotations': {'math-eval/request-sha256': request_hash(request)},
    }, 'spec': pod}


@contextmanager
def transfer(cluster, request, phase, sleep=time.sleep):
    document = transfer_pod(request, phase)
    name = document['metadata']['name']
    cluster.call(['create', '-f', '-'], document)
    try:
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline:
            pod = json.loads(cluster.call(['get', 'pod', name, '-o', 'json']))
            status = pod.get('status', {})
            if status.get('phase') in ('Failed', 'Succeeded'):
                raise RuntimeError('Transfer stopped: ' + cluster.call(['logs', name]))
            for container in status.get('containerStatuses', []):
                waiting = container.get('state', {}).get('waiting', {})
                if waiting.get('reason') in ('ErrImagePull', 'ImagePullBackOff'):
                    raise RuntimeError('Transfer image pull failed: ' + waiting.get('message', waiting['reason']))
            if status.get('phase') == 'Running':
                ready = cluster.call(['exec', name, '--', 'python', '-c',
                                      'from pathlib import Path; print(Path("/tmp/transfer/ready").exists())'])
                if ready.strip() == 'True':
                    break
            sleep(2)
        else:
            raise TimeoutError('Transfer pod did not become ready in fifteen minutes')
        yield name
    finally:
        try:
            cluster.call(['delete', 'pod', name, '--ignore-not-found', '--wait=false'])
        except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
            print(f'Could not remove {name}: {error}. It has a thirty-minute deadline.', flush=True)


@contextmanager
def locked(directory):
    with (directory / '.coordinator.lock').open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('Another laptop process is coordinating this evaluation') from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def install_suite(request, directory):
    destination = directory / 'suite'
    expected = request['config']['pilot_sha256'] if request['pilot'] else request['config']['full_sha256']
    if destination.exists():
        if load_suite(destination)[2] != expected:
            raise ValueError('Preserve the modified local suite; its fingerprint differs')
        return destination
    with staged_suite(request, directory / 'suite-volume') as suite:
        staging = directory / ('.suite-' + uuid4().hex[:8])
        shutil.copytree(suite, staging)
        staging.rename(destination)
    return destination


def fetch_generation(cluster, request, directory):
    packed = directory / 'generation.tgz'
    with transfer(cluster, request, 'download') as pod:
        cluster.call(['cp', pod + ':/tmp/transfer/generation.tgz', str(packed)], timeout=660)
    with tarfile.open(packed, 'r:gz') as archive:
        members = archive.getmembers()
        if {m.name for m in members} != {'generation.json', 'run.json', 'predictions.jsonl'} or len(members) != 3 or any(not m.isfile() for m in members):
            raise ValueError('Unexpected generation archive contents')
        pointer = json.load(archive.extractfile('generation.json'))
        if pointer['request_sha256'] != request_hash(request):
            raise ValueError('Downloaded generation belongs to another request')
        parts = Path(nrp.relative_path(pointer['run_relative_path'])).parts
        if len(parts) != 2 or parts[0] != 'runs':
            raise ValueError('Invalid generation pointer')
        run = directory / 'run'
        run.mkdir()
        for name in ('run.json', 'predictions.jsonl'):
            (run / name).write_bytes(archive.extractfile(name).read())
    atomic_json(directory / 'generation.json', pointer)
    return run


def finish_report(request, directory, summary):
    pointer = json.loads((directory / 'generation.json').read_text())
    report = {key: summary[key] for key in ('status', 'grading_backend', 'subset', 'protocol', 'benchmarks',
                                          'model', 'run_id', 'suite_sha256', 'generation_kind')}
    report.update(evaluation_id=request['evaluation_id'], request_sha256=request_hash(request),
                  coordinator='laptop', harbor_environment='math_eval.harbor_nrp:NRPEnvironment',
                  output_pvc=request['config']['output_pvc'], run_relative_path=pointer['run_relative_path'],
                  evaluation_relative_path=request['config']['output_relative_path'] + '/' + request['evaluation_id'])
    atomic_json(directory / 'report.json', report)
    with (directory / 'benchmarks.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['benchmark', 'accuracy', 'correct', 'expected', 'errors', 'truncated'])
        writer.writeheader()
        for name, score in report['benchmarks'].items():
            writer.writerow({'benchmark': name, **{key: score[key] for key in writer.fieldnames[1:]}})
    return report


def upload_results(cluster, request, directory, completed):
    run = directory / 'run'
    diagnostics = run / 'grading-diagnostics.tgz'
    if not diagnostics.exists():
        with tarfile.open(diagnostics, 'w:gz') as archive:
            archive.add(run / 'grading', arcname='grading')
    packed = directory / 'upload.tgz'
    atomic_json(directory / 'upload-pipeline.json', completed)
    with tarfile.open(packed, 'w:gz') as archive:
        for root, names in ((run, ('run.json', 'summary.json', 'grading-diagnostics.tgz')),
                            (directory, ('report.json', 'benchmarks.csv'))):
            for name in names:
                archive.add(root / name, arcname=name)
        archive.add(directory / 'upload-pipeline.json', arcname='pipeline.json')
    with transfer(cluster, request, 'upload') as pod:
        cluster.call(['cp', str(packed), pod + ':/tmp/transfer/upload.tgz'], timeout=660)
        result = cluster.call(['exec', pod, '--', 'python', '-u', '-c', TRANSFER, json.dumps(request), 'apply'], timeout=660)
        if 'MATH_EVAL_UPLOAD_COMPLETE' not in result:
            raise RuntimeError('The result upload did not confirm completion')


def coordinate(request, directory, cluster=None, sleep=time.sleep, grader=grade_run, retry_grading=False):
    request = validate_request(deepcopy(request))
    config = request['config']
    cluster = cluster or UserCluster(config)
    state_file = directory / 'pipeline.json'
    state = json.loads(state_file.read_text()) if state_file.exists() else {
        'evaluation_id': request['evaluation_id'], 'request_sha256': request_hash(request),
        'coordinator': 'laptop', 'status': 'starting', 'uploaded': False,
    }
    if state['request_sha256'] != request_hash(request):
        raise ValueError('Local evaluation belongs to another request')
    if state.get('status') == 'cancelled':
        raise RuntimeError('This evaluation was cancelled. Preserve it and choose a new evaluation ID.')
    def phase(name):
        state.update(status=name)
        state.pop('error', None)
        atomic_json(state_file, state)
        print('Evaluation:', request['evaluation_id'], 'Phase:', name, flush=True)
    try:
        if state.get('uploaded'):
            return json.loads((directory / 'report.json').read_text())
        if not (directory / 'staged.json').exists():
            phase('staging')
            with transfer(cluster, request, 'stage', sleep=sleep) as pod:
                archive = directory / 'suite-volume' / config['suite_archive']
                archive.parent.mkdir(parents=True, exist_ok=True)
                cluster.call(['cp', pod + ':/tmp/transfer/suite.tgz', str(archive)], timeout=660)
            install_suite(request, directory)
            atomic_json(directory / 'staged.json', {'request_sha256': request_hash(request)})
        suite = install_suite(request, directory)
        run = directory / 'run'
        if not (directory / 'generation.json').exists():
            phase('generating')
            cluster.generation_job(request)
            name = request['evaluation_id'] + '-generate'
            while True:
                job = json.loads(cluster.call(['get', 'job', name, '-o', 'json']))
                if job['metadata'].get('annotations', {}).get('math-eval/request-sha256') != request_hash(request):
                    raise ValueError('Generation job changed identity')
                conditions = job.get('status', {}).get('conditions', [])
                failed = next((c for c in conditions if c['type'] == 'Failed' and c['status'] == 'True'), None)
                if failed:
                    raise RuntimeError('GPU generation failed: ' + json.dumps(failed))
                if any(c['type'] == 'Complete' and c['status'] == 'True' for c in conditions):
                    break
                sleep(config['poll_seconds'])
            # A Complete Job has released its GPU before Harbor begins.
            phase('fetching')
            if run.exists():
                # Preserve partial downloads while a new copy is fetched.
                run.rename(directory / ('incomplete-download-' + uuid4().hex[:8]))
            fetch_generation(cluster, request, directory)
        validate_predictions(suite, run)
        summary_file = run / 'summary.json'
        summary = json.loads(summary_file.read_text()) if summary_file.exists() else None
        if not summary or summary.get('status') != 'complete':
            if (run / 'grading').exists():
                if not retry_grading:
                    raise RuntimeError('Interrupted/failed grading is preserved. Resume with --retry-grading to retry saved answers.')
                attempts = run / ('grading-attempt-' + uuid4().hex[:8])
                attempts.mkdir()
                for name in ('grading', 'summary.json', 'grading-diagnostics.tgz'):
                    if (run / name).exists():
                        (run / name).rename(attempts / name)
                cluster.call(['delete', 'pods', '-l', 'math-eval/role=harbor-task,math-eval/evaluation=' + request['evaluation_id'],
                              '--ignore-not-found', '--wait=false'])
            phase('grading')
            summary = grader(suite, run, backend='harbor', concurrency=config['grading_concurrency'],
                environment_import='math_eval.harbor_nrp:NRPEnvironment', environment_kwargs={
                    'namespace': config['namespace'], 'context': config['context'],
                    'image_map': {hashlib.sha256(DOCKERFILE.encode()).hexdigest(): config['task_image'], **config['task_images']},
                    'image_pull_secret': config['image_pull_secret'], 'evaluation_id': request['evaluation_id'],
                    'sandbox_timeout': config['sandbox_timeout'],
                })
        if summary['status'] != 'complete':
            raise RuntimeError('Incomplete grading cannot be uploaded as a completed report')
        report = finish_report(request, directory, summary)
        phase('uploading')
        # Never mark local state complete until the remote upload is confirmed.
        completed = {**state, 'status': 'complete', 'uploaded': True}
        upload_results(cluster, request, directory, completed)
        state.update(completed)
        atomic_json(state_file, state)
        print('Results:', directory / 'report.json', flush=True)
        print('MATH_EVAL_REPORT=' + json.dumps(report), flush=True)
        return report
    except BaseException as error:
        state.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                     error=f'{type(error).__name__}: {error}')
        atomic_json(state_file, state)
        print('Saved local state. Resume this evaluation ID; do not submit a new GPU run to reconnect.', flush=True)
        raise


def add_parser(subparsers):
    parser = subparsers.add_parser('nrp-laptop', help='Laptop coordinates NRP jobs and Harbor; no new service-account roles')
    parser.add_argument('--config', type=Path, default=Path('nrp.toml'))
    parser.add_argument('--workspace', type=Path, default=Path('runs/nrp-laptop'))
    commands = parser.add_subparsers(dest='laptop_command', required=True)
    run = commands.add_parser('run')
    run.add_argument('--model', required=True, help='PVC:relative/export/path')
    run.add_argument('--pilot', action='store_true')
    run.add_argument('--name')
    run.add_argument('--dry-run', action='store_true')
    resume = commands.add_parser('resume')
    resume.add_argument('evaluation_id')
    resume.add_argument('--retry-grading', action='store_true')
    for name in ('status', 'results', 'cancel'):
        commands.add_parser(name).add_argument('evaluation_id')


def handle(args):
    if args.laptop_command == 'run':
        config = nrp.load_config(args.config)
        request = nrp.request_for(config, args.model, args.pilot, args.name)
        if args.dry_run:
            print(json.dumps({'coordinator': 'laptop', 'items': [transfer_pod(request, 'stage'),
                generation_document(request), transfer_pod(request, 'download'), transfer_pod(request, 'upload')],
                'grading': 'Harbor runs locally with isolated NRP task pods'}, indent=2))
            return
        nrp.preflight(request, require_accounts=False)
        directory = args.workspace.resolve() / request['evaluation_id']
        directory.mkdir(parents=True, exist_ok=False)
        atomic_json(directory / 'request.json', request)
    else:
        nrp.dns_name(args.evaluation_id, maximum=48)
        directory = args.workspace.resolve() / args.evaluation_id
        request = validate_request(json.loads((directory / 'request.json').read_text()))
        config = request['config']  # Resume the saved request, not a silently changed profile.
    cluster = UserCluster(config)
    if args.laptop_command == 'status':
        print((directory / 'pipeline.json').read_text() if (directory / 'pipeline.json').exists() else 'Not started')
        print(cluster.call(['get', 'jobs', '-l', 'math-eval/evaluation=' + request['evaluation_id']]))
        return
    if args.laptop_command == 'results':
        state = json.loads((directory / 'pipeline.json').read_text())
        if not state.get('uploaded'):
            raise RuntimeError('Results are not yet uploaded; inspect status or resume the evaluation')
        print((directory / 'report.json').read_text())
        return
    with locked(directory):
        if args.laptop_command == 'cancel':
            name = request['evaluation_id'] + '-generate'
            job = cluster.call(['get', 'job', name, '--ignore-not-found', '-o', 'json'])
            if job:
                job = json.loads(job)
                if job['metadata'].get('annotations', {}).get('math-eval/request-sha256') != request_hash(request):
                    raise ValueError('Refusing to cancel another request\'s job')
                print(cluster.call(['delete', 'job', name, '--wait=false']))
            selector = 'app.kubernetes.io/name=math-eval,math-eval/evaluation=' + request['evaluation_id']
            print(cluster.call(['delete', 'pods', '-l', selector, '--wait=false']))
            state_file = directory / 'pipeline.json'
            state = json.loads(state_file.read_text()) if state_file.exists() else {}
            if not state.get('uploaded'):
                atomic_json(state_file, {'status': 'cancelled', 'request_sha256': request_hash(request), 'uploaded': False})
            return
        coordinate(request, directory, cluster=cluster, retry_grading=getattr(args, 'retry_grading', False))
