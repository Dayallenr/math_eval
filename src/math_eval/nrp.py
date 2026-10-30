"""Submit self-contained NRP evaluations; the controller runs on the cluster."""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import subprocess
import tomllib
from uuid import uuid4

DEFAULTS = {
    'context': 'nautilus', 'namespace': 'wang-research-lab', 'image': '',
    'image_pull_secret': '', 'service_account': 'math-eval-controller',
    'task_image': '', 'task_images': {}, 'harbor_service_account': 'math-eval-harbor',
    'sandbox_timeout': 1800, 'model_pvc': '', 'suite_pvc': 'shared-pvc', 'output_pvc': 'shared-pvc',
    'output_relative_path': 'math-eval/evaluations',
    'suite_archive': 'math-eval/archives/math-seven-b43163a56bb3-pilot-b7a747d5908a.tgz',
    'archive_sha256': '208e3344aaba2a3e1be1fb6edd50eb8a28dd09a32f9213918148b277ecfd5643',
    'full_suite': 'math-seven',
    'full_sha256': 'b43163a56bb3944876ec01c781feedb8dcd4fcea74ca1d44491f17cd8c1ad121',
    'pilot_suite': 'pilot-seven',
    'pilot_sha256': 'b7a747d5908a0c2436c06c388fde4479816cb80bded0e292a0617123bccd88ce',
    'gpu_resource': 'nvidia.com/gpu', 'min_gpu_memory_mib': 40960, 'generation_timeout': 172800,
    'grading_timeout': 21600, 'poll_seconds': 10, 'grading_concurrency': 2,
    'settings': {'batch_size': 1, 'max_new_tokens': 12288, 'max_input_tokens': 8192,
                 'temperature': 0.7, 'top_p': 0.8, 'top_k': 20, 'min_p': 0.0, 'seed': 42,
                 'enable_thinking': False, 'device': 'cuda', 'dtype': 'auto'},
}


def relative_path(value):
    path = PurePosixPath(value)
    if not value or path.is_absolute() or '..' in path.parts or str(path) == '.':
        raise ValueError('Use a nonempty path relative to the PVC root, without ..')
    return str(path)


def dns_name(value, maximum=63):
    if not isinstance(value, str) or len(value) > maximum or not re.fullmatch(r'[a-z0-9]([-a-z0-9]*[a-z0-9])?', value):
        raise ValueError(f'Invalid Kubernetes name: {value!r}')
    return value


def validate_config(config):
    for key in ('namespace', 'service_account', 'harbor_service_account', 'suite_pvc', 'output_pvc'):
        dns_name(config[key])
    if config['model_pvc']:
        dns_name(config['model_pvc'])
    if config['image_pull_secret']:
        dns_name(config['image_pull_secret'])
    if not isinstance(config['image'], str) or '/' not in config['image'] or 'YOUR' in config['image'].upper():
        raise ValueError('Set image to the published evaluation image in your NRP config')
    if not isinstance(config['task_image'], str) or '/' not in config['task_image'] or 'YOUR' in config['task_image'].upper():
        raise ValueError('Set task_image to the published math task image')
    if not isinstance(config['task_images'], dict) or any(not re.fullmatch(r'[a-f0-9]{64}', k) or not isinstance(v, str) or '/' not in v for k,v in config['task_images'].items()):
        raise ValueError('task_images must map Dockerfile SHA256 values to image URLs')
    for key in ('output_relative_path', 'suite_archive', 'full_suite', 'pilot_suite'):
        config[key] = relative_path(config[key])
    for key in ('archive_sha256', 'full_sha256', 'pilot_sha256'):
        if not re.fullmatch(r'[a-f0-9]{64}', config[key]):
            raise ValueError(f'{key} must be a SHA-256 fingerprint')
    for key in ('generation_timeout', 'grading_timeout', 'poll_seconds', 'grading_concurrency', 'sandbox_timeout', 'min_gpu_memory_mib'):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if not isinstance(config['gpu_resource'], str) or not re.fullmatch(r'nvidia\.com/[a-z0-9][a-z0-9._-]*', config['gpu_resource']):
        raise ValueError('gpu_resource must be an NRP NVIDIA extended resource, such as nvidia.com/a100')
    settings = config['settings']
    for key in ('batch_size', 'max_new_tokens', 'max_input_tokens'):
        if type(settings[key]) is not int or settings[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if not all(math.isfinite(settings[k]) for k in ('temperature', 'top_p')) or settings['temperature'] < 0 or not 0 < settings['top_p'] <= 1:
        raise ValueError('Invalid sampling settings')
    if type(settings['top_k']) is not int or settings['top_k'] < 0:
        raise ValueError('top_k must be a nonnegative integer')
    if not math.isfinite(settings['min_p']) or not 0 <= settings['min_p'] <= 1:
        raise ValueError('min_p must be in [0,1]')
    if type(settings['seed']) is not int or type(settings['enable_thinking']) is not bool:
        raise ValueError('seed must be an integer and enable_thinking must be a boolean')
    if settings['device'] != 'cuda' or settings['dtype'] not in ('auto', 'bfloat16', 'float16', 'float32'):
        raise ValueError('NRP generation requires CUDA and a supported dtype')
    return config


def load_config(path):
    supplied = tomllib.loads(Path(path).read_text())
    if set(supplied) - set(DEFAULTS):
        raise ValueError(f'Unknown NRP settings: {sorted(set(supplied) - set(DEFAULTS))}')
    config = deepcopy(DEFAULTS)
    settings = supplied.pop('settings', {})
    if set(settings) - set(config['settings']):
        raise ValueError('Unknown generation setting')
    config.update(supplied)
    config['settings'].update(settings)
    return validate_config(config)


def request_for(config, model, pilot=False, name=None):
    config = validate_config(deepcopy(config))
    if ':' in model:
        pvc, path = model.split(':', 1)
    else:
        pvc, path = config['model_pvc'], model
    if not pvc:
        raise ValueError('Supply --model PVC:relative/export/path or set model_pvc in nrp.toml')
    dns_name(pvc)
    path = relative_path(path)
    name = name or 'math-eval-' + datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S') + '-' + uuid4().hex[:6]
    dns_name(name, maximum=48)
    return {'schema_version': 1, 'evaluation_id': name, 'config': config,
            'model_pvc': pvc, 'model_relative_path': path, 'pilot': bool(pilot)}


def request_hash(request):
    return hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()


def evaluation_path(request):
    return '/results/' + request['config']['output_relative_path'] + '/' + dns_name(request['evaluation_id'], 48)


def job_for(request, role, owner_uid=None):
    config, name = request['config'], request['evaluation_id']
    if role not in ('controller', 'generate', 'grade'):
        raise ValueError('Unknown evaluation role')
    job_name = name if role == 'controller' else name + '-' + role
    metadata = {'name': job_name, 'namespace': config['namespace'],
                'labels': {'app.kubernetes.io/name': 'math-eval', 'math-eval/evaluation': name,
                           'math-eval/role': role},
                'annotations': {'math-eval/request-sha256': request_hash(request)}}
    if owner_uid:
        metadata['ownerReferences'] = [{'apiVersion': 'batch/v1', 'kind': 'Job', 'name': name,
                                       'uid': owner_uid, 'controller': False}]
    volumes = [{'name': 'output', 'persistentVolumeClaim': {'claimName': config['output_pvc']}}]
    mounts = [{'name': 'output', 'mountPath': '/results'}]
    if role != 'controller':
        volumes.append({'name': 'suite', 'persistentVolumeClaim': {'claimName': config['suite_pvc'], 'readOnly': True}})
        mounts.append({'name': 'suite', 'mountPath': '/suites', 'readOnly': True})
    if role == 'generate':
        volumes.append({'name': 'model', 'persistentVolumeClaim': {'claimName': request['model_pvc'], 'readOnly': True}})
        mounts.append({'name': 'model', 'mountPath': '/models', 'readOnly': True})
        resources = {config['gpu_resource']: 1, 'cpu': '4', 'memory': '48Gi', 'ephemeral-storage': '20Gi'}
        timeout = config['generation_timeout']
    elif role == 'grade':
        resources = {'cpu': str(config['grading_concurrency']), 'memory': '4Gi', 'ephemeral-storage': '4Gi'}
        timeout = config['grading_timeout']
    else:
        resources = {'cpu': '1', 'memory': '512Mi', 'ephemeral-storage': '1Gi'}
        timeout = config['generation_timeout'] + config['grading_timeout'] + 600
    args = (['nrp-controller', '--request-json', json.dumps(request)] if role == 'controller'
            else ['nrp-worker', role, '--evaluation-dir', evaluation_path(request)])
    pod = {'restartPolicy': 'Never', 'automountServiceAccountToken': role in ('controller', 'grade'),
           'containers': [{'name': role, 'image': config['image'], 'args': args,
                           'resources': {'requests': resources, 'limits': resources},
                           'volumeMounts': mounts}], 'volumes': volumes}
    if role == 'controller':
        pod['serviceAccountName'] = config['service_account']
    if role == 'grade':
        pod['serviceAccountName'] = config['harbor_service_account']
        pod['containers'][0]['env'] = [{'name': 'MATH_EVAL_PARENT_UID', 'valueFrom': {'fieldRef': {'fieldPath': "metadata.labels['batch.kubernetes.io/controller-uid']"}}}]
    if role == 'generate':
        pod['nodeSelector'] = {'kubernetes.io/arch': 'amd64'}
        pod['affinity'] = {'nodeAffinity': {'requiredDuringSchedulingIgnoredDuringExecution': {
            'nodeSelectorTerms': [{'matchExpressions': [{'key': 'nvidia.com/gpu.memory',
                'operator': 'Gt', 'values': [str(config['min_gpu_memory_mib'])]}]}]}}}
    else:
        pod['affinity'] = {'nodeAffinity': {'requiredDuringSchedulingIgnoredDuringExecution': {
            'nodeSelectorTerms': [{'matchExpressions': [{'key': 'feature.node.kubernetes.io/pci-10de.present',
                                                         'operator': 'NotIn', 'values': ['true']}]}]}}}
    if config['image_pull_secret']:
        pod['imagePullSecrets'] = [{'name': config['image_pull_secret']}]
    return {'apiVersion': 'batch/v1', 'kind': 'Job', 'metadata': metadata,
            'spec': {'backoffLimit': 0, 'activeDeadlineSeconds': timeout,
                     'ttlSecondsAfterFinished': 604800,
                     'template': {'metadata': {'labels': metadata['labels']}, 'spec': pod}}}


def rbac_for(config):
    namespace, account = config['namespace'], config['service_account']
    return {'apiVersion': 'v1', 'kind': 'List', 'items': [
        {'apiVersion': 'v1', 'kind': 'ServiceAccount', 'metadata': {'name': account, 'namespace': namespace}},
        {'apiVersion': 'rbac.authorization.k8s.io/v1', 'kind': 'Role',
         'metadata': {'name': account, 'namespace': namespace},
         'rules': [{'apiGroups': ['batch'], 'resources': ['jobs'], 'verbs': ['create', 'get']}]},
        {'apiVersion': 'rbac.authorization.k8s.io/v1', 'kind': 'RoleBinding',
         'metadata': {'name': account, 'namespace': namespace},
         'roleRef': {'apiGroup': 'rbac.authorization.k8s.io', 'kind': 'Role', 'name': account},
         'subjects': [{'kind': 'ServiceAccount', 'name': account, 'namespace': namespace}]},
    ] + [
        {'apiVersion': 'v1', 'kind': 'ServiceAccount', 'metadata': {'name': config['harbor_service_account'], 'namespace': namespace}},
        {'apiVersion': 'rbac.authorization.k8s.io/v1', 'kind': 'Role',
         'metadata': {'name': config['harbor_service_account'], 'namespace': namespace},
         'rules': [{'apiGroups': [''], 'resources': ['pods'], 'verbs': ['create', 'get', 'delete']},
                   {'apiGroups': [''], 'resources': ['pods/exec'], 'verbs': ['create', 'get']}]},
        {'apiVersion': 'rbac.authorization.k8s.io/v1', 'kind': 'RoleBinding',
         'metadata': {'name': config['harbor_service_account'], 'namespace': namespace},
         'roleRef': {'apiGroup': 'rbac.authorization.k8s.io', 'kind': 'Role', 'name': config['harbor_service_account']},
         'subjects': [{'kind': 'ServiceAccount', 'name': config['harbor_service_account'], 'namespace': namespace}]},
    ]}


def kubectl(config, arguments, document=None):
    command = ['kubectl', '--context', config['context'], '--namespace', config['namespace'], '--request-timeout=30s', *arguments]
    try:
        result = subprocess.run(command, input=json.dumps(document) if document is not None else None,
                                text=True, capture_output=True, timeout=45)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError(f'kubectl failed: {error}') from error
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return result.stdout


def preflight(request):
    config = request['config']
    names = list(dict.fromkeys([request['model_pvc'], config['suite_pvc'], config['output_pvc']]))
    claims = json.loads(kubectl(config, ['get', 'pvc', *names, '-o', 'json']))
    items = claims.get('items', [claims])
    by_name = {item['metadata']['name']: item for item in items}
    for name in names:
        if name not in by_name or by_name[name].get('status', {}).get('phase') != 'Bound':
            raise ValueError(f'PVC {name} must exist and be Bound in {config["namespace"]}')
    if 'ReadWriteMany' not in by_name[config['output_pvc']]['spec']['accessModes']:
        raise ValueError('The output PVC must support ReadWriteMany: controller and workers mount it together')
    kubectl(config, ['get', 'serviceaccount', config['service_account'], config['harbor_service_account'], '-o', 'name'])


def add_parser(subparsers):
    nrp = subparsers.add_parser('nrp', help='Submit and inspect automated NRP evaluations')
    nrp.add_argument('--config', type=Path, default=Path('nrp.toml'))
    commands = nrp.add_subparsers(dest='nrp_command', required=True)
    setup = commands.add_parser('setup', help='Install controller and Harbor ServiceAccounts and namespace roles once')
    setup.add_argument('--dry-run', action='store_true')
    run = commands.add_parser('run', help='Submit a controller; generation and grading run on NRP')
    run.add_argument('--model', required=True, help='PVC:relative/export/path, or relative path with model_pvc configured')
    run.add_argument('--pilot', action='store_true')
    run.add_argument('--name')
    run.add_argument('--dry-run', action='store_true')
    for verb in ('status', 'results', 'cancel'):
        command = commands.add_parser(verb)
        command.add_argument('evaluation_id')
    return nrp


def handle(args):
    config = load_config(args.config)
    if args.nrp_command == 'setup':
        objects = rbac_for(config)
        if args.dry_run:
            print(json.dumps(objects, indent=2))
        else:
            print(kubectl(config, ['apply', '-f', '-'], objects).strip())
        return
    if args.nrp_command == 'run':
        request = request_for(config, args.model, args.pilot, args.name)
        job = job_for(request, 'controller')
        if args.dry_run:
            print(json.dumps({'apiVersion': 'v1', 'kind': 'List', 'items': [job, job_for(request, 'generate'), job_for(request, 'grade')]}, indent=2))
            return
        preflight(request)
        kubectl(config, ['create', '--dry-run=server', '-f', '-'], job)
        print(kubectl(config, ['create', '-f', '-'], job).strip())
        print('Evaluation ID:', request['evaluation_id'])
        print('Output PVC:', config['output_pvc'])
        print('Output directory relative to PVC:', config['output_relative_path'] + '/' + request['evaluation_id'])
        print('The controller runs on NRP; this terminal can be closed.')
        return
    name = dns_name(args.evaluation_id, maximum=48)
    if args.nrp_command == 'status':
        print(kubectl(config, ['get', 'jobs', '-l', 'math-eval/evaluation=' + name]).strip())
    elif args.nrp_command == 'cancel':
        print(kubectl(config, ['delete', 'job', name, '--cascade=foreground', '--wait=false']).strip())
    else:
        logs = kubectl(config, ['logs', 'job/' + name, '-c', 'controller'])
        reports = [line.removeprefix('MATH_EVAL_REPORT=') for line in logs.splitlines() if line.startswith('MATH_EVAL_REPORT=')]
        if not reports:
            raise RuntimeError('No completed report yet. Inspect nrp status and the controller/worker logs.')
        print(json.dumps(json.loads(reports[-1]), indent=2))
