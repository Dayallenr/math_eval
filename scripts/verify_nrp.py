#!/usr/bin/env python3
"""Verify a published runner in an NRP CPU Job; no GitLab runner or GPU needed."""

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import tarfile
from uuid import uuid4


ROOT = Path(__file__).resolve().parents[1]
VERIFY = r'''
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import tarfile
import unittest

import math_eval
import math_verify
import torch
import transformers

metadata = json.loads(Path('/verification/metadata.json').read_text())
print('Verifying source commit:', metadata['commit'], flush=True)
installed = Path(math_eval.__file__).resolve().parent
for name, expected in metadata['sources'].items():
    actual = hashlib.sha256((installed / name).read_bytes()).hexdigest()
    if actual != expected:
        raise RuntimeError('Image source does not match the selected commit: ' + name)
assert torch.version.cuda, 'Runner must contain a CUDA-enabled Torch build'
assert version('harbor') == '0.23.0', 'Unexpected Harbor version'
print('Versions:', json.dumps({name: version(name) for name in
      ('math-eval', 'harbor', 'torch', 'transformers', 'math-verify')}), flush=True)
print('Torch CUDA build:', torch.version.cuda, flush=True)
subprocess.run(['math-eval', 'nrp', '--help'], check=True)
subprocess.run(['kubectl', 'version', '--client'], check=True)
work = Path('/tmp/math-eval-verification')
work.mkdir()
with tarfile.open('/verification/tests.tar.gz', 'r:gz') as archive:
    archive.extractall(work, filter='data')
suite = unittest.defaultTestLoader.discover(str(work / 'tests'))
result = unittest.TextTestRunner(verbosity=2).run(suite)
if not result.testsRun or result.skipped or not result.wasSuccessful():
    raise RuntimeError('Verification failed, discovered no tests, or skipped a test')
print('VERIFICATION PASSED:', result.testsRun, 'tests; no skips', flush=True)
'''.lstrip()


def command(arguments, **kwargs):
    result = subprocess.run(arguments, capture_output=True, timeout=60, **kwargs)
    if result.returncode:
        error = result.stderr
        raise RuntimeError(error.decode() if isinstance(error, bytes) else error)
    return result.stdout


def kubectl(args, arguments, document=None):
    return command(['kubectl', '--context', args.context, '-n', args.namespace,
                    '--request-timeout=30s', *arguments], text=True,
                   input=json.dumps(document) if document is not None else None)


def documents(args):
    commit = command(['git', '-C', str(ROOT), 'rev-parse', '--verify', args.commit + '^{commit}'],
                     text=True).strip()
    snapshot = command(['git', '-C', str(ROOT), 'archive', '--format=tar', commit,
                        'src/math_eval', 'tests', 'tasks/aime-example/tests/grade.py'])
    sources = {}
    payload = io.BytesIO()
    with tarfile.open(fileobj=io.BytesIO(snapshot)) as original, \
            tarfile.open(fileobj=payload, mode='w:gz') as packed:
        for member in original.getmembers():
            if not member.isfile():
                continue
            data = original.extractfile(member).read()
            if member.name.startswith('src/math_eval/'):
                sources[member.name.removeprefix('src/math_eval/')] = hashlib.sha256(data).hexdigest()
            else:
                packed.addfile(member, io.BytesIO(data))
    if not sources or len(payload.getvalue()) > 700_000:
        raise ValueError('Missing source snapshot or test payload too large for a ConfigMap')
    metadata = {'commit': commit, 'image': args.image, 'sources': sources}
    configmap = {'apiVersion': 'v1', 'kind': 'ConfigMap',
                 'metadata': {'name': args.name, 'namespace': args.namespace},
                 'immutable': True,
                 'data': {'verify.py': VERIFY, 'metadata.json': json.dumps(metadata)},
                 'binaryData': {'tests.tar.gz': base64.b64encode(payload.getvalue()).decode()}}
    resources = {'cpu': '2', 'memory': '4Gi', 'ephemeral-storage': '25Gi'}
    pod = {
        'restartPolicy': 'Never', 'automountServiceAccountToken': False,
        'nodeSelector': {'kubernetes.io/arch': 'amd64'},
        'affinity': {'nodeAffinity': {'requiredDuringSchedulingIgnoredDuringExecution': {
            'nodeSelectorTerms': [{'matchExpressions': [{
                'key': 'feature.node.kubernetes.io/pci-10de.present',
                'operator': 'NotIn', 'values': ['true'],
            }]}],
        }}},
        'containers': [{
            'name': 'verify', 'image': args.image, 'imagePullPolicy': 'Always',
            'command': ['python', '/verification/verify.py'],
            'env': [{'name': name, 'value': value} for name, value in {
                'PYTHONUNBUFFERED': '1', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                'OMP_NUM_THREADS': '2', 'TOKENIZERS_PARALLELISM': 'false',
            }.items()],
            'resources': {'requests': resources.copy(), 'limits': resources.copy()},
            'volumeMounts': [{'name': 'verification', 'mountPath': '/verification', 'readOnly': True}],
        }],
        'volumes': [{'name': 'verification', 'configMap': {'name': args.name}}],
    }
    if args.pull_secret:
        pod['imagePullSecrets'] = [{'name': args.pull_secret}]
    job = {'apiVersion': 'batch/v1', 'kind': 'Job',
           'metadata': {'name': args.name, 'namespace': args.namespace,
                        'labels': {'app.kubernetes.io/name': 'math-eval-verification'},
                        'annotations': {'math-eval/source-commit': commit}},
           'spec': {'backoffLimit': 0, 'activeDeadlineSeconds': 3600,
                    'ttlSecondsAfterFinished': 604800,
                    'template': {'spec': pod}}}
    return job, configmap


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--context', default='nautilus')
    parser.add_argument('--namespace', default='wang-research-lab')
    commands = parser.add_subparsers(dest='action', required=True)
    submit = commands.add_parser('submit', help='Submit verification and return immediately')
    submit.add_argument('--image', required=True, help='Published image tag or digest')
    submit.add_argument('--commit', required=True, help='Source commit used to build this image')
    submit.add_argument('--pull-secret', default='nrp-registry-dayallen', help='Empty string for public images')
    submit.add_argument('--name', default='math-eval-verify-' + datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S') + '-' + uuid4().hex[:4])
    submit.add_argument('--dry-run', action='store_true', help='Print manifests without cluster access')
    for name in ('status', 'cleanup'):
        commands.add_parser(name).add_argument('name')
    args = parser.parse_args()
    try:
        if not re.fullmatch(r'math-eval-verify-[a-z0-9-]{1,45}', args.name):
            raise ValueError('Use a job name beginning math-eval-verify- with at most 62 characters')
        if args.action == 'submit':
            job, configmap = documents(args)
            if args.dry_run:
                print(json.dumps({'apiVersion': 'v1', 'kind': 'List', 'items': [configmap, job]}, indent=2))
                return
            created = json.loads(kubectl(args, ['create', '-f', '-', '-o', 'json'], job))
            configmap['metadata']['ownerReferences'] = [{
                'apiVersion': 'batch/v1', 'kind': 'Job', 'name': args.name,
                'uid': created['metadata']['uid'],
            }]
            try:
                kubectl(args, ['create', '-f', '-'], configmap)
            except (OSError, RuntimeError, subprocess.TimeoutExpired):
                kubectl(args, ['delete', 'job', args.name, '--wait=false'])
                raise
            print('Submitted CPU verification:', args.name)
            print('Check: python3 scripts/verify_nrp.py --context', args.context,
                  '--namespace', args.namespace, 'status', args.name)
        elif args.action == 'cleanup':
            job = json.loads(kubectl(args, ['get', 'job', args.name, '-o', 'json']))
            if job['metadata'].get('labels', {}).get('app.kubernetes.io/name') != 'math-eval-verification':
                raise ValueError('Refusing to delete a job not created by this verifier')
            print(kubectl(args, ['delete', 'job', args.name, '--wait=false']), end='')
        else:
            print(kubectl(args, ['get', 'job', args.name]), end='')
            pods = json.loads(kubectl(args, ['get', 'pods', '-l', 'job-name=' + args.name, '-o', 'json']))
            for pod in pods['items']:
                name = pod['metadata']['name']
                print('Pod:', name, 'Phase:', pod.get('status', {}).get('phase'))
                for container in pod.get('status', {}).get('containerStatuses', []):
                    print('Container:', container['name'], json.dumps(container['state']))
                    if container.get('imageID'):
                        print('Pulled image:', container['imageID'])
                    if 'running' in container['state'] or 'terminated' in container['state']:
                        print(kubectl(args, ['logs', name, '--tail=100']), end='')
                print(kubectl(args, ['get', 'events', '--field-selector',
                                    'involvedObject.name=' + name, '--sort-by=.lastTimestamp']), end='')
    except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as error:
        parser.exit(1, f'Error: {error}\n')


if __name__ == '__main__':
    main()
