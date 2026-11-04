#!/usr/bin/env python3
"""Prepare a CPU-only NRP tmux host using the user's interactive OIDC login.

No user tokens are copied. Only run state is kept on the output PVC. The host
runs as a Deployment, following NRP's policy for interactive/idle CPU pods.
"""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import tomllib

ROOT = Path(__file__).resolve().parents[1]
LABEL = 'math-eval/remote-coordinator'


def call(command, **kwargs):
    return subprocess.check_output(command, text=True, **kwargs)


def public_login(config):
    """Whitelist public OIDC settings; never copy cached tokens or client secrets."""
    source = json.loads(call(['kubectl', '--context', config['context'], 'config', 'view',
                              '--minify', '--flatten', '--raw', '-o', 'json']))
    existing = source['users'][0]['user'].get('exec', {})
    if existing.get('command') != 'kubectl' or existing.get('args', [])[:2] != ['oidc-login', 'get-token']:
        raise ValueError('Expected the standard NRP kubectl oidc-login configuration')
    prefixes = ('--oidc-issuer-url=', '--oidc-client-id=', '--oidc-extra-scope=')
    args = [arg for arg in existing['args'][2:] if arg.startswith(prefixes)]
    if not any(a.startswith('--oidc-issuer-url=') for a in args) or not any(a.startswith('--oidc-client-id=') for a in args):
        raise ValueError('Missing public OIDC issuer/client ID settings')
    args += ['--grant-type=device-code', '--skip-open-browser', '--token-cache-storage=disk']
    cluster = source['clusters'][0]['cluster']
    cluster = {key: cluster[key] for key in ('server', 'certificate-authority-data') if key in cluster}
    if not cluster.get('server', '').startswith('https://') or 'certificate-authority-data' not in cluster:
        raise ValueError('Expected HTTPS cluster with embedded CA certificate')
    context = config['context']
    return {'apiVersion': 'v1', 'kind': 'Config', 'current-context': context,
            'clusters': [{'name': 'nrp', 'cluster': cluster}],
            'contexts': [{'name': context, 'context': {'cluster': 'nrp', 'user': 'oidc', 'namespace': config['namespace']}}],
            'users': [{'name': 'oidc', 'user': {'exec': {'apiVersion': 'client.authentication.k8s.io/v1beta1',
                       'command': 'kubectl', 'args': ['oidc-login', 'get-token', *args], 'interactiveMode': 'IfAvailable'}}}]}


def documents(config, name, login, payload):
    labels = {LABEL: name, 'app.kubernetes.io/name': 'math-eval'}
    resources = {'cpu': '1', 'memory': '4Gi', 'ephemeral-storage': '8Gi'}
    digest = hashlib.sha256(payload).hexdigest()
    cm = {'apiVersion': 'v1', 'kind': 'ConfigMap', 'metadata': {'name': name, 'namespace': config['namespace'], 'labels': labels},
          'immutable': True, 'data': {'start.sh': (ROOT / 'scripts/remote-coordinator-start.sh').read_text(),
                                    'kubeconfig.json': json.dumps(login), 'nrp.toml': (ROOT / 'nrp.toml').read_text()},
          'binaryData': {'source.tgz': base64.b64encode(payload).decode()}}
    # ConfigMaps are limited to 1 MiB; leave room for metadata.
    if len(json.dumps(cm).encode()) > 950_000:
        raise ValueError('Remote coordinator source snapshot is too large')
    spec = {'automountServiceAccountToken': False, 'terminationGracePeriodSeconds': 30,
            'nodeSelector': {'kubernetes.io/arch': 'amd64'},
            'affinity': {'nodeAffinity': {'requiredDuringSchedulingIgnoredDuringExecution': {
                'nodeSelectorTerms': [{'matchExpressions': [{'key': 'feature.node.kubernetes.io/pci-10de.present',
                                                            'operator': 'NotIn', 'values': ['true']}]}]}}},
            'containers': [{'name': 'coordinator', 'image': config['task_image'],
                'command': ['/bin/sh', '/bootstrap/start.sh'],
                'resources': {'requests': resources, 'limits': resources.copy()},
                'env': [{'name': k, 'value': v} for k, v in {
                    'UV_PROJECT_ENVIRONMENT': '/opt/math-eval-venv', 'UV_CACHE_DIR': '/tmp/uv-cache',
                    'KUBECONFIG': '/root/.kube/config', 'PATH': '/opt/math-eval-venv/bin:/usr/local/bin:/usr/bin:/bin',
                    'MATH_EVAL_WORKSPACE': '/state/math-eval/remote/' + name + '/runs',
                    'OMP_NUM_THREADS': '1', 'PYTHONUNBUFFERED': '1'}.items()],
                'volumeMounts': [{'name': 'bootstrap', 'mountPath': '/bootstrap', 'readOnly': True},
                                 {'name': 'state', 'mountPath': '/state'}],
                'readinessProbe': {'exec': {'command': ['test', '-f', '/tmp/coordinator-ready']}, 'periodSeconds': 5}}],
            'volumes': [{'name': 'bootstrap', 'configMap': {'name': name}},
                        {'name': 'state', 'persistentVolumeClaim': {'claimName': config['output_pvc']}}]}
    if config.get('image_pull_secret'):
        spec['imagePullSecrets'] = [{'name': config['image_pull_secret']}]
    deployment = {'apiVersion': 'apps/v1', 'kind': 'Deployment',
        'metadata': {'name': name, 'namespace': config['namespace'], 'labels': labels,
                     'annotations': {'math-eval/source-sha256': digest}},
        'spec': {'replicas': 1, 'strategy': {'type': 'Recreate'}, 'selector': {'matchLabels': {LABEL: name}},
                 'template': {'metadata': {'labels': labels}, 'spec': spec}}}
    return cm, deployment


def snapshot():
    payload = io.BytesIO()
    paths = ['pyproject.toml', 'uv.lock', 'README.md', 'nrp.example.toml', 'src', 'tests', 'scripts', 'tasks/aime-example']
    with tarfile.open(fileobj=payload, mode='w:gz') as archive:
        for item in paths:
            path = ROOT / item
            files = sorted(p for p in path.rglob('*') if p.is_file()) if path.is_dir() else [path]
            for file in files:
                if '__pycache__' not in file.parts and file.suffix != '.pyc':
                    archive.add(file, arcname=file.relative_to(ROOT), recursive=False)
    return payload.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', default='math-eval-remote-dayallen')
    parser.add_argument('command', choices=['create', 'status', 'logs', 'login', 'check', 'shell', 'delete'])
    parser.add_argument('--dry-run', action='store_true', help='Print the CPU Deployment and public bootstrap ConfigMap')
    args = parser.parse_args()
    if not re.fullmatch(r'[a-z0-9]([-a-z0-9]{0,46}[a-z0-9])?', args.name):
        parser.error('Use a Kubernetes name with at most 48 characters')
    config = tomllib.loads((ROOT / 'nrp.toml').read_text())
    base = ['kubectl', '--context', config['context'], '-n', config['namespace']]
    if args.command == 'create':
        cm, deployment = documents(config, args.name, public_login(config), snapshot())
        if args.dry_run:
            print(json.dumps({'apiVersion': 'v1', 'kind': 'List', 'items': [cm, deployment]}, indent=2))
            return
        for kind in ('deployment', 'configmap'):
            if call([*base, 'get', kind, args.name, '--ignore-not-found', '-o', 'name']).strip():
                raise ValueError('Coordinator already exists; inspect it before deleting/replacing it')
        subprocess.run([*base, 'create', '-f', '-'], input=json.dumps(cm), text=True, check=True)
        subprocess.run([*base, 'create', '-f', '-'], input=json.dumps(deployment), text=True, check=True)
        print('Use status/logs until Ready, then login and check. No GPU has been requested.')
        return
    if args.dry_run:
        parser.error('--dry-run is only supported with create')
    deployment = json.loads(call([*base, 'get', 'deployment', args.name, '-o', 'json']))
    if deployment['metadata'].get('labels', {}).get(LABEL) != args.name:
        raise ValueError('Refusing to operate on a Deployment not marked as this coordinator')
    if args.command == 'delete':
        subprocess.run([*base, 'delete', 'deployment', args.name], check=True)
        cm = json.loads(call([*base, 'get', 'configmap', args.name, '-o', 'json']))
        if cm['metadata'].get('labels', {}).get(LABEL) != args.name:
            raise ValueError('Refusing to delete unrelated ConfigMap')
        subprocess.run([*base, 'delete', 'configmap', args.name], check=True)
        print('Coordinator removed. PVCs, models and evaluation results retained.')
        return
    if args.command == 'status':
        subprocess.run([*base, 'get', 'deployment', args.name], check=True)
        subprocess.run([*base, 'get', 'pods', '-l', LABEL + '=' + args.name, '-o', 'wide'], check=True)
        return
    pods = json.loads(call([*base, 'get', 'pods', '-l', LABEL + '=' + args.name, '-o', 'json']))['items']
    pods = [p for p in pods if not p['metadata'].get('deletionTimestamp')]
    if len(pods) != 1:
        raise RuntimeError('Expected one coordinator pod; inspect status')
    pod = pods[0]['metadata']['name']
    if args.command == 'logs':
        subprocess.run([*base, 'logs', pod, '--tail=50'], check=True)
    elif args.command == 'login':
        # kubectl consumes the plugin token; only the browser instructions are printed.
        subprocess.run([*base, 'exec', '-it', pod, '--', 'kubectl', '--context', config['context'],
                        '-n', config['namespace'], 'get', 'pvc', config['output_pvc']], check=True)
    elif args.command == 'check':
        checks = [('create', 'jobs.batch'), ('get', 'jobs.batch'), ('delete', 'jobs.batch'),
                  ('create', 'pods'), ('get', 'pods'), ('delete', 'pods'), ('create', 'pods/exec'), ('get', 'pods/log')]
        for verb, resource in checks:
            subprocess.run([*base, 'exec', pod, '--', 'kubectl', '--context', config['context'],
                            '-n', config['namespace'], 'auth', 'can-i', verb, resource], check=True)
        subprocess.run([*base, 'exec', pod, '--', 'evalctl', 'run', '--model',
                        'math-eval-models:Qwen3-8B-SFT-no-labels', '--pilot', '--dry-run'], stdout=subprocess.DEVNULL, check=True)
        print('User permissions and pilot configuration verified. No GPU evaluation submitted.')
    elif args.command == 'shell':
        subprocess.run([*base, 'exec', '-it', pod, '--', 'tmux', 'attach-session', '-t', 'eval'], check=True)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        sys.exit(str(error))
