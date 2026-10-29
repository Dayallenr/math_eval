"""Render an NRP GPU generation Job as Kubernetes JSON; never submits it."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import PurePosixPath
import re


def relative_path(text):
    path = PurePosixPath(text)
    if path.is_absolute() or '..' in path.parts:
        raise argparse.ArgumentTypeError('Use a path relative to the PVC root, without ..')
    return str(path)


def render(args):
    volumes, mounts = [], []
    for role, mount, read_only in [('model', '/models', True), ('suite', '/suites', True), ('output', '/results', False)]:
        volumes.append({'name': role, 'persistentVolumeClaim': {'claimName': getattr(args, role + '_pvc')}})
        mounts.append({'name': role, 'mountPath': mount, 'readOnly': read_only})
    arguments = ['generate', '--suite', '/suites/' + args.suite_relative_path,
                 '--model', '/models/' + args.model_relative_path,
                 '--output-root', '/results/' + args.output_relative_path,
                 '--batch-size', str(args.batch_size), '--max-new-tokens', str(args.max_new_tokens),
                 '--device', 'cuda']
    resources = {args.gpu_resource: 1, 'cpu': '4', 'memory': '48Gi', 'ephemeral-storage': '20Gi'}
    container = {'name': 'generate', 'image': args.image, 'args': arguments,
                 'resources': {'requests': resources, 'limits': resources}, 'volumeMounts': mounts}
    pod = {'restartPolicy': 'Never', 'containers': [container], 'volumes': volumes}
    if args.image_pull_secret:
        pod['imagePullSecrets'] = [{'name': args.image_pull_secret}]
    return {'apiVersion': 'batch/v1', 'kind': 'Job',
            'metadata': {'name': args.name, 'namespace': args.namespace},
            'spec': {'backoffLimit': 0, 'ttlSecondsAfterFinished': 604800, 'template': {'spec': pod}}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('image', 'namespace', 'model-pvc', 'suite-pvc', 'output-pvc'):
        p.add_argument('--' + key, required=True)
    p.add_argument('--model-relative-path', type=relative_path, required=True)
    p.add_argument('--suite-relative-path', type=relative_path, required=True)
    p.add_argument('--output-relative-path', type=relative_path, default='math-eval/runs')
    p.add_argument('--name', default='math-eval-' + datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S'))
    p.add_argument('--gpu-resource', default='nvidia.com/a100')
    p.add_argument('--image-pull-secret')
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--max-new-tokens', type=int, default=8192)
    args = p.parse_args()
    if re.fullmatch(r'[a-z0-9]([-a-z0-9]*[a-z0-9])?', args.name) is None or len(args.name) > 63:
        p.error('Job name must be a Kubernetes DNS label, at most 63 characters')
    if args.batch_size < 1 or args.max_new_tokens < 1:
        p.error('Batch size and token budget must be positive')
    print(json.dumps(render(args), indent=2))


if __name__ == '__main__':
    main()
