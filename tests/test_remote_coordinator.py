"""Protect the remote host's credential boundary and persistent-state layout."""
import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('remote_coordinator', ROOT / 'scripts/remote_coordinator.py')
remote = importlib.util.module_from_spec(spec)
spec.loader.exec_module(remote)


class RemoteCoordinatorTests(unittest.TestCase):
    def test_only_public_oidc_configuration_crosses_to_host(self):
        source = {'users': [{'user': {'token': 'SECRET', 'exec': {
            'command': 'kubectl', 'args': ['oidc-login', 'get-token', '--oidc-issuer-url=https://login.example',
                '--oidc-client-id=public', '--oidc-extra-scope=email,offline_access',
                '--oidc-client-secret=SECRET', '--listen-address=127.0.0.1:8000'],
            'env': [{'name': 'SECRET', 'value': 'SECRET'}]}}}],
            'clusters': [{'cluster': {'server': 'https://cluster.example', 'certificate-authority-data': 'CA',
                                      'proxy-url': 'http://SECRET@proxy'}}]}
        with patch.object(remote, 'call', return_value=json.dumps(source)):
            login = remote.public_login({'context': 'nautilus', 'namespace': 'lab'})
        self.assertNotIn('SECRET', json.dumps(login))
        args = login['users'][0]['user']['exec']['args']
        self.assertIn('--grant-type=device-code', args)
        self.assertIn('--skip-open-browser', args)
        self.assertFalse(any(a.startswith('--listen-address') for a in args))
        self.assertEqual(login['contexts'][0]['context']['namespace'], 'lab')

    def test_host_has_no_gpu_or_automatic_service_account_token(self):
        config = {'namespace': 'lab', 'task_image': 'registry/cpu@sha256:test',
                  'output_pvc': 'results', 'image_pull_secret': 'pull'}
        with patch.object(remote.Path, 'read_text', return_value='fixture'):
            cm, deployment = remote.documents(config, 'host', {'public': True}, b'archive')
        pod = deployment['spec']['template']['spec']
        self.assertFalse(pod['automountServiceAccountToken'])
        self.assertNotIn('nvidia.com/gpu', json.dumps(pod))
        self.assertEqual([v.get('persistentVolumeClaim', {}).get('claimName') for v in pod['volumes']], [None, 'results'])
        self.assertNotIn('Secret', cm['kind'])
        env = {v['name']: v['value'] for v in pod['containers'][0]['env']}
        self.assertTrue(env['MATH_EVAL_WORKSPACE'].startswith('/state/'))
        self.assertFalse(env['KUBECONFIG'].startswith('/state/'))
        self.assertFalse(env['UV_PROJECT_ENVIRONMENT'].startswith('/state/'))
        self.assertEqual(deployment['spec']['strategy']['type'], 'Recreate')

    def test_snapshot_excludes_credentials_and_generated_runs(self):
        import io
        import tarfile
        with tarfile.open(fileobj=io.BytesIO(remote.snapshot())) as archive:
            names = archive.getnames()
        self.assertIn('uv.lock', names)
        self.assertIn('src/math_eval/laptop.py', names)
        self.assertNotIn('nrp.toml', names)
        self.assertFalse(any(n.startswith(('runs/', '.cache/', '.venv/', '.git/')) for n in names))
        self.assertFalse(any('__pycache__' in n for n in names))
