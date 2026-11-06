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
        self.assertIn('--grant-type=authcode', args)
        self.assertIn('--skip-open-browser', args)
        self.assertIn('--listen-address=127.0.0.1:18000', args)
        self.assertNotIn('--listen-address=127.0.0.1:8000', args)
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

    def test_image_verification_bundle_is_self_contained(self):
        """Image checks must include the source files required by remote-host tests."""
        import argparse
        import io
        import tarfile
        import tempfile
        verifier_spec = importlib.util.spec_from_file_location('verify_nrp', ROOT / 'scripts/verify_nrp.py')
        verifier = importlib.util.module_from_spec(verifier_spec)
        verifier_spec.loader.exec_module(verifier)
        args = argparse.Namespace(commit='HEAD', image='registry/runner@sha256:fixture',
            name='math-eval-verify-fixture', namespace='lab', pull_secret='')
        # Model Git's archive from a fixture snapshot so this test also runs
        # in published containers, which deliberately have no .git directory.
        raw = io.BytesIO()
        with tarfile.open(fileobj=io.BytesIO(remote.snapshot())) as source, \
                tarfile.open(fileobj=raw, mode='w') as packed:
            names = source.getnames()
            for member in source.getmembers():
                packed.addfile(member, source.extractfile(member))
        def git_fixture(command, **kwargs):
            if 'rev-parse' in command:
                return 'fixture-commit\n'
            if 'ls-tree' in command:
                return '\n'.join(names)
            if 'archive' in command:
                return raw.getvalue()
            raise AssertionError(command)
        with patch.object(verifier, 'command', side_effect=git_fixture):
            _, configmap = verifier.documents(args)
        import base64
        payload = base64.b64decode(configmap['binaryData']['tests.tar.gz'])
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            with tarfile.open(fileobj=io.BytesIO(payload)) as packed:
                packed.extractall(directory, filter='data')
            bundled_spec = importlib.util.spec_from_file_location('bundled_remote', directory / 'scripts/remote_coordinator.py')
            bundled = importlib.util.module_from_spec(bundled_spec)
            bundled_spec.loader.exec_module(bundled)
            with tarfile.open(fileobj=io.BytesIO(bundled.snapshot())) as packed:
                names = packed.getnames()
            self.assertIn('src/math_eval/laptop.py', names)
            self.assertIn('uv.lock', names)
            self.assertIn('tasks/aime-example/tests/grade.py', names)
            self.assertNotIn('nrp.toml', names)
