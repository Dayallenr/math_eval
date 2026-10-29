"""Check Harbor's real custom-provider interface against a recorded Kubernetes API."""
import asyncio
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest

from harbor.environments.base import ExecResult
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.paths import TrialPaths
from math_eval.harbor_nrp import NRPEnvironment
from math_eval.suite import DOCKERFILE


class RecordedEnvironment(NRPEnvironment):
    async def _kubectl(self, args, **kwargs):
        self.calls.append((args, kwargs))
        if args[:2] == ['get', 'pod']:
            status = {'phase': 'Running', 'containerStatuses': [{'ready': True}]}
            return ExecResult(stdout=json.dumps({'status': status}), stderr='', return_code=0)
        return ExecResult(stdout='', stderr='', return_code=0)


class HarborNRPTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        environment = self.root / 'environment'
        environment.mkdir()
        (environment / 'Dockerfile').write_text(DOCKERFILE)
        self.environment = RecordedEnvironment(environment_dir=environment, environment_name='math-test',
            session_id='math-test__env', trial_paths=TrialPaths(trial_dir=self.root / 'trial'),
            task_env_config=EnvironmentConfig(), namespace='wang-research-lab',
            image_map={sha256(DOCKERFILE.encode()).hexdigest(): 'registry.test/lab/math-task:v1'},
            evaluation_id='math-eval-test', owner_job='math-eval-test-grade', owner_uid='grade-uid')
        self.environment.calls = []

    def test_harbor_starts_isolated_pod_and_deletes_it(self):
        env = self.environment
        asyncio.run(env.start(False))
        pod = json.loads(next(options['payload'] for args, options in env.calls if args[0] == 'create'))
        self.assertFalse(pod['spec']['automountServiceAccountToken'])
        self.assertNotIn('volumes', pod['spec'])
        self.assertNotIn('nvidia.com/a100', pod['spec']['containers'][0]['resources']['requests'])
        self.assertEqual(pod['metadata']['ownerReferences'][0]['uid'], 'grade-uid')
        self.assertGreater(pod['spec']['activeDeadlineSeconds'], 0)
        asyncio.run(env.stop(True))
        self.assertEqual(env.calls[-1][0][:3], ['delete', 'pod', env.pod_name])

    def test_unmapped_dockerfile_is_rejected_before_pod_creation(self):
        env = self.environment
        (env.environment_dir / 'Dockerfile').write_text('FROM another-image\n')
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            asyncio.run(env.start(False))
        self.assertFalse(env.calls)
        env.task_env_config.docker_image = 'registry.test/coding-task:v1'
        self.assertEqual(env.task_image(), 'registry.test/coding-task:v1')

    def test_exec_escapes_environment_and_enforces_remote_timeout(self):
        env = self.environment
        asyncio.run(env.exec('python --version', cwd='/app/space here', env={'VALUE': 'a; echo unsafe'}, timeout_sec=7))
        args, options = env.calls[-1]
        self.assertIn('timeout', args)
        self.assertIn('7s', args)
        self.assertIn("export VALUE='a; echo unsafe'", args[-1])
        self.assertIn("cd '/app/space here'", args[-1])
        self.assertEqual(options['timeout'], 27)
        with self.assertRaises(ValueError):
            asyncio.run(env.exec('true', env={'INVALID;key': 'value'}))

    def test_directory_copy_preserves_contents_semantics(self):
        env = self.environment
        source = self.root / 'tests'
        source.mkdir()
        asyncio.run(env.upload_dir(source, '/tests'))
        args, _ = env.calls[-1]
        self.assertEqual(args[:3], ['cp', str(source.resolve()) + '/.', env.pod_name + ':/tests'])
        asyncio.run(env.download_dir('/logs/verifier', self.root / 'download'))
        self.assertIn(env.pod_name + ':/logs/verifier/.', env.calls[-1][0])


if __name__ == '__main__':
    unittest.main()
