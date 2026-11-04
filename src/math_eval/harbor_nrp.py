"""Harbor environment provider using isolated, prebuilt Kubernetes task pods."""

import asyncio
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shlex
import time
from uuid import uuid4

from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.environments.capabilities import EnvironmentCapabilities, EnvironmentResourceCapabilities


class NRPEnvironment(BaseEnvironment):
    def __init__(self, *args, namespace, context='', image_map=None, image_pull_secret='',
                 evaluation_id='', owner_job='', owner_uid='', sandbox_timeout=1800, **kwargs):
        self.namespace = namespace
        self.context = context
        self.image_map = image_map or {}
        self.image_pull_secret = image_pull_secret
        self.evaluation_id = evaluation_id
        self.owner_job, self.owner_uid = owner_job, owner_uid
        self.sandbox_timeout = int(sandbox_timeout)
        if self.sandbox_timeout < 1:
            raise ValueError('sandbox_timeout must be positive')
        self.pod_name = 'math-eval-task-' + uuid4().hex[:20]
        super().__init__(*args, **kwargs)

    @staticmethod
    def type():
        return 'nrp-kubernetes'

    @property
    def capabilities(self):
        return EnvironmentCapabilities()  # Single Linux container; no GPU/compose/network-policy claims.

    @classmethod
    def resource_capabilities(cls):
        return EnvironmentResourceCapabilities(cpu_limit=True, cpu_request=True,
                                               memory_limit=True, memory_request=True)

    def _validate_definition(self):
        if any((self.environment_dir / name).exists() for name in ('docker-compose.yaml', 'docker-compose.yml', 'compose.yaml', 'compose.yml')):
            raise ValueError('NRP environment currently supports single-container tasks; compose requires another provider')
        if not self.task_env_config.docker_image and not (self.environment_dir / 'Dockerfile').is_file():
            raise FileNotFoundError('Task needs a prebuilt docker_image or a mapped Dockerfile')

    def task_image(self, force_build=False):
        if force_build:
            raise ValueError('Build and publish task images before submitting an NRP run')
        if self.task_env_config.docker_image:
            return self.task_env_config.docker_image
        dockerfile = self.environment_dir / 'Dockerfile'
        fingerprint = hashlib.sha256(dockerfile.read_bytes()).hexdigest()
        if fingerprint not in self.image_map:
            raise ValueError('Publish an image for this task Dockerfile and add its SHA256 to task_images: ' + fingerprint)
        return self.image_map[fingerprint]

    def pod_document(self, image):
        cpu = str(self.task_env_config.cpus or 1)
        memory = str(self.task_env_config.memory_mb or 1024) + 'Mi'
        storage = str(self.task_env_config.storage_mb or 1024) + 'Mi'
        resources = {'cpu': cpu, 'memory': memory, 'ephemeral-storage': storage}
        labels = {'app.kubernetes.io/name': 'math-eval', 'math-eval/role': 'harbor-task'}
        if self.evaluation_id:
            labels['math-eval/evaluation'] = self.evaluation_id
        metadata = {'name': self.pod_name, 'namespace': self.namespace, 'labels': labels}
        if self.owner_job and self.owner_uid:
            metadata['ownerReferences'] = [{'apiVersion': 'batch/v1', 'kind': 'Job', 'name': self.owner_job,
                                           'uid': self.owner_uid, 'controller': False}]
        container = {'name': 'task', 'image': image, 'command': ['bash', '-lc', f'exec sleep {self.sandbox_timeout}'],
                     'resources': {'requests': resources, 'limits': resources},
                     'env': [{'name': k, 'value': v} for k, v in self._startup_env().items()]}
        if self.task_env_config.workdir:
            container['workingDir'] = self.task_env_config.workdir
        pod = {'restartPolicy': 'Never', 'automountServiceAccountToken': False,
               'nodeSelector': {'kubernetes.io/arch': 'amd64'},
               'activeDeadlineSeconds': self.sandbox_timeout,
               'containers': [container],
               'affinity': {'nodeAffinity': {'requiredDuringSchedulingIgnoredDuringExecution': {
                   'nodeSelectorTerms': [{'matchExpressions': [{'key': 'feature.node.kubernetes.io/pci-10de.present',
                                                                'operator': 'NotIn', 'values': ['true']}]}]}}}}
        if self.image_pull_secret:
            pod['imagePullSecrets'] = [{'name': self.image_pull_secret}]
        # No PVCs, model weights, controller token, hostPath, or privileged mode in task pods.
        return {'apiVersion': 'v1', 'kind': 'Pod', 'metadata': metadata, 'spec': pod}

    async def _kubectl(self, arguments, *, payload=None, timeout=60, check=True):
        command = ['kubectl']
        if self.context:
            command.extend(['--context', self.context])
        command.extend(['--namespace', self.namespace, '--request-timeout=30s', *arguments])
        process = await asyncio.create_subprocess_exec(*command,
                    stdin=asyncio.subprocess.PIPE if payload is not None else asyncio.subprocess.DEVNULL,
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        try:
            out, err = await asyncio.wait_for(process.communicate(payload), timeout=timeout)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            process.terminate()
            try:
                await asyncio.wait_for(process.communicate(), timeout=5)
            except asyncio.TimeoutError:
                process.kill()
                await process.communicate()
            raise
        result = ExecResult(stdout=out.decode(errors='replace'), stderr=err.decode(errors='replace'),
                            return_code=process.returncode)
        if check and result.return_code:
            raise RuntimeError(f'kubectl {arguments[0]} failed: {result.stderr or result.stdout}')
        return result

    async def start(self, force_build=False):
        if self.task_env_config.gpus or self.task_env_config.tpu:
            raise ValueError('This NRP task provider supports CPU task pods only')
        document = self.pod_document(self.task_image(force_build))
        self._validate_definition()
        self._validate_network_policy_support()
        self._validate_windows_support()
        try:
            await self._kubectl(['create', '-f', '-'], payload=json.dumps(document).encode())
            deadline = time.monotonic() + min(self.sandbox_timeout, 600)
            while time.monotonic() < deadline:
                result = await self._kubectl(['get', 'pod', self.pod_name, '-o', 'json'])
                pod = json.loads(result.stdout)
                phase = pod.get('status', {}).get('phase')
                if phase in ('Failed', 'Succeeded'):
                    raise RuntimeError(f'Task pod ended before ready: {pod.get("status")}')
                statuses = pod.get('status', {}).get('containerStatuses', [])
                for status in statuses:
                    waiting = status.get('state', {}).get('waiting', {})
                    if waiting.get('reason') in ('ImagePullBackOff', 'ErrImagePull'):
                        raise RuntimeError('Task image pull failed: ' + waiting.get('message', waiting['reason']))
                if phase == 'Running' and statuses and all(s.get('ready') for s in statuses):
                    break
                await asyncio.sleep(2)
            else:
                raise TimeoutError('Task pod did not become ready before startup deadline')
            directories = ['/logs/agent', '/logs/verifier']
            for mount in self._mounts:
                if mount.get('target'):
                    target = mount['target']
                    directories.append(str(PurePosixPath(target).parent) if Path(mount.get('source', '')).is_file() else target)
            await self.exec('mkdir -p ' + ' '.join(shlex.quote(p) for p in directories))
            for mount in self._mounts:
                source, target = Path(mount.get('source', '')), mount.get('target')
                if target and source.is_file():
                    await self.upload_file(source, target)
                elif target and source.is_dir():
                    await self.upload_dir(source, target)
            await self._upload_environment_dir_after_start()
        except BaseException:
            await self.stop(delete=True)
            raise

    async def stop(self, delete=True):
        if delete:
            await self._kubectl(['delete', 'pod', self.pod_name, '--ignore-not-found', '--wait=false'], check=False)

    async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
        env = self._merge_env(env)
        parts = []
        for key, value in (env or {}).items():
            if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', key):
                raise ValueError('Invalid environment variable name')
            parts.append(f'export {key}={shlex.quote(value)}')
        if cwd or self.task_env_config.workdir:
            parts.append('cd ' + shlex.quote(cwd or self.task_env_config.workdir))
        parts.append(command)
        shell = ' && '.join(parts)
        user = self._resolve_user(user)
        if user is not None and str(user) not in ('0', 'root'):
            if not isinstance(user, str) or not re.fullmatch(r'[a-z_][a-z0-9_-]*', user):
                raise ValueError('NRP exec supports root or an existing named container user')
            shell = 'exec su -s /bin/bash ' + shlex.quote(user) + ' -c ' + shlex.quote(shell)
        remote = ['bash', '-lc', shell]
        if timeout_sec is not None:
            # Enforce timeout inside the pod too: closing kubectl alone may leave a process alive.
            remote = ['timeout', '--signal=TERM', '--kill-after=5s', str(max(1, timeout_sec)) + 's', *remote]
        return await self._kubectl(['exec', self.pod_name, '-c', 'task', '--', *remote],
                                   timeout=(timeout_sec + 20) if timeout_sec else 600, check=False)

    async def upload_file(self, source_path, target_path):
        result = await self.exec('mkdir -p ' + shlex.quote(str(PurePosixPath(target_path).parent)))
        if result.return_code:
            raise RuntimeError('Could not create task directory: ' + str(result.stderr))
        await self._kubectl(['cp', str(source_path), self.pod_name + ':' + target_path, '-c', 'task'])

    async def upload_dir(self, source_dir, target_dir):
        result = await self.exec('mkdir -p ' + shlex.quote(target_dir))
        if result.return_code:
            raise RuntimeError('Could not create task directory: ' + str(result.stderr))
        await self._kubectl(['cp', str(Path(source_dir).resolve()) + '/.', self.pod_name + ':' + target_dir, '-c', 'task'])

    async def download_file(self, source_path, target_path):
        Path(target_path).parent.mkdir(parents=True, exist_ok=True)
        await self._kubectl(['cp', self.pod_name + ':' + source_path, str(target_path), '-c', 'task'])

    async def download_dir(self, source_dir, target_dir):
        Path(target_dir).mkdir(parents=True, exist_ok=True)
        await self._kubectl(['cp', self.pod_name + ':' + source_dir.rstrip('/') + '/.', str(target_dir), '-c', 'task'])
