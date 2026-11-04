"""Exercise transfers, saved-state recovery and GPU/Harbor ordering without a cluster."""

from contextlib import contextmanager
from copy import deepcopy
import io
import json
from pathlib import Path
import shutil
import tarfile
import tempfile
import unittest
from unittest.mock import Mock, patch

from math_eval import _transfer, laptop, nrp
from math_eval.nrp_workflow import atomic_json, worker_generate
from math_eval.results import grade_run
from math_eval.suite import SOURCES, digest, load_suite, prepare
from test_checkpoint_pipeline import FakeGenerator, fixtures


class SimulatedUserCluster:
    def __init__(self, request, output, suites, models):
        self.request, self.output, self.suites, self.models = request, output, suites, models
        self.generation_calls = 0
        self.completed_observed = False
        self.fail_generation = False
        self.lose_upload_ack = False
        self.calls = []
        self.scratch = output.parent / 'transfer'

    @contextmanager
    def transfer(self, cluster, request, phase, **kwargs):
        self.scratch = self.output.parent / ('transfer-' + phase)
        self.scratch.mkdir(exist_ok=True)
        _transfer.prepare(request, phase, self.output, self.suites, self.scratch)
        yield 'fixture-pod'

    def generation_job(self, request):
        self.generation_calls += 1
        if not self.fail_generation:
            directory = self.output / request['config']['output_relative_path'] / request['evaluation_id']
            worker_generate(directory, self.suites, self.models, generator_factory=FakeGenerator)

    def call(self, args, *unused, **kwargs):
        self.calls.append(args)
        if args[:2] == ['get', 'job']:
            self.completed_observed = not self.fail_generation
            return json.dumps({'metadata': {'annotations': {'math-eval/request-sha256': nrp.request_hash(self.request)}},
                               'status': {'conditions': [{'type': 'Failed' if self.fail_generation else 'Complete', 'status': 'True'}]}})
        if args[0] == 'cp':
            source, destination = args[1:3]
            if source.startswith('fixture-pod:'):
                source = self.scratch / Path(source.split(':', 1)[1]).name
            if destination.startswith('fixture-pod:'):
                destination = self.scratch / Path(destination.split(':', 1)[1]).name
            shutil.copyfile(source, destination)
            return ''
        if args[0] == 'exec':
            _transfer.apply_upload(self.request, self.output, self.scratch)
            if self.lose_upload_ack:
                self.lose_upload_ack = False
                raise RuntimeError('Lost connection after the upload was applied')
            return 'MATH_EVAL_UPLOAD_COMPLETE'
        if args[:2] == ['delete', 'pods']:
            return ''
        raise AssertionError('Unexpected cluster command: ' + str(args))


class LaptopWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.suites, self.output, self.models = [self.root / name for name in ('suites', 'output', 'models')]
        self.suites.mkdir()
        self.output.mkdir()
        suite = self.root / 'fixture-suite'
        prepare(suite, list(SOURCES), local_sources=fixtures(self.root / 'sources'))
        archive = self.suites / 'suite.tgz'
        with tarfile.open(archive, 'w:gz') as packed:
            packed.add(suite, arcname='fixture-suite')
        model = self.models / 'exports/model'
        model.mkdir(parents=True)
        (model / 'config.json').write_text('{}')
        (model / 'model.safetensors').write_bytes(b'fixture only; never loaded')
        self.config = deepcopy(nrp.DEFAULTS)
        self.config.update(image='registry.example/math-eval:test', task_image='registry.example/math-task:test',
            suite_archive='suite.tgz', archive_sha256=digest(archive.read_bytes()),
            full_suite='fixture-suite', pilot_suite='fixture-suite',
            full_sha256=load_suite(suite)[2], pilot_sha256=load_suite(suite)[2])
        self.request = nrp.request_for(self.config, 'model-pvc:exports/model', name='math-eval-laptop-fixture')
        self.local = self.root / 'local'
        self.local.mkdir()
        atomic_json(self.local / 'request.json', self.request)
        self.cluster = SimulatedUserCluster(self.request, self.output, self.suites, self.models)
        self.grade_count = 0

    def grader(self, suite, run, **kwargs):
        self.grade_count += 1
        self.assertTrue(self.cluster.completed_observed, 'Do not start grading before the GPU job completes')
        self.assertEqual(kwargs['backend'], 'harbor')
        self.assertEqual(kwargs['environment_kwargs']['context'], self.config['context'])
        self.assertNotIn('owner_uid', kwargs['environment_kwargs'])
        # This unit test substitutes only the Harbor execution; transfers/report writes are real.
        summary = grade_run(suite, run, backend='local')
        summary.update(grading_backend='harbor', harbor_environment=kwargs['environment_import'])
        atomic_json(run / 'summary.json', summary)
        return summary

    def coordinate(self, **kwargs):
        with patch('math_eval.laptop.transfer', side_effect=self.cluster.transfer):
            return laptop.coordinate(self.request, self.local, self.cluster, grader=self.grader, sleep=lambda _: None, **kwargs)

    def test_complete_cycle_uploads_separate_scores_without_roles(self):
        report = self.coordinate()
        self.assertEqual(report['status'], 'complete')
        self.assertEqual(len(report['benchmarks']), 7)
        self.assertEqual(report['coordinator'], 'laptop')
        self.assertEqual(self.cluster.generation_calls, 1)
        self.assertEqual(self.grade_count, 1)
        remote = self.output / self.config['output_relative_path'] / self.request['evaluation_id']
        self.assertEqual(json.loads((remote / 'report.json').read_text()), report)
        self.assertTrue(json.loads((self.local / 'pipeline.json').read_text())['uploaded'])
        self.assertTrue((remote / 'runs').is_dir())
        self.assertFalse(any('role' in argument or 'serviceaccount' in argument for args in self.cluster.calls for argument in args))

    def test_lost_upload_ack_resumes_without_generation_or_regrading(self):
        self.cluster.lose_upload_ack = True
        with self.assertRaisesRegex(RuntimeError, 'Lost connection'):
            self.coordinate()
        self.assertFalse(json.loads((self.local / 'pipeline.json').read_text())['uploaded'])
        self.coordinate()
        self.assertEqual(self.cluster.generation_calls, 1)
        self.assertEqual(self.grade_count, 1)
        self.assertTrue(json.loads((self.local / 'pipeline.json').read_text())['uploaded'])

    def test_generation_failure_never_calls_harbor(self):
        self.cluster.fail_generation = True
        with self.assertRaisesRegex(RuntimeError, 'GPU generation failed'):
            self.coordinate()
        self.assertEqual(self.grade_count, 0)
        self.assertFalse((self.local / 'report.json').exists())

    def test_interrupted_grading_requires_explicit_retry_and_preserves_attempt(self):
        def interrupted(suite, run, **kwargs):
            (run / 'grading').mkdir()
            (run / 'grading/partial.log').write_text('preserve me')
            raise KeyboardInterrupt()
        with patch('math_eval.laptop.transfer', side_effect=self.cluster.transfer):
            with self.assertRaises(KeyboardInterrupt):
                laptop.coordinate(self.request, self.local, self.cluster, grader=interrupted, sleep=lambda _: None)
        with self.assertRaisesRegex(RuntimeError, '--retry-grading'):
            self.coordinate()
        self.coordinate(retry_grading=True)
        self.assertEqual(self.cluster.generation_calls, 1)
        self.assertEqual(len(list((self.local / 'run').glob('grading-attempt-*/grading/partial.log'))), 1)

    def test_foreign_stored_request_cannot_be_overwritten(self):
        remote = _transfer.evaluation_directory(self.request, self.output)
        remote.mkdir(parents=True)
        atomic_json(remote / 'request.json', {'foreign': True})
        with self.assertRaisesRegex(ValueError, 'another request'):
            _transfer.prepare(self.request, 'stage', self.output, self.suites, self.root / 'scratch')
        self.assertEqual(json.loads((remote / 'request.json').read_text()), {'foreign': True})

    def test_tampered_archive_and_generation_paths_are_rejected(self):
        (self.suites / 'suite.tgz').write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError, 'fingerprint'):
            _transfer.prepare(self.request, 'stage', self.output, self.suites, self.root / 'scratch')
        remote = _transfer.evaluation_directory(self.request, self.output)
        atomic_json(remote / 'generation.json', {'request_sha256': nrp.request_hash(self.request), 'run_relative_path': '../outside'})
        with self.assertRaisesRegex(ValueError, 'Unsafe'):
            _transfer.generation_path(self.request, remote)

    def test_transfer_and_generation_manifests_have_no_api_credentials(self):
        for phase in ('stage', 'download', 'upload'):
            pod = laptop.transfer_pod(self.request, phase)['spec']
            self.assertFalse(pod['automountServiceAccountToken'])
            self.assertNotIn('serviceAccountName', pod)
            self.assertNotIn('nvidia.com/gpu', pod['containers'][0]['resources']['requests'])
            claims = [v['persistentVolumeClaim']['claimName'] for v in pod['volumes']]
            self.assertEqual(len(claims), len(set(claims)))
            if phase == 'stage':
                mounts = {m['mountPath']: m for m in pod['containers'][0]['volumeMounts']}
                self.assertTrue(mounts['/suites']['readOnly'])
                self.assertFalse(mounts['/results']['readOnly'])
        shared = deepcopy(self.request)
        shared['model_pvc'] = self.config['output_pvc']
        generation = laptop.generation_document(shared)['spec']['template']['spec']
        claims = [v['persistentVolumeClaim']['claimName'] for v in generation['volumes']]
        self.assertEqual(len(claims), len(set(claims)))
        mounts = {m['mountPath']: m for m in generation['containers'][0]['volumeMounts']}
        self.assertTrue(mounts['/suites']['readOnly'])
        self.assertTrue(mounts['/models']['readOnly'])
        self.assertFalse(mounts['/results'].get('readOnly', False))
        self.assertFalse(generation['automountServiceAccountToken'])
        self.assertNotIn('serviceAccountName', generation)
        self.assertEqual(generation['containers'][0]['resources']['requests'][self.config['gpu_resource']], 1)

    def test_laptop_preflight_never_requests_service_accounts(self):
        pvcs = {'items': [{'metadata': {'name': name}, 'status': {'phase': 'Bound'},
                          'spec': {'accessModes': ['ReadWriteMany']}}
                         for name in ('model-pvc', self.config['suite_pvc'])]}
        with patch('math_eval.nrp.kubectl', return_value=json.dumps(pvcs)) as api:
            nrp.preflight(self.request, require_accounts=False)
            self.assertEqual(api.call_count, 1)
            self.assertEqual(api.call_args.args[1][:2], ['get', 'pvc'])

    def test_cancelled_state_never_restarts_generation(self):
        atomic_json(self.local / 'pipeline.json', {'request_sha256': nrp.request_hash(self.request), 'status': 'cancelled'})
        with self.assertRaisesRegex(RuntimeError, 'cancelled'):
            self.coordinate()
        self.assertEqual(self.cluster.generation_calls, 0)

    def test_lost_job_create_response_reuses_matching_gpu_job(self):
        cluster = laptop.UserCluster(self.config)
        document = nrp.job_for(self.request, 'generate')
        cluster.call = Mock(side_effect=['', RuntimeError('Lost create response'), json.dumps(document)])
        self.assertEqual(cluster.generation_job(self.request), document)
        self.assertEqual(sum(args.args[0][0] == 'create' for args in cluster.call.call_args_list), 1)
        document['metadata']['annotations']['math-eval/request-sha256'] = 'different'
        cluster.call = Mock(return_value=json.dumps(document))
        with self.assertRaisesRegex(ValueError, 'different request'):
            cluster.generation_job(self.request)
        self.assertEqual(cluster.call.call_count, 1)

    def test_upload_rejects_modified_model_identity_and_archive_paths(self):
        self.coordinate()
        remote = _transfer.evaluation_directory(self.request, self.output)
        report_before = (remote / 'report.json').read_bytes()
        packed = self.cluster.scratch / 'upload.tgz'
        with tarfile.open(packed, 'r:gz') as archive:
            contents = {m.name: archive.extractfile(m).read() for m in archive.getmembers()}
        changed = json.loads(contents['run.json'])
        changed['model']['label'] = 'another checkpoint'
        contents['run.json'] = json.dumps(changed).encode()
        for traversal in (False, True):
            with tarfile.open(packed, 'w:gz') as archive:
                for name, value in contents.items():
                    member = tarfile.TarInfo(name)
                    member.size = len(value)
                    archive.addfile(member, io.BytesIO(value))
                if traversal:
                    member = tarfile.TarInfo('../escape')
                    member.size = 1
                    archive.addfile(member, io.BytesIO(b'x'))
            with self.assertRaisesRegex(ValueError, 'generation identity|archive member'):
                _transfer.apply_upload(self.request, self.output, self.cluster.scratch)
            self.assertEqual((remote / 'report.json').read_bytes(), report_before)


if __name__ == '__main__':
    unittest.main()
