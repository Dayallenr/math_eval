"""Exercise the real orchestration and graders with fixture generation and a fake API."""
from copy import deepcopy
import contextlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from math_eval.nrp import DEFAULTS, job_for, preflight, rbac_for, request_for, request_hash, relative_path, validate_config
from math_eval.nrp_workflow import (KubernetesError, atomic_json, coordinate, ensure_job,
                                    run_path, staged_suite, worker_generate, worker_grade)
from math_eval.suite import SOURCES, digest, load_suite, prepare
from test_checkpoint_pipeline import FakeGenerator, fixtures


class FakeJobs:
    def __init__(self, request, root, suite_volume, model_volume, fail=None):
        self.request, self.root = request, root
        self.suite_volume, self.model_volume = suite_volume, model_volume
        self.fail = fail
        self.created = []
        self.jobs = {request['evaluation_id']: {'metadata': {'uid': 'controller-uid'}}}

    def get(self, name):
        if name not in self.jobs:
            raise KubernetesError(404, 'Not found')
        return self.jobs[name]

    def create(self, document):
        name = document['metadata']['name']
        role = document['metadata']['labels']['math-eval/role']
        self.created.append(role)
        if role == self.fail:
            document['status'] = {'conditions': [{'type': 'Failed', 'status': 'True', 'reason': 'FixtureFailure'}]}
        else:
            directory = self.root / self.request['config']['output_relative_path'] / self.request['evaluation_id']
            if role == 'generate':
                worker_generate(directory, self.suite_volume, self.model_volume, generator_factory=FakeGenerator)
            else:
                from math_eval.results import grade_run as real_grade
                def fixture_harbor(suite, run, **kwargs):
                    assert kwargs['backend'] == 'harbor'
                    assert kwargs['environment_import'] == 'math_eval.harbor_nrp:NRPEnvironment'
                    summary = real_grade(suite, run, backend='local')
                    summary['grading_backend'] = 'harbor'
                    summary['harbor_environment'] = kwargs['environment_import']
                    return summary
                with patch('math_eval.nrp_workflow.configure_incluster_kubectl'), patch.dict('os.environ', {'MATH_EVAL_PARENT_UID': 'grade-uid'}), patch('math_eval.nrp_workflow.grade_run', side_effect=fixture_harbor):
                    worker_grade(directory, self.suite_volume)
            document['status'] = {'conditions': [{'type': 'Complete', 'status': 'True'}]}
        self.jobs[name] = document
        return document


class NRPWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.suites = self.root / 'suite-volume'
        self.suites.mkdir()
        suite = self.root / 'math-five'
        prepare(suite, list(SOURCES), local_sources=fixtures(self.root / 'source'))
        archive = self.suites / 'suite.tgz'
        with tarfile.open(archive, 'w:gz') as packed:
            packed.add(suite, arcname='math-five')
        self.models = self.root / 'model-volume'
        model = self.models / 'exports/model-v1'
        model.mkdir(parents=True)
        (model / 'config.json').write_text('{}')
        (model / 'model.safetensors').write_bytes(b'test fixture only')
        self.config = deepcopy(DEFAULTS)
        self.config.update(image='registry.example/lab/math-eval:test', task_image='registry.example/lab/math-task:test', suite_archive='suite.tgz',
                           archive_sha256=digest(archive.read_bytes()), full_sha256=load_suite(suite)[2],
                           pilot_suite='math-five', pilot_sha256=load_suite(suite)[2])
        self.config['settings'].update(device='cuda', max_new_tokens=16)
        self.request = request_for(self.config, 'model-pvc:exports/model-v1', name='math-eval-fixture')
        self.output = self.root / 'output-volume'

    def test_full_controller_generation_grading_report_without_client_or_docker(self):
        api = FakeJobs(self.request, self.output, self.suites, self.models)
        with contextlib.redirect_stdout(io.StringIO()):
            report = coordinate(self.request, self.output, api=api, sleep=lambda _: None)
        self.assertEqual(api.created, ['generate', 'grade'])
        self.assertEqual(report['status'], 'complete')
        self.assertEqual(report['grading_backend'], 'harbor')
        self.assertEqual(len(report['benchmarks']), 5)
        self.assertTrue(all(s['accuracy'] == 1.0 for s in report['benchmarks'].values()))
        directory = self.output / self.config['output_relative_path'] / self.request['evaluation_id']
        self.assertEqual(json.loads((directory / 'pipeline.json').read_text())['status'], 'complete')
        self.assertTrue((directory / 'benchmarks.csv').is_file())
        run = run_path(directory, self.request)
        self.assertTrue((run / 'grading-diagnostics.tgz').is_file())
        self.assertFalse((run / 'grading').exists())
        self.assertEqual(json.loads((run / 'run.json').read_text())['status'], 'graded')
        self.assertEqual(api.jobs[self.request['evaluation_id'] + '-grade']['metadata']['ownerReferences'][0]['uid'], 'controller-uid')
        with self.assertRaises(FileExistsError):
            coordinate(self.request, self.output, api=api)

    def test_failed_generation_never_submits_grading(self):
        api = FakeJobs(self.request, self.output, self.suites, self.models, fail='generate')
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, 'FixtureFailure'):
            coordinate(self.request, self.output, api=api, sleep=lambda _: None)
        self.assertEqual(api.created, ['generate'])
        directory = self.output / self.config['output_relative_path'] / self.request['evaluation_id']
        state = json.loads((directory / 'pipeline.json').read_text())
        self.assertEqual(state['status'], 'failed')
        self.assertFalse((directory / 'report.json').exists())

    def test_failed_harbor_grade_preserves_diagnostics_without_fallback(self):
        directory = self.output / self.config['output_relative_path'] / self.request['evaluation_id']
        directory.mkdir(parents=True)
        atomic_json(directory / 'request.json', self.request)
        run = worker_generate(directory, self.suites, self.models, generator_factory=FakeGenerator)
        def failed_grade(suite, scratch, **kwargs):
            self.assertEqual(kwargs['backend'], 'harbor')
            grading = scratch / 'grading'
            grading.mkdir()
            (grading / 'aime2026.log').write_text('Fixture infrastructure failure')
            atomic_json(scratch / 'summary.json', {'status': 'incomplete'})
            raise RuntimeError('Fixture grading error')
        with patch('math_eval.nrp_workflow.configure_incluster_kubectl'), patch.dict('os.environ', {'MATH_EVAL_PARENT_UID': 'grade-uid'}), patch('math_eval.nrp_workflow.grade_run', side_effect=failed_grade) as grader:
            with self.assertRaisesRegex(RuntimeError, 'Fixture grading error'):
                worker_grade(directory, self.suites)
        self.assertEqual(grader.call_count, 1)
        self.assertTrue((run / 'grading-diagnostics.tgz').is_file())
        self.assertEqual(json.loads((run / 'summary.json').read_text())['status'], 'incomplete')
        self.assertFalse((directory / 'report.json').exists())

    def test_lost_create_response_reuses_existing_matching_job(self):
        document = job_for(self.request, 'generate')
        class LostResponse:
            calls = 0
            def get(self, name):
                self.calls += 1
                if self.calls == 1:
                    raise KubernetesError(404, 'Not found')
                return document
            def create(self, document):
                raise KubernetesError(409, 'Already exists after lost create response')
        self.assertEqual(ensure_job(LostResponse(), document), document)

    def test_gpu_only_generation_and_controller_has_no_model_mount(self):
        for role in ('controller', 'generate', 'grade'):
            job = job_for(self.request, role)
            pod = job['spec']['template']['spec']
            container = pod['containers'][0]
            self.assertEqual(self.config['gpu_resource'] in container['resources']['requests'], role == 'generate')
            self.assertEqual('model' in [m['name'] for m in container['volumeMounts']], role == 'generate')
            self.assertEqual(pod['automountServiceAccountToken'], role in ('controller', 'grade'))
            self.assertEqual(job['spec']['backoffLimit'], 0)
            for mount in container['volumeMounts']:
                if mount['name'] in ('suite', 'model'):
                    self.assertTrue(mount['readOnly'])
        rules = rbac_for(self.config)['items'][1]['rules']
        self.assertEqual(rules, [{'apiGroups': ['batch'], 'resources': ['jobs'], 'verbs': ['create', 'get']}])

    def test_gpu_memory_constraint_and_a6000_resource_selection(self):
        request = deepcopy(self.request)
        request['config']['gpu_resource'] = 'nvidia.com/rtxa6000'
        pod = job_for(request, 'generate')['spec']['template']['spec']
        self.assertEqual(pod['containers'][0]['resources']['requests']['nvidia.com/rtxa6000'], 1)
        self.assertNotIn('nvidia.com/gpu', pod['containers'][0]['resources']['requests'])
        expression = pod['affinity']['nodeAffinity']['requiredDuringSchedulingIgnoredDuringExecution']['nodeSelectorTerms'][0]['matchExpressions'][0]
        self.assertEqual(expression, {'key': 'nvidia.com/gpu.memory', 'operator': 'Gt', 'values': ['40960']})
        self.assertEqual(pod['nodeSelector'], {'kubernetes.io/arch': 'amd64'})
        request['config']['min_gpu_memory_mib'] = 0
        with self.assertRaisesRegex(ValueError, 'min_gpu_memory_mib'):
            validate_config(request['config'])

    def test_bad_archive_and_foreign_existing_job_are_rejected(self):
        request = deepcopy(self.request)
        request['config']['archive_sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'archive fingerprint'):
            with staged_suite(request, self.suites):
                self.fail('Corrupt archive was accepted')
        api = FakeJobs(self.request, self.output, self.suites, self.models)
        document = job_for(self.request, 'generate')
        api.jobs[document['metadata']['name']] = {'metadata': {'labels': {}, 'annotations': {}}}
        with self.assertRaisesRegex(ValueError, 'does not match'):
            ensure_job(api, document)
        self.assertEqual(api.created, [])

    def test_request_paths_and_output_claim_validation(self):
        for value in ('../escape', '/absolute', '', '.'):
            with self.assertRaises(ValueError):
                relative_path(value)
        with self.assertRaises(ValueError):
            request_for(self.config, 'model-pvc:../escape')
        claims = {'items': [{'metadata': {'name': name}, 'status': {'phase': 'Bound'},
                            'spec': {'accessModes': ['ReadWriteOnce']}}
                           for name in ['model-pvc', 'shared-pvc']]}
        with patch('math_eval.nrp.kubectl', return_value=json.dumps(claims)), self.assertRaisesRegex(ValueError, 'ReadWriteMany'):
            preflight(self.request)

    def test_generation_pointer_rejects_modified_request_and_escape(self):
        directory = self.root / 'evaluation'
        directory.mkdir()
        atomic_json(directory / 'generation.json', {'request_sha256': request_hash(self.request), 'run_relative_path': 'runs/../../outside'})
        with self.assertRaisesRegex(ValueError, 'pointer'):
            run_path(directory, self.request)
        atomic_json(directory / 'generation.json', {'request_sha256': 'other', 'run_relative_path': 'runs/id'})
        with self.assertRaisesRegex(ValueError, 'another'):
            run_path(directory, self.request)


if __name__ == '__main__':
    unittest.main()
