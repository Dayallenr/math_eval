# Reuse the lab's uploaded math benchmarks

The prepared full suite and pilot were uploaded and verified on NRP on
October 28, 2026. Teammates with access to `wang-research-lab` can reuse them for
later checkpoints. No dataset upload or new suite PVC is needed for each model.

| Setting | Value |
| --- | --- |
| Kubernetes context | `nautilus` |
| Namespace | `wang-research-lab` |
| Existing suite PVC | `shared-pvc` |
| Storage class / access | `rook-cephfs` / `ReadWriteMany` |
| Full suite path relative to PVC root | `math-eval/suites/math-five` |
| Pilot path relative to PVC root | `math-eval/suites/pilot` |
| Downloadable archive relative to PVC root | `math-eval/archives/math-five-51ba33d52138-pilot-65433b136342.tgz` |
| Verification receipt relative to PVC root | `math-eval/shared-suite.json` |

The full suite contains 1,437 questions: MATH-500 500, AIME 2026 30, HMMT February
2026 33, OlympiadBench 674, and OlymMATH 200. The pilot contains two per benchmark,
ten total. [The checked upload receipt](nrp-shared-suites.json) records the hashes.

This setup only prepares benchmark storage. The inference image, exported model,
GPU Job, result download, and grading still need to be configured/run. GPU loading
with an actual lab checkpoint has not been validated by uploading these files.

## Use the existing storage for generation

Install the local tools and authenticate as described in
[README sections 1 and 2](../README.md#1-install-local-tools). Namespace membership
is required; another namespace cannot directly mount this claim. From the repo
root, use these suite settings when following the README:

```bash
export MATH_EVAL_NAMESPACE='wang-research-lab'
export MATH_EVAL_SUITE_PVC='shared-pvc'
export MATH_EVAL_SUITE_REL='math-eval/suites/pilot'
```

Keep the model PVC/path, image, and output PVC/path appropriate to your checkpoint.
The renderer's existing `--suite-pvc "$MATH_EVAL_SUITE_PVC"` and
`--suite-relative-path "$MATH_EVAL_SUITE_REL"` arguments select the stored pilot.
For the full run, change only the suite selection and use a newly named Job:

```bash
export MATH_EVAL_SUITE_REL='math-eval/suites/math-five'
```

The GPU container sees these paths as `/suites/math-eval/suites/pilot` and
`/suites/math-eval/suites/math-five`. Its suite mount is read-only. Do not edit or
add files inside either stored suite, regenerate it in place, or delete the
shared PVC. New benchmark versions need a different directory and matching local
copy. Several people may read the same stored suite, while each evaluation writes
to its own run directory.

## Get the identical local copy for grading

You need this once per laptop/checkout; skip it if local `load_suite` validation
already matches both hashes below. These commands use a separate temporary CPU
pod mounting only the suite PVC read-only. No model, inference image, registry
secret, or GPU is needed for this download. They follow NRP's
[small-file transfer guidance](https://nrp.ai/documentation/userdocs/storage/move-data/);
the compressed archive is about 1.2 MB.

Run from the repo root after completing tool installation and NRP login. With a
separate kubeconfig, first re-export `KUBECONFIG` as described in the README.

```bash
uv sync --locked --extra grading
mkdir -p .cache/nrp
export MATH_EVAL_NAMESPACE='wang-research-lab'
export MATH_EVAL_SUITE_PVC='shared-pvc'
export MATH_EVAL_SHARED_DOWNLOAD_POD="math-eval-suite-download-$(date -u +%Y%m%d%H%M%S)"
uv run --no-sync python - <<'PY' > .cache/nrp/shared-suite-download-pod.json
import json
import os
print(json.dumps({
    'apiVersion': 'v1', 'kind': 'Pod',
    'metadata': {'name': os.environ['MATH_EVAL_SHARED_DOWNLOAD_POD'],
                 'namespace': os.environ['MATH_EVAL_NAMESPACE']},
    'spec': {
        'restartPolicy': 'Never', 'activeDeadlineSeconds': 1800,
        'automountServiceAccountToken': False,
        'affinity': {'nodeAffinity': {'requiredDuringSchedulingIgnoredDuringExecution': {
            'nodeSelectorTerms': [{'matchExpressions': [{
                'key': 'feature.node.kubernetes.io/pci-10de.present',
                'operator': 'NotIn', 'values': ['true'],
            }]}],
        }}},
        'containers': [{
            'name': 'download', 'image': 'python:3.12-slim',
            'command': ['python', '-c', 'import time; time.sleep(1800)'],
            'resources': {
                'requests': {'cpu': '1', 'memory': '512Mi', 'ephemeral-storage': '1Gi'},
                'limits': {'cpu': '1', 'memory': '512Mi', 'ephemeral-storage': '1Gi'},
            },
            'volumeMounts': [{'name': 'suite', 'mountPath': '/suites', 'readOnly': True}],
        }],
        'volumes': [{'name': 'suite', 'persistentVolumeClaim': {
            'claimName': os.environ['MATH_EVAL_SUITE_PVC'], 'readOnly': True,
        }}],
    },
}, indent=2))
PY
kubectl --context nautilus create -f .cache/nrp/shared-suite-download-pod.json
kubectl --context nautilus -n "$MATH_EVAL_NAMESPACE" wait \
  --for=condition=Ready "pod/$MATH_EVAL_SHARED_DOWNLOAD_POD" --timeout=10m
kubectl --context nautilus -n "$MATH_EVAL_NAMESPACE" cp --retries=3 \
  "$MATH_EVAL_SHARED_DOWNLOAD_POD:/suites/math-eval/archives/math-five-51ba33d52138-pilot-65433b136342.tgz" \
  .cache/nrp/shared-suites.tgz
```

Check the archive and safely install the local suites. This preserves existing
matching directories and refuses different ones. It verifies every task file,
question count, and manifest with the repo's actual suite loader:

```bash
uv run --no-sync python - <<'PY'
import hashlib
from pathlib import Path
import tarfile
import tempfile
from math_eval.suite import load_suite
archive = Path('.cache/nrp/shared-suites.tgz')
assert hashlib.sha256(archive.read_bytes()).hexdigest() == '899553e4869174b347dcd9924fef41528c3334be14c2750c524263400e1347ba', 'Archive differs; download again'
expected = {
    'math-five': '51ba33d52138a6593897ef0fe1169af4e7d5b34a90099389fe4b5efcdba7b217',
    'pilot': '65433b13634216d12811ba5776ed59d589a51e2207718822013400debf2c1207',
}
root = Path('suites')
root.mkdir(exist_ok=True)
for name, fingerprint in expected.items():
    destination = root / name
    if destination.exists():
        assert load_suite(destination)[2] == fingerprint, f'Existing {destination} differs; preserve it and use a separate checkout'
with tempfile.TemporaryDirectory(prefix='.shared-download-', dir=root) as temporary:
    staging = Path(temporary)
    with tarfile.open(archive, 'r:gz') as packed:
        packed.extractall(staging, filter='data')
    for name, fingerprint in expected.items():
        assert load_suite(staging / name)[2] == fingerprint, f'Invalid extracted suite: {name}'
    for name in expected:
        destination = root / name
        if not destination.exists():
            (staging / name).rename(destination)
        manifest, questions, fingerprint = load_suite(destination)
        print(destination, len(questions), manifest['counts'], 'SHA256:', fingerprint)
PY
```

Delete only your temporary download pod once the copy succeeds:

```bash
kubectl --context nautilus -n "$MATH_EVAL_NAMESPACE" delete pod \
  "$MATH_EVAL_SHARED_DOWNLOAD_POD" --wait=true
```

The helper has a 30-minute deadline. If it stops before download finishes, recreate
it with a new name; the benchmark data remains on the PVC. For pending/image/mount
errors, use [README troubleshooting](../README.md#troubleshooting). The pod's GPU
node exclusion follows [NRP's CPU workload guidance](https://nrp.ai/documentation/userdocs/running/cpu-only/).

## Continue the evaluation

In the README, skip the two `prepare` commands in section 3, creation of a **suite**
claim in section 5, and the **Upload the suites** subsection in section 6. You still
need valid model/output storage; do not skip those checks. Build/publish your
inference image, run the pilot, then run the full suite using the existing paths.
After generation, download the run and grade with the matching local suite:

```bash
uv run --no-sync math-eval grade \
  --suite suites/math-five \
  --run runs/YOUR-COMPLETED-RUN-ID
```

Use `--suite suites/pilot` for pilot outputs. Keep the suite identity fixed when
comparing SFT-only, SFT + GRPO, and later model checkpoints. If the shared PVC is
removed or you evaluate from another namespace/cluster, storage must be restored
or copied there; this upload does not provide cross-namespace mounts or backups.
