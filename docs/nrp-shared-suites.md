# Reuse the lab's uploaded math benchmarks

The prepared full suite and pilot were uploaded and verified on NRP on
October 29, 2026. Teammates with access to `wang-research-lab` can reuse them for
later checkpoints. No dataset upload or new suite PVC is needed for each model.

| Setting | Value |
| --- | --- |
| Kubernetes context | `nautilus` |
| Namespace | `wang-research-lab` |
| Existing suite PVC | `shared-pvc` |
| Storage class / access | `rook-cephfs` / `ReadWriteMany` |
| Full suite path relative to PVC root | `math-eval/suites/math-seven` |
| Pilot path relative to PVC root | `math-eval/suites/pilot-seven` |
| Downloadable archive relative to PVC root | `math-eval/archives/math-seven-b43163a56bb3-pilot-b7a747d5908a.tgz` |
| Verification receipt relative to PVC root | `math-eval/shared-suite-seven.json` |

The full suite contains **823 questions** across the seven benchmarks requested
from the screenshot: MATH-500 500, BeyondAIME 100, OlymMATH English HARD 100,
HMMT February 2025 30, HMMT February 2026 33, AIME 2025 30, and AIME 2026 30.
The pilot has two per benchmark, fourteen total. These are the complete pinned
releases, not the screenshot's unidentified 20-question samples.

The previous five-benchmark suite and archive remain available unchanged for old
runs; [its historical receipt](nrp-shared-suites-legacy.json) records those files.
Do not use that old suite for new seven-benchmark comparisons.
[The current receipt](nrp-shared-suites.json) records the new fingerprints.

## Default: generation and Harbor grading on NRP

First [verify the published runner directly on NRP](nrp-verification.md); this
does not use GitLab CI or a GPU. Then use the [automatic NRP workflow](../README.md). Copy `nrp.example.toml` to
`nrp.toml` and configure the published runner/task images and pull Secret. Keep
these already-uploaded settings:

```toml
namespace = "wang-research-lab"
suite_pvc = "shared-pvc"
output_pvc = "shared-pvc"
suite_archive = "math-eval/archives/math-seven-b43163a56bb3-pilot-b7a747d5908a.tgz"
```

Keep the archive/full/pilot SHA-256 values from the example unchanged. Install the
controller and Harbor namespace roles once, then supply a completed model export:

```bash
uv run --no-sync math-eval nrp run --model YOUR-MODEL-PVC:exports/model-v1 --pilot
# After inspecting a successful pilot, submit the full suite:
uv run --no-sync math-eval nrp run --model YOUR-MODEL-PVC:exports/model-v1
```

The controller starts the GPU worker, waits for generation, then starts Harbor
on a CPU worker. Both workers mount the suite PVC read-only, copy the one pinned
archive to temporary container disk, and validate its contents and fingerprints.
Harbor creates isolated task pods and runs each verifier there. Reports and
compressed Harbor diagnostics are saved in a unique evaluation directory under
`shared-pvc/math-eval/evaluations/`.

**No benchmark download or upload is required on a teammate's laptop.** The
laptop submits Jobs and reads status/report logs. These stored files alone do not
mean deployment is complete: published images, namespace roles, and a real model
pilot must also be validated. See the README's deployment status.

The suite is immutable. Do not edit/regenerate it in place or delete the PVC.
A new benchmark version needs a new archive, directory, and pinned fingerprints.
Keep the suite identity and decoding settings fixed when comparing checkpoints.
Claims cannot be mounted across namespaces; another namespace/cluster needs its
own prepared storage and profile. Shared storage is not a backup.

## Optional: download a local copy for manual debugging

Only follow this section for the optional [manual workflow](manual-evaluation.md).
The automated NRP workflow does not need it. Skip the download if local
`load_suite` validation already matches both hashes below. These commands use a separate temporary CPU
pod mounting only the suite PVC read-only. No model, inference image, registry
secret, or GPU is needed for this download. They follow NRP's
[small-file transfer guidance](https://nrp.ai/documentation/userdocs/storage/move-data/);
the compressed archive is about 0.75 MB.

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
  "$MATH_EVAL_SHARED_DOWNLOAD_POD:/suites/math-eval/archives/math-seven-b43163a56bb3-pilot-b7a747d5908a.tgz" \
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
assert hashlib.sha256(archive.read_bytes()).hexdigest() == '208e3344aaba2a3e1be1fb6edd50eb8a28dd09a32f9213918148b277ecfd5643', 'Archive differs; download again'
expected = {
    'math-seven': 'b43163a56bb3944876ec01c781feedb8dcd4fcea74ca1d44491f17cd8c1ad121',
    'pilot-seven': 'b7a747d5908a0c2436c06c388fde4479816cb80bded0e292a0617123bccd88ce',
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
errors, use [manual troubleshooting](manual-evaluation.md#troubleshooting). The pod's GPU
node exclusion follows [NRP's CPU workload guidance](https://nrp.ai/documentation/userdocs/running/cpu-only/).
