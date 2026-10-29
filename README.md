# Math evaluation with Harbor

Run an exported SFT or GRPO model on five math benchmarks and save a separate
score for each benchmark. This guide covers the complete NRP workflow, from a
fresh laptop checkout to downloading and grading a real model's answers.

**Status:** the code, local tests, and Harbor/Docker integration have been checked.
The inference image and a real lab checkpoint still need to be tested on your NRP
namespace. Start with the pilot below before scheduling the full suite.

**Lab datasets are already on NRP:** teammates in `wang-research-lab` can reuse
`shared-pvc` at `math-eval/suites/math-five` (full) and `math-eval/suites/pilot`.
Follow [the shared-suite instructions](docs/nrp-shared-suites.md) to obtain the
matching local grading copy and skip dataset preparation/upload. Model and image
setup, generation, and grading are still needed.

| Benchmark | Selection | Questions |
| --- | --- | ---: |
| MATH-500 | Test split | 500 |
| AIME 2026 | MathArena release | 30 |
| HMMT February 2026 | MathArena release | 33 |
| OlympiadBench | English text-only open-ended math | 674 |
| OlymMATH | English EASY + HARD | 200 |
| Total | | 1,437 |

The source releases are pinned in `src/math_eval/suite.py`; each generated
`suite.json` records the selections, source commits, and file hashes. Several
upstream splits are named `train`; this workflow uses their questions for evaluation.

The suite stores questions in one file with benchmark labels. The model loads
once and answers independent prompts, in the order above. Its weights do not
change. After all responses are saved, Harbor grades each benchmark separately.
You get five scores, not one combined accuracy. No model-serving endpoint is
required.

```text
Laptop: prepare suites and build/publish the inference image
                          |
NRP storage: model export + identical prepared suites
                          |
NRP GPU Job: load model once -> generate responses -> save run to PVC
                          |
Laptop: download run -> Harbor/Docker graders -> summary.json
```

Follow sections 1–10 in order for the first run. All shell commands run **on your
laptop, from the repository root**, unless explicitly stated otherwise. Use Bash
or zsh on macOS/Linux; Windows users should use WSL2 with Docker integration.
Run blocks one at a time and resolve any error before continuing.

- [1. Install local tools](#1-install-local-tools)
- [2. Connect to NRP and choose your settings](#2-connect-to-nrp-and-choose-your-settings)
- [3. Prepare the model and benchmark suites](#3-prepare-the-model-and-benchmark-suites)
- [4. Build and publish the inference image](#4-build-and-publish-the-inference-image)
- [5. Prepare persistent storage](#5-prepare-persistent-storage)
- [6. Upload and check files using a CPU transfer pod](#6-upload-and-check-files-using-a-cpu-transfer-pod)
- [7. Submit the NRP pilot](#7-submit-the-nrp-pilot)
- [8. Download the completed run](#8-download-the-completed-run)
- [9. Grade and read the results](#9-grade-and-read-the-results)
- [10. Run the full suite and evaluate later checkpoints](#10-run-the-full-suite-and-evaluate-later-checkpoints)
- [Other execution options](#other-execution-options)
- [Troubleshooting](#troubleshooting)

## 1. Install local tools

| Tool or resource | Role |
| --- | --- |
| Git and a checkout containing this pipeline | Source code, image definition, and job renderer |
| `uv` and Python 3.12+ | Install the CLI and run preparation/grading |
| Docker Desktop on macOS/Windows, or Docker Engine on Linux | Build the inference image and run Harbor graders |
| `kubectl` and the **int128** OIDC `kubelogin` plugin | Authenticate and submit NRP workloads |
| NRP account, lab namespace, and GPU quota | Permission and resources for generation |
| Container registry with push access | Publish the image so NRP can pull it |
| NRP persistent storage (PVCs) | Model, benchmark files, and saved responses |

On macOS with [Homebrew](https://brew.sh/) already installed:

```bash
brew install git uv kubectl kubelogin
```

Install and start [Docker Desktop](https://docs.docker.com/desktop/). For other
operating systems, use the official [uv installation](https://docs.astral.sh/uv/getting-started/installation/),
[kubectl installation](https://kubernetes.io/docs/tasks/tools/), and
[int128/kubelogin installation](https://github.com/int128/kubelogin) instructions.
The OIDC plugin is different from Azure's similarly named tool.

Use your existing checkout, or clone a branch that contains the evaluation code:

```bash
git clone https://github.com/Dayallenr/math_eval.git
cd math_eval
```

A clone only contains published commits. If `src/math_eval/cli.py` or
`scripts/render_nrp_job.py` is missing, obtain the branch/checkout containing this
pipeline before continuing. Local uncommitted changes are not included in a clone.

```bash
uv python install 3.12
uv sync --python 3.12 --locked --extra grading
uv run --no-sync math-eval --help
uv run --no-sync harbor --version
docker info
kubectl version --client
kubectl oidc-login --version
mkdir -p .cache/nrp runs
```

`uv` installs Harbor for you. Your laptop does not need CUDA or model weights for
Harbor grading. Internet access is needed to fetch packages, public benchmark
files, and container images on the first run. Keep `uv.lock` with the source;
use `--locked` and `--no-sync` as shown rather than independently upgrading packages.

## 2. Connect to NRP and choose your settings

First log in at [nrp.ai](https://nrp.ai/) and ask your lab's namespace administrator
to add your account. Follow [NRP's access instructions](https://nrp.ai/documentation/userdocs/start/getting-started/)
for authentication and membership. If you already have working NRP `kubectl`
access, keep that configuration and skip the download below.

For a new connection, save the NRP config separately from other cluster configs:

```bash
curl -fSL https://nrp.ai/config -o .cache/nrp/kubeconfig
chmod 600 .cache/nrp/kubeconfig
export KUBECONFIG="$PWD/.cache/nrp/kubeconfig"
kubectl config get-contexts
kubectl config use-context nautilus
```

The first authenticated command opens a browser for institutional login. Do not
replace an unrelated `~/.kube/config`. In a new terminal, re-export `KUBECONFIG`
if using this separate file, and rerun the settings block below.

Edit every `your-...` value. These are examples, not your lab's actual paths:

```bash
export MATH_EVAL_NAMESPACE='your-lab-namespace'
export MATH_EVAL_REGISTRY='ghcr.io'
export MATH_EVAL_IMAGE='ghcr.io/your-account/math-eval:nrp-v1'
export MATH_EVAL_PULL_SECRET=''
export MATH_EVAL_MODEL_PVC='your-existing-model-pvc'
export MATH_EVAL_SUITE_PVC='math-eval-data'
export MATH_EVAL_OUTPUT_PVC='math-eval-data'
export MATH_EVAL_MODEL_REL='runs/your-completed-model-export'
export MATH_EVAL_OUTPUT_REL='math-eval/runs'
export MATH_EVAL_TRANSFER_POD='math-eval-transfer'
export MATH_EVAL_UPLOAD_MODEL='0'
```

| Setting | Meaning |
| --- | --- |
| `MATH_EVAL_NAMESPACE` | Namespace containing **all** selected PVCs and the Job |
| `MATH_EVAL_IMAGE` | Full registry/repository/tag of the image you will publish |
| `MATH_EVAL_PULL_SECRET` | Empty for a publicly pullable image; namespace Secret name for a private image |
| `MATH_EVAL_MODEL_PVC` + `MATH_EVAL_MODEL_REL` | PVC and path **relative to its root** containing the completed model |
| `MATH_EVAL_SUITE_PVC` | PVC receiving the prepared benchmark suites |
| `MATH_EVAL_OUTPUT_PVC` + `MATH_EVAL_OUTPUT_REL` | PVC and relative directory receiving unique evaluation runs |
| `MATH_EVAL_UPLOAD_MODEL` | Keep `0` to read an existing model; use `1` only when uploading a completed local export |

For the lab's existing shared datasets, override these values after editing the
settings block above:

```bash
export MATH_EVAL_NAMESPACE='wang-research-lab'
export MATH_EVAL_SUITE_PVC='shared-pvc'
```

Use the [shared-suite guide](docs/nrp-shared-suites.md) for the local grading copy.
Keep your model/output settings appropriate to your run; the suite claim is
already provisioned and populated.

The three PVC settings may name the same claim. A claim in a different namespace
cannot be mounted directly by this Job. PVC names are not laptop folders.
These exports configure the documentation commands; the CLI does not automatically
read them or a `.env` file.

```bash
kubectl -n "$MATH_EVAL_NAMESPACE" get pods
kubectl -n "$MATH_EVAL_NAMESPACE" get pvc
kubectl -n "$MATH_EVAL_NAMESPACE" get resourcequota
kubectl -n "$MATH_EVAL_NAMESPACE" auth can-i create jobs.batch
kubectl -n "$MATH_EVAL_NAMESPACE" auth can-i create pods
kubectl -n "$MATH_EVAL_NAMESPACE" auth can-i create pods/exec
```

For Qwen3-8B, start with one A100 and batch size 1; test actual memory use in the
pilot. Larger models may not fit this single-GPU runner. The renderer requests
`nvidia.com/a100`, 4 CPUs, 48 GiB host RAM, and 20 GiB temporary disk. These are
starting allocations, not measured requirements for every model.

NRP gates A100 allocation through namespace quota. Have the lab confirm access
or use its [A100 access procedure](https://nrp.ai/documentation/userdocs/running/gpu-pods/#requesting-special-gpus).
Do not assume an NRP account alone grants an A100. An `opportunistic` workload may
be preempted; this runner does not resume partial generation, so the main recipe
uses regular approved quota.

## 3. Prepare the model and benchmark suites

The model directory needs a Hugging Face export: `config.json`, full weights
(single or sharded), tokenizer files, and its chat template. Use a **completed,
unchanging** export. Do not evaluate a directory while training is saving into it.

- **SFT:** use the completed `save_model()` output with its saved tokenizer.
- **GRPO:** use the training repo's export/merge step first (the lab GRPO repo
  provides `./run.sh export`). Point evaluation at that export, not an optimizer
  checkpoint or standalone LoRA adapter. Preserve `export_manifest.json` if present;
  the runner uses it to record the training run and step automatically.

If the model is already on NRP, reuse that PVC and determine its relative path.
For example, a training pod mounting a claim at `/data` and a model saved at
`/data/runs/model-v1` means `MATH_EVAL_MODEL_REL='runs/model-v1'`.

**Reusing the lab upload:** the suites are already prepared. Follow
[the shared-suite guide](docs/nrp-shared-suites.md) to download their exact local
copies, skip the two `prepare` commands below, and then validate them here.
Also skip the upload subsection in section 6. Continue configuring your model,
inference image, and output storage normally.

For a new independent setup, prepare a ten-question pilot and the full suite
**on your laptop**:

```bash
uv run --no-sync math-eval prepare --output suites/pilot --limit 2
uv run --no-sync math-eval prepare --output suites/math-five
```

Run each preparation command only if its output directory does not already exist.
This checkout may already contain `suites/math-five`; generated suites are ignored
by Git, so fresh clones must prepare them. Existing output is never overwritten.
The pilot has two questions per benchmark and is not a full benchmark score.

Validate the exact files before transferring them:

```bash
uv run --no-sync python - <<'PY'
from pathlib import Path
from math_eval.suite import load_suite
for name in ('pilot', 'math-five'):
    manifest, questions, fingerprint = load_suite(Path('suites') / name)
    print(name, len(questions), manifest['counts'], 'subset=', manifest['subset'])
    print('suite SHA256:', fingerprint)
PY
```

Expected totals: pilot **10**, full suite **1,437**. Keep these exact directories
for local grading. Do not add notes, edit graders, or regenerate a suite between
generation and grading; the runner verifies file hashes. The model receives only
the question and math instruction, never the reference answers or oracle script.

## 4. Build and publish the inference image

Build from the repository root, with Docker running. Log in to a registry account
that can push the repository named by `MATH_EVAL_IMAGE`:

```bash
docker login "$MATH_EVAL_REGISTRY"
docker build --platform linux/amd64 \
  -f docker/inference.Dockerfile \
  -t "$MATH_EVAL_IMAGE" .
docker run --rm --platform linux/amd64 "$MATH_EVAL_IMAGE" --help
docker push "$MATH_EVAL_IMAGE"
docker buildx imagetools inspect "$MATH_EVAL_IMAGE"
```

The image packages code and inference dependencies; it does not contain your
checkpoint or benchmark suites. `linux/amd64` targets typical NRP A100 nodes even
when building on Apple Silicon. This build can be large and cross-architecture
builds can be slow. An x86 Linux builder is another option. The local `--help`
check does not require a GPU and does not prove CUDA inference works.

Use an image tag you do not overwrite when changing code, or record its registry
digest and use `registry/repository@sha256:...` for the Job. Docker push credentials
on your laptop do not automatically authorize NRP to pull a private image.

### Private image only: configure pull credentials

For a public image, leave `MATH_EVAL_PULL_SECRET=''` and skip this section. For a
private image, use an existing authorized registry-pull Secret in this namespace,
or create a dedicated one below. Enter a registry username and a token with pull
access (for GHCR, a token authorized to read the package, including any required
organization authorization). Credentials are prompted rather than placed in the
README or command history.

```bash
export MATH_EVAL_PULL_SECRET='math-eval-registry'
uv run --no-sync python - <<'PY'
import base64
import getpass
import json
import os
import subprocess
username = getpass.getpass('Registry username (hidden): ')
token = getpass.getpass('Registry token with pull access: ')
auth = base64.b64encode(f'{username}:{token}'.encode()).decode()
config = json.dumps({'auths': {os.environ['MATH_EVAL_REGISTRY']: {'auth': auth}}})
secret = {
    'apiVersion': 'v1', 'kind': 'Secret', 'type': 'kubernetes.io/dockerconfigjson',
    'metadata': {'name': os.environ['MATH_EVAL_PULL_SECRET'],
                 'namespace': os.environ['MATH_EVAL_NAMESPACE']},
    'data': {'.dockerconfigjson': base64.b64encode(config.encode()).decode()},
}
subprocess.run(['kubectl', 'apply', '-f', '-'], input=json.dumps(secret), text=True, check=True)
PY
kubectl -n "$MATH_EVAL_NAMESPACE" get secret "$MATH_EVAL_PULL_SECRET"
```

This creates or updates the named Secret; use a name dedicated to your evaluation
image. See [Kubernetes private image authentication](https://kubernetes.io/docs/tasks/configure-pod-container/pull-image-private-registry/).
Do not commit registry tokens or kubeconfigs. `.cache/` and `.env` are ignored.

## 5. Prepare persistent storage

Reuse suitable existing claims when available. Model, suite, and output claims
must all exist in your selected namespace. Use storage reachable from the GPU
nodes you can schedule on; your lab can identify its normal region/storage class.

If using the example `math-eval-data` claim for suite/output and it does not exist,
create it once. This example requests 100 GiB of shared storage; adjust its size
and storage class for your lab, especially if also uploading model weights:

```bash
export MATH_EVAL_DATA_PVC='math-eval-data'
export MATH_EVAL_STORAGE_CLASS='rook-cephfs'
uv run --no-sync python - <<'PY' > .cache/nrp/data-pvc.json
import json
import os
print(json.dumps({
    'apiVersion': 'v1', 'kind': 'PersistentVolumeClaim',
    'metadata': {'name': os.environ['MATH_EVAL_DATA_PVC'],
                 'namespace': os.environ['MATH_EVAL_NAMESPACE']},
    'spec': {'storageClassName': os.environ['MATH_EVAL_STORAGE_CLASS'],
             'accessModes': ['ReadWriteMany'],
             'resources': {'requests': {'storage': '100Gi'}}},
}, indent=2))
PY
kubectl apply -f .cache/nrp/data-pvc.json
kubectl -n "$MATH_EVAL_NAMESPACE" get pvc
```

The storage-class example `rook-cephfs` is NRP's shared US West CephFS class;
other regions use different classes. Verify current classes with
`kubectl get storageclass` if permitted, or ask your namespace administrator.
Wait until each selected claim is `Bound` before proceeding. For an existing
claim, do not apply a new spec that changes its storage class or access mode.
[NRP storage details](https://nrp.ai/documentation/userdocs/storage/ceph/)

If all files will use this new claim, set the corresponding PVC exports:

```bash
export MATH_EVAL_MODEL_PVC="$MATH_EVAL_DATA_PVC"
export MATH_EVAL_SUITE_PVC="$MATH_EVAL_DATA_PVC"
export MATH_EVAL_OUTPUT_PVC="$MATH_EVAL_DATA_PVC"
```

Run that block only if the model will also be uploaded to this claim; otherwise
keep the existing model claim selected. Uploading a local model requires a model
PVC; either choose this same claim or create another approved claim. A `ReadWriteOnce` claim can block mounting
from another node while training or another pod uses it. Stop only your own
conflicting workload, or ask its owner to arrange access/a separate copy.

Libraries are installed into the container image, not the PVC. NRP does not allow
pip/conda installations on its CephFS storage. The image places its environment
under `/opt/math-eval` and uses `/tmp` for the Hugging Face cache.

## 6. Upload and check files using a CPU transfer pod

The transfer pod mounts PVCs for upload/download and requests **no GPU**. It runs
for at most two hours and should be deleted after each transfer. Use a unique
`MATH_EVAL_TRANSFER_POD` name if that name is already in use by another person.

If uploading a model from your laptop, first run
`export MATH_EVAL_UPLOAD_MODEL='1'`; keep `0` when reading an existing NRP model.
Render the transfer-pod definition from your current settings:

```bash
uv run --no-sync python - <<'PY' > .cache/nrp/transfer-pod.json
import json
import os
volumes, mounts = [], []
for role, mount in [('model', '/models'), ('suite', '/suites'), ('output', '/results')]:
    volumes.append({'name': role, 'persistentVolumeClaim': {
        'claimName': os.environ[f'MATH_EVAL_{role.upper()}_PVC']}})
    mounts.append({'name': role, 'mountPath': mount,
                   'readOnly': role == 'model' and os.environ['MATH_EVAL_UPLOAD_MODEL'] != '1'})
pod = {
    'apiVersion': 'v1', 'kind': 'Pod',
    'metadata': {'name': os.environ['MATH_EVAL_TRANSFER_POD'],
                 'namespace': os.environ['MATH_EVAL_NAMESPACE']},
    'spec': {
        'restartPolicy': 'Never', 'activeDeadlineSeconds': 7200,
        'containers': [{
            'name': 'transfer', 'image': os.environ['MATH_EVAL_IMAGE'],
            'command': ['python', '-c', 'import time; time.sleep(7200)'],
            'resources': {'requests': {'cpu': '1', 'memory': '1Gi'},
                          'limits': {'cpu': '1', 'memory': '1Gi'}},
            'volumeMounts': mounts,
        }],
        'volumes': volumes,
    },
}
if os.environ['MATH_EVAL_PULL_SECRET']:
    pod['spec']['imagePullSecrets'] = [{'name': os.environ['MATH_EVAL_PULL_SECRET']}]
print(json.dumps(pod, indent=2))
PY
kubectl apply --dry-run=server -f .cache/nrp/transfer-pod.json
kubectl apply -f .cache/nrp/transfer-pod.json
kubectl -n "$MATH_EVAL_NAMESPACE" wait \
  --for=condition=Ready "pod/$MATH_EVAL_TRANSFER_POD" --timeout=10m
kubectl -n "$MATH_EVAL_NAMESPACE" exec "$MATH_EVAL_TRANSFER_POD" -- tar --version
```

`kubectl cp` requires `tar` in the container. If the pod does not become ready,
use the troubleshooting section before continuing. The server dry run validates
API fields and admission rules; it does not prove the image or storage will work.

### Upload the suites

**Skip this subsection when reusing the lab's verified upload on `shared-pvc`.**
Use the existing `math-eval/suites/pilot` and `math-eval/suites/math-five` paths.
The model/output checks later in this section still apply.

Use one archive to transfer the many task files. `COPYFILE_DISABLE=1` avoids
adding macOS metadata files that would change the suite's file inventory.
The destination suite directories must not already exist; reuse an identical
uploaded suite or choose a new destination rather than overwriting an older one.

```bash
COPYFILE_DISABLE=1 tar -czf .cache/nrp/suites.tgz -C suites pilot math-five
kubectl -n "$MATH_EVAL_NAMESPACE" cp .cache/nrp/suites.tgz \
  "$MATH_EVAL_TRANSFER_POD:/tmp/math-eval-suites.tgz"
kubectl -n "$MATH_EVAL_NAMESPACE" exec -i "$MATH_EVAL_TRANSFER_POD" -- python - <<'PY'
from pathlib import Path
import subprocess
root = Path('/suites/math-eval/suites')
assert not any((root / name).exists() for name in ('pilot', 'math-five')), 'Suite destination exists; verify/reuse it or choose a fresh path'
root.mkdir(parents=True, exist_ok=True)
subprocess.run(['tar', '-xzf', '/tmp/math-eval-suites.tgz', '-C', str(root)], check=True)
PY
kubectl -n "$MATH_EVAL_NAMESPACE" exec -i "$MATH_EVAL_TRANSFER_POD" -- python - <<'PY'
from pathlib import Path
from math_eval.suite import load_suite
for name in ('pilot', 'math-five'):
    manifest, questions, fingerprint = load_suite(Path('/suites/math-eval/suites') / name)
    print(name, len(questions), 'suite SHA256:', fingerprint)
PY
```

The two remote SHA-256 values must match those printed locally in section 3.
If either destination already exists, skip the upload/extraction only after the
existing remote suite passes validation and its fingerprint matches your local
copy. If it differs, preserve the old suite, upload under a new directory, and
use that new path in the Job and local grading command.

### Only if the model is currently on your laptop

Skip this subsection when reusing a completed model already on NRP. To upload a
local model, set `MATH_EVAL_UPLOAD_MODEL='1'` **before rendering the transfer pod**
so its model mount is writable. If the pod already exists, delete your transfer
pod, rerender its JSON, and recreate it. Choose a new model-export directory;
do not overwrite an existing training run or previous export.

```bash
export MATH_EVAL_LOCAL_MODEL='/absolute/path/to/completed/model-export'
kubectl -n "$MATH_EVAL_NAMESPACE" exec "$MATH_EVAL_TRANSFER_POD" -- \
  python -c 'from pathlib import Path; import sys; p=Path(sys.argv[1]); assert not p.exists(), "Model destination exists"; p.parent.mkdir(parents=True, exist_ok=True)' \
  "/models/$MATH_EVAL_MODEL_REL" &&
kubectl -n "$MATH_EVAL_NAMESPACE" cp "$MATH_EVAL_LOCAL_MODEL" \
  "$MATH_EVAL_TRANSFER_POD:/models/$MATH_EVAL_MODEL_REL"
```

Copy the entire export, including tokenizer files and every shard. For large
models, use your lab's reliable transfer method or the alternatives in
[NRP's data-transfer guide](https://nrp.ai/documentation/userdocs/storage/move-data/).
The CPU transfer pod's two-hour deadline may need extending for a long upload.

### Check the model and output paths

This checks files and shard presence; actual model loading is checked by the GPU
pilot. Paths shown are the renderer's mount paths, not the training pod's paths:

```bash
kubectl -n "$MATH_EVAL_NAMESPACE" exec -i "$MATH_EVAL_TRANSFER_POD" -- \
  python - "/models/$MATH_EVAL_MODEL_REL" <<'PY'
import json
from pathlib import Path
import sys
from math_eval.inference import identify_model
root = Path(sys.argv[1])
identity = identify_model(str(root))
assert (root / 'tokenizer_config.json').is_file(), 'Missing tokenizer_config.json'
assert (root / 'tokenizer.json').is_file() or (root / 'tokenizer.model').is_file() or (root / 'vocab.json').is_file(), 'Missing tokenizer vocabulary files'
for index in root.glob('*.index.json'):
    shards = set(json.loads(index.read_text()).get('weight_map', {}).values())
    assert all((root / name).is_file() for name in shards), f'Missing shards in {index.name}'
print('Model directory:', root)
print('Recorded model label:', identity['label'])
print('File checks passed; the pilot must still validate model/tokenizer loading.')
PY
kubectl -n "$MATH_EVAL_NAMESPACE" exec "$MATH_EVAL_TRANSFER_POD" -- \
  python -c 'from pathlib import Path; import sys, tempfile; p=Path(sys.argv[1]); p.mkdir(parents=True, exist_ok=True); f=tempfile.TemporaryFile(dir=p); f.write(b"ok"); f.close(); print("Output directory writable:", p)' \
  "/results/$MATH_EVAL_OUTPUT_REL"
kubectl -n "$MATH_EVAL_NAMESPACE" delete pod "$MATH_EVAL_TRANSFER_POD" --wait=true
```

Deleting the transfer pod leaves the PVC contents intact. It also avoids holding
a claim open when scheduling the GPU Job. If a check failed, fix the problem
before deleting/recreating your helper as needed.

## 7. Submit the NRP pilot

Render a Job for the uploaded pilot. The renderer only writes JSON; it does not
submit anything. An empty pull-secret argument omits `imagePullSecrets`.

```bash
export MATH_EVAL_SUITE_REL='math-eval/suites/pilot'
export MATH_EVAL_JOB="math-eval-pilot-$(date -u +%Y%m%d%H%M%S)"
export MATH_EVAL_JOB_FILE=".cache/nrp/$MATH_EVAL_JOB.json"
uv run --no-sync python scripts/render_nrp_job.py \
  --image "$MATH_EVAL_IMAGE" \
  --namespace "$MATH_EVAL_NAMESPACE" \
  --name "$MATH_EVAL_JOB" \
  --model-pvc "$MATH_EVAL_MODEL_PVC" \
  --model-relative-path "$MATH_EVAL_MODEL_REL" \
  --suite-pvc "$MATH_EVAL_SUITE_PVC" \
  --suite-relative-path "$MATH_EVAL_SUITE_REL" \
  --output-pvc "$MATH_EVAL_OUTPUT_PVC" \
  --output-relative-path "$MATH_EVAL_OUTPUT_REL" \
  --gpu-resource nvidia.com/a100 \
  --batch-size 1 \
  --max-new-tokens 8192 \
  --image-pull-secret "$MATH_EVAL_PULL_SECRET" \
  > "$MATH_EVAL_JOB_FILE"
uv run --no-sync python -m json.tool "$MATH_EVAL_JOB_FILE"
kubectl apply --dry-run=server -f "$MATH_EVAL_JOB_FILE"
kubectl apply -f "$MATH_EVAL_JOB_FILE"
kubectl -n "$MATH_EVAL_NAMESPACE" get pods -l "job-name=$MATH_EVAL_JOB"
```

Check the rendered namespace, claims, paths, image, and requested resources before
applying. The GPU Job mounts model and suite read-only and output read/write.
The container entrypoint runs `math-eval generate`; no Docker daemon or model
server is needed inside NRP. Dependencies are already built into the image.

Once a pod is running, follow its logs:

```bash
kubectl -n "$MATH_EVAL_NAMESPACE" logs -f "job/$MATH_EVAL_JOB"
kubectl -n "$MATH_EVAL_NAMESPACE" wait \
  --for=condition=complete "job/$MATH_EVAL_JOB" --timeout=24h
```

If no pod is running yet, inspect `get pods` and `describe` rather than assuming a
logs error means inference failed. Ctrl-C on `logs -f` stops following logs; it
does not stop the Job. A wait timeout also does not stop the Job. If a Job fails,
`wait` may time out; inspect the Job/pod status and logs directly.

The logs should progress to `Generated 10/10` and print a path like
`/results/math-eval/runs/20261028T000000Z-abcdefgh`. The directory name is the run ID.
The example is illustrative; copy the ID from your actual logs. Save logs while
they are available:

```bash
kubectl -n "$MATH_EVAL_NAMESPACE" logs "job/$MATH_EVAL_JOB" \
  > ".cache/nrp/$MATH_EVAL_JOB.log"
```

A running GPU pod can also be inspected with:

```bash
export MATH_EVAL_GPU_POD="$(kubectl -n "$MATH_EVAL_NAMESPACE" get pods \
  -l "job-name=$MATH_EVAL_JOB" -o jsonpath='{.items[0].metadata.name}')"
kubectl -n "$MATH_EVAL_NAMESPACE" exec "$MATH_EVAL_GPU_POD" -- nvidia-smi
```

The template does not retry failed GPU Jobs (`backoffLimit: 0`), so it will not
silently restart an expensive run. Finished Job objects/pods have a seven-day TTL;
responses on PVC storage remain after those objects are removed.

## 8. Download the completed run

A finished GPU pod is not a running file-transfer server. Recreate the CPU helper
from section 6, using the saved JSON, to read the output PVC:

```bash
kubectl apply -f .cache/nrp/transfer-pod.json
kubectl -n "$MATH_EVAL_NAMESPACE" wait \
  --for=condition=Ready "pod/$MATH_EVAL_TRANSFER_POD" --timeout=10m
kubectl -n "$MATH_EVAL_NAMESPACE" exec "$MATH_EVAL_TRANSFER_POD" -- \
  ls "/results/$MATH_EVAL_OUTPUT_REL"
export MATH_EVAL_RUN_ID='paste-the-run-directory-name-from-your-logs'
mkdir -p runs
uv run --no-sync python -c 'from pathlib import Path; import sys; assert not Path(sys.argv[1]).exists(), "Local run directory already exists; use a new destination"' \
  "runs/$MATH_EVAL_RUN_ID" &&
kubectl -n "$MATH_EVAL_NAMESPACE" cp --retries=3 \
  "$MATH_EVAL_TRANSFER_POD:/results/$MATH_EVAL_OUTPUT_REL/$MATH_EVAL_RUN_ID" \
  "runs/$MATH_EVAL_RUN_ID"
kubectl -n "$MATH_EVAL_NAMESPACE" delete pod "$MATH_EVAL_TRANSFER_POD" --wait=true
```

Use a new local destination; do not merge a downloaded run into an existing run
folder. If PVC names or helper settings changed, rerender the transfer JSON first.
Download the **whole run directory**, not just `predictions.jsonl`.

```bash
uv run --no-sync python - "$MATH_EVAL_RUN_ID" <<'PY'
import json
from pathlib import Path
import sys
root = Path('runs') / sys.argv[1]
run = json.loads((root / 'run.json').read_text())
print('Status:', run['status'], 'subset:', run['subset'])
print('Counts:', run['suite_counts'])
assert run['status'] == 'generated', 'Generation failed or is incomplete; inspect logs/run.json'
assert (root / 'predictions.jsonl').is_file(), 'Predictions were not copied'
PY
```

A failed generation directory may contain partial predictions. It cannot receive
a completed benchmark score and should not be patched by hand to look complete.

## 9. Grade and read the results

Run on your laptop with Docker started, using the **same pilot suite**:

```bash
uv sync --locked --extra grading
docker info
uv run --no-sync math-eval grade \
  --suite suites/pilot \
  --run "runs/$MATH_EVAL_RUN_ID"
```

Harbor starts CPU-only task containers and uploads each saved answer. It runs
`tests/test.sh`, which launches `grade.py` and collects its reward and diagnostics.
This step neither loads Qwen nor calls a model endpoint. The first grader build
needs access to its Python base image and pinned math packages.

Inspect the five separate scores:

```bash
uv run --no-sync python - "$MATH_EVAL_RUN_ID" <<'PY'
import json
from pathlib import Path
import sys
summary = json.loads((Path('runs') / sys.argv[1] / 'summary.json').read_text())
print('Status:', summary['status'], 'subset:', summary['subset'])
for benchmark, score in summary['benchmarks'].items():
    accuracy = score['accuracy']
    display = 'incomplete' if accuracy is None else f'{100 * accuracy:.2f}%'
    print(benchmark, display, f"{score['correct']}/{score['expected']}",
          'errors=', score['errors'], 'truncated=', score['truncated'])
PY
```

| Run artifact | Contents |
| --- | --- |
| `run.json` | Model/export identity, suite identity, settings, versions, generation status |
| `predictions.jsonl` | Original/rendered prompts, full responses, extracted answers, token counts, finish reasons |
| `grading/` | Harbor jobs, trial logs, and verifier results |
| `summary.json` | Five scores and per-question diagnostics; created after grading |

A pilot verifies that loading, generation, transfer, and grading work. Low scores
alone do not mean the harness is broken; inspect raw responses and truncation.
The pilot's `subset: true` means its scores are not full benchmark results.
Verifier/infrastructure errors make a benchmark's accuracy `null` and the command
unsuccessful. A wrong or malformed model answer receives zero without becoming
an infrastructure error.

## 10. Run the full suite and evaluate later checkpoints

After the pilot completes and its outputs make sense, switch the uploaded suite
path and choose a new Job name:

```bash
export MATH_EVAL_SUITE_REL='math-eval/suites/math-five'
export MATH_EVAL_JOB="math-eval-full-$(date -u +%Y%m%d%H%M%S)"
export MATH_EVAL_JOB_FILE=".cache/nrp/$MATH_EVAL_JOB.json"
```

Rerun the **renderer and submission block in section 7** with these variables,
then monitor it. A full run must reach `Generated 1437/1437`. Follow section 8 to
retrieve its new run ID, then grade with the full local suite:

```bash
uv run --no-sync math-eval grade \
  --suite suites/math-five \
  --run "runs/$MATH_EVAL_RUN_ID"
```

The downloaded `run.json` and resulting summary should report `subset: false`
and the five full counts shown at the top. Keep the image, suite, token budget,
thinking mode, and sampling settings fixed when comparing checkpoints.

For each later model: finish/export it, update `MATH_EVAL_MODEL_PVC` and/or
`MATH_EVAL_MODEL_REL`, check the export, and submit a newly named Job using the
same full suite. Rerender the CPU helper if a PVC changes. Each evaluation gets a
new run directory automatically; the model label comes from export metadata or
its directory. Rebuild/push a new image tag only when code or dependencies change,
not for each new checkpoint. This is explicit invocation per model; there is no
background checkpoint watcher.

### Cleanup after saving results

Keep the local run directory, exact suite, source/lockfile version, submitted Job
JSON, image tag/digest, and logs for the comparison. To remove your own finished
Job/pods or stop an unwanted run:

```bash
kubectl -n "$MATH_EVAL_NAMESPACE" delete job "$MATH_EVAL_JOB"
kubectl -n "$MATH_EVAL_NAMESPACE" delete pod "$MATH_EVAL_TRANSFER_POD" --ignore-not-found
```

Deleting a running Job stops its generation; that partial run cannot be resumed
by this implementation. These commands do not delete PVCs. Do not delete a shared
lab PVC as cleanup. Retain or remove old data according to your lab's storage plan.

## Other execution options

### A GPU machine that also has Docker

From this repository on that machine:

```bash
uv sync --locked --extra inference --extra grading
docker info
uv run --no-sync math-eval evaluate \
  --suite suites/math-five \
  --model /path/to/exported/model \
  --output-root runs
```

`evaluate` generates and then grades. On a standard NRP pod, use `generate` and
local `grade` as above instead of expecting an in-pod Docker daemon.

### An existing GPU pod or terminal

If the approved inference image is already running and has the needed mounts,
run this **inside that GPU pod**, adjusting paths to its actual mounts:

```bash
math-eval generate \
  --suite /suites/math-eval/suites/pilot \
  --model /models/runs/your-completed-model-export \
  --output-root /results/math-eval/runs
```

An arbitrary training image may have incompatible dependencies; use the evaluation
image or install the locked inference environment on local container disk.
Do not install a virtual environment on a shared CephFS PVC.

### Without Docker for grading

```bash
uv run --no-sync math-eval grade \
  --suite suites/math-five \
  --run "runs/$MATH_EVAL_RUN_ID" \
  --backend local
```

The grading extra must be installed. This invokes the same task verifiers as
local Python processes and labels the result `grading_backend: local`. It does
not exercise Harbor/Docker execution. Use a fresh grading attempt as explained
below if the run already has a `grading/` directory.

### Generation settings and Hub models

Defaults: one response per question, batch size 1, greedy decoding, seed 42,
8,192 maximum input tokens, 8,192 new tokens, thinking disabled. CUDA uses BF16
when supported, otherwise FP16. The original SFT evaluation used thinking off;
choose the mode that matches your experiment and keep it fixed for comparisons.

The generation CLI supports `--batch-size`, `--max-new-tokens`,
`--max-input-tokens`, `--temperature`, `--top-p`, `--seed`, `--dtype`, and
`--enable-thinking` / `--no-enable-thinking`. Oversized inputs are rejected, not
silently truncated. This runner does not shard models across multiple GPUs.

The NRP renderer exposes only batch size and new-token budget. To change other
CLI settings, edit the rendered Job's `spec.template.spec.containers[0].args`
before its first submission and keep that document with the results. Add CLI
arguments as separate JSON strings; do not add unsupported flags to the renderer.
To lower host memory/CPU or add node affinity, likewise edit the rendered Job
resources/scheduling fields before submission. A changed Job template needs a
new Job name.

For a Hub-hosted model, `generate --model organization/model --revision COMMIT`
resolves and records its exact revision. The main NRP renderer intentionally uses
a local exported directory; a Hub workflow requires adapting the rendered args
and supplying any required `HF_TOKEN` as a namespace Secret/env reference. Public
local exports do not need an HF token or a serving API key.

## Troubleshooting

First collect status without changing anything:

```bash
kubectl -n "$MATH_EVAL_NAMESPACE" get jobs,pods,pvc
kubectl -n "$MATH_EVAL_NAMESPACE" describe job "$MATH_EVAL_JOB"
kubectl -n "$MATH_EVAL_NAMESPACE" get events --sort-by=.metadata.creationTimestamp
```

Use `kubectl -n "$MATH_EVAL_NAMESPACE" describe pod POD-NAME` for the affected
pod, and `kubectl ... logs POD-NAME` for a pod that actually started. A failed Job
will not restart automatically. Fix the cause and use a new Job name; there is
no automatic resume of a partly generated run.

| Symptom | What to check/do |
| --- | --- |
| `oidc-login` executable missing, login loop, or `Unauthorized` | Install int128/kubelogin, check `KUBECONFIG`/context, and repeat institutional login. Follow NRP's access guide for browser or token-cache issues. |
| `Forbidden` | Confirm membership and permissions in the selected namespace; ask its administrator about the denied operation. |
| Quota exceeded / no pod created | Inspect ResourceQuota and Job events; confirm A100 authorization and available CPU/memory quota. |
| GPU pod stays `Pending` | Inspect scheduling events for GPU availability, quota, resource size, affinity, or conflicting PVC attachment. A `logs` command cannot fix scheduling. |
| PVC stays `Pending` / mount error | Check claim names, namespace, storage class, capacity, region, and access mode. All three claims must be mountable by the same pod. |
| `ImagePullBackOff` / `ErrImagePull` | Confirm the pushed tag/digest exists, `linux/amd64` is present, and the pull Secret is correct in this namespace. |
| `exec format error` | Rebuild for the GPU node's CPU architecture; the main example is `linux/amd64`, not an Apple Silicon-only image. |
| CUDA unavailable / driver too old | Confirm a GPU was requested and inspect `nvidia-smi` while running. The locked Linux PyTorch build uses CUDA 12.8; select a compatible node/driver with your lab. |
| CUDA out of memory | Keep batch size 1, reduce the new-token/input budgets deliberately, or choose adequate GPU memory. A too-large model needs a different runner; requesting extra GPUs alone does not enable sharding. |
| Model/tokenizer load failure | Use a completed full export, all shards, tokenizer, and chat template; raw optimizer checkpoints or adapter-only files are insufficient. Remote custom model code is disabled. |
| Missing input/output path | Translate paths from the PVC root to `/models`, `/suites`, `/results`; a laptop path or a training pod's `/data` prefix is not automatically valid here. |
| Transfer pod stopped / copy fails | Recreate your CPU helper, check its two-hour deadline and `tar`, and verify mount/write permissions. Do not try `kubectl cp` from a completed GPU container. |
| Suite hashes differ / extra suite files | Use the exact original suite. Check for incomplete copies or macOS metadata; preserve the old suite and transfer a fresh validated directory rather than editing its manifest. |
| Generation did not complete | Inspect `run.json` and logs. Keep partial artifacts for debugging, then launch a fresh generation run; do not fabricate missing predictions. |
| Missing/duplicate/modified predictions | Download the entire completed run again to a new local folder. Never mix run folders or edit response files. |
| Docker connection error | Start Docker Desktop / check Docker Engine and rerun `docker info`. The GPU generation output can still be graded later. |
| Slow first Harbor run / grader build failure | Check laptop Internet access, base-image/PyPI downloads, Docker disk space, and the benchmark log under `grading/`. Lower `--concurrency` if local resources are constrained. |
| Accuracy is `null`, summary incomplete | Inspect `summary.json` per-question errors, `grading/BENCHMARK.log`, and trial verifier logs. Infrastructure errors are not model mistakes. |
| Many zeros / truncated responses | Inspect full responses, extracted answers, chat template, thinking mode, and token budget. A low score can be a valid model result. |
| Grading refuses an existing attempt | Preserve it and create a clean retry as below; copying the folder alone also copies `grading/`. |

To retry **grading only** after fixing an infrastructure problem, preserve the
original run and copy only the generation artifacts into a fresh local folder:

```bash
export MATH_EVAL_RETRY_DIR="runs/$MATH_EVAL_RUN_ID-regrade-$(date -u +%Y%m%d%H%M%S)"
mkdir "$MATH_EVAL_RETRY_DIR" &&
cp "runs/$MATH_EVAL_RUN_ID/run.json" "$MATH_EVAL_RETRY_DIR/run.json" &&
cp "runs/$MATH_EVAL_RUN_ID/predictions.jsonl" "$MATH_EVAL_RETRY_DIR/predictions.jsonl" &&
uv run --no-sync math-eval grade \
  --suite suites/math-five \
  --run "$MATH_EVAL_RETRY_DIR"
```

For a pilot retry, use `--suite suites/pilot` instead. This preserves the original
model/run identity and prediction hash; it creates another grading attempt, not
new model responses. It cannot repair incomplete or changed generation.

## Interpretation, tests, and repository layout

This is single-response, unaided math evaluation with a shared custom protocol,
not an exact reproduction of each paper's prompt, scorer, or multi-sample results.
AIME uses integer comparison; other benchmarks use exact matching and Math-Verify
0.9.0. Valid alternative expressions can still be misinterpreted by an automatic
grader, and source labels may contain errors. Review diagnostics when auditing
scores. Check training overlap before claiming held-out performance.

Generated task `solve.sh` files are known-answer oracles for harness checks; real
model evaluation uses `SavedPredictionAgent`. `test.sh` launches the verifier.
The small `tasks/aime-example` remains for learning these roles. See
[the implementation walkthrough](docs/checkpoint-evaluation.md) for module-level
explanations and interview talking points.

```bash
uv sync --locked --extra grading --extra inference
uv run --no-sync python -m unittest discover -s tests -v
```

The tiny Qwen CPU test uses random weights to check plumbing, not trained-model
quality. Without the inference extra it is skipped. Oracle/no-op runs likewise
validate the harness rather than measure Qwen's reasoning ability.

```text
src/math_eval/            Adapters, model runner, replay agent, grading, summaries
scripts/render_nrp_job.py NRP GPU Job renderer (JSON; no automatic submission)
docker/                  Inference image definition
tests/                   Pipeline, teaching-grader, tiny-model checks
docs/                    Implementation explanation
tasks/aime-example/      Small teaching task
suites/                  Generated benchmark tasks; ignored by Git
runs/                    Model outputs and grading results; ignored by Git
.cache/nrp/              Local manifests/config/logs; ignored by Git
```

Keep code/lockfile versions and experiment settings with results. Virtual
environments and generated data stay local; cloning does not restore them.
The image and NRP Job must be validated with your actual model and namespace;
local tests alone do not establish cluster compatibility.
