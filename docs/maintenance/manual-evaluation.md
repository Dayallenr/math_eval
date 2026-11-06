# Manual evaluation and debugging

For normal model evaluations, follow the [automated NRP workflow](../../README.md).
It runs generation and Harbor grading on the cluster. This guide covers the
existing CLI commands for deliberately running those stages yourself, plus
private-image authentication and troubleshooting.

## Prepare a local environment and suite

Use the [README's installation steps](../../README.md#setup), then
install the extras required for local inference and symbolic grading:

```bash
uv sync --locked --extra inference --extra grading
```

The prepared NRP suites are described in [shared storage](../nrp-shared-suites.md).
For local debugging, prepare the same pinned sources in a fresh directory below.
Normal evaluations stage the existing archive automatically in the remote host.

To prepare a new suite instead, use a new output directory:

```bash
uv run --no-sync math-eval prepare --output suites/new-pilot --limit 2
uv run --no-sync math-eval prepare --output suites/new-full
```

Preparation downloads the seven pinned sources and creates Harbor tasks,
`questions.jsonl`, and a hashed `suite.json` manifest. It refuses to overwrite
an existing directory. A pilot contains the first two questions per benchmark
and is marked as a subset. A newly prepared suite is not automatically uploaded
to NRP or selected by `nrp.toml`; the normal profile continues to use its pinned
stored archive.

## Generate answers

On a GPU machine with the model export and suite available:

```bash
uv run --no-sync math-eval generate \
  --suite suites/pilot-seven \
  --model /path/to/completed-model-export \
  --output-root runs
```

The command prints the unique run directory. It contains `run.json` and
`predictions.jsonl`; preserve both together and do not edit the responses.
Use `suites/math-seven` for the full suite. The export must contain full model
weights, `config.json`, tokenizer files, and a chat template.

Inside an existing GPU pod running the evaluation image, use its mounted paths:

```bash
math-eval generate \
  --suite /suites/math-eval/suites/pilot-seven \
  --model /models/exports/model-v1 \
  --output-root /results/math-eval/runs
```

Adjust these paths to that pod's actual mounts. This starts generation only;
it does not submit a separate Harbor grading job. Use the automated command in
the README when both stages should run on NRP without manual coordination.

Defaults match the lab's Qwen3-8B non-thinking profile: batch size 1, seed 42,
8,192 maximum input tokens, 12,288 new tokens, temperature 0.7, top-p 0.8,
top-k 20, and min-p 0. CUDA selects BF16 when supported, otherwise FP16.
Keep these settings fixed when comparing checkpoints. Inputs exceeding the
configured limit are rejected rather than silently truncated.

Use `math-eval generate --help` for overrides. Public Hub models are also
supported by this manual command with `--model ORGANIZATION/MODEL --revision COMMIT`;
the exact resolved revision is recorded. The automated NRP command instead takes
a completed model export on a PVC. This runner uses one GPU and does not shard
models when additional GPUs are requested.

## Grade saved answers

For local Harbor execution, start Docker Engine/Desktop, then run:

```bash
docker info
uv run --no-sync math-eval grade \
  --suite suites/pilot-seven \
  --run runs/YOUR-RUN-ID \
  --concurrency 2
```

The suite must match the one used for generation. Harbor runs the replay agent
and task verifiers; it makes no model calls. Inspect `summary.json` for the
separate benchmark scores and question results, and `grading/` for trial logs.
An ordinary NRP pod has no Docker daemon; the automated NRP provider creates
Kubernetes task pods instead.

For debugging the same verifiers without Harbor or Docker:

```bash
uv run --no-sync math-eval grade \
  --suite suites/pilot-seven \
  --run runs/YOUR-RUN-ID \
  --backend local
```

This records `grading_backend: local` and does not validate Harbor execution.
The automated NRP workflow always uses Harbor.

A GPU machine with Docker can combine generation and grading:

```bash
uv run --no-sync math-eval evaluate \
  --suite suites/pilot-seven \
  --model /path/to/completed-model-export \
  --output-root runs
```

### Retry grading without generating again

An existing `grading/` directory prevents accidental overwrite. Preserve it and
copy only the completed generation artifacts into a new directory:

```bash
export MATH_EVAL_RUN_ID='YOUR-RUN-ID'
export MATH_EVAL_RETRY_DIR="runs/$MATH_EVAL_RUN_ID-regrade-$(date -u +%Y%m%d%H%M%S)"
```

```bash
(
  set -eu
  mkdir "$MATH_EVAL_RETRY_DIR"
  cp "runs/$MATH_EVAL_RUN_ID/run.json" "$MATH_EVAL_RETRY_DIR/run.json"
  cp "runs/$MATH_EVAL_RUN_ID/predictions.jsonl" "$MATH_EVAL_RETRY_DIR/predictions.jsonl"
  uv run --no-sync math-eval grade \
    --suite suites/pilot-seven --run "$MATH_EVAL_RETRY_DIR"
)
```

For a full-suite run, substitute `suites/math-seven`. This preserves model
identity and the prediction hash. It cannot repair incomplete generation.

## Private image only: configure pull credentials

For public images, set `image_pull_secret = ""` in `nrp.toml` and skip this
section. For private images, reuse a namespace Secret authorized to read both
the runner and task images, or create a dedicated one below. Laptop Docker login
does not configure NRP image pulls.

Run after the README's NRP login setup. The token is entered through a hidden
prompt; use registry credentials with permission to pull both images.

```bash
export MATH_EVAL_REGISTRY='gitlab-registry.nrp-nautilus.io'
export MATH_EVAL_NAMESPACE='wang-research-lab'
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
subprocess.run(['kubectl', '--context', 'nautilus', 'apply', '-f', '-'],
               input=json.dumps(secret), text=True, check=True)
PY
kubectl --context nautilus -n "$MATH_EVAL_NAMESPACE" get secret "$MATH_EVAL_PULL_SECRET"
```

This creates or updates the named Secret. Set `image_pull_secret` in `nrp.toml`
to that name. Never commit the token or kubeconfig. See
[Kubernetes private image authentication](https://kubernetes.io/docs/tasks/configure-pod-container/pull-image-private-registry/).

## Troubleshooting

For NRP jobs, first use the
[status and log commands](../nrp-remote-tmux.md#terminal-and-progress-tips).
Inspect the affected pod's events with `kubectl --context nautilus -n
wang-research-lab describe pod POD-NAME`.

| Symptom | What to check |
| --- | --- |
| Unauthorized / missing `oidc-login` | Install int128/kubelogin, check `KUBECONFIG` and context, and repeat institutional login. |
| Forbidden | Ask the namespace administrator about the specific denied operation; evaluation accounts need the documented roles. |
| Pending pod / quota exceeded | Inspect scheduling events, GPU resource quota, memory requests, node affinity, and conflicting PVC attachments. |
| PVC mount failure | Check claim names, namespace, Bound status, and access modes. Automated output storage requires ReadWriteMany. |
| ImagePullBackOff | Confirm both image references exist and the namespace pull Secret can read them. |
| Exec format error | Publish the images for the node architecture; the supported runner target is Linux amd64. |
| CUDA unavailable / driver incompatibility | Confirm the GPU resource request and a driver compatible with the locked CUDA 12.8 Torch build. |
| CUDA out of memory | Use batch size 1, choose adequate VRAM, or deliberately reduce token budgets. Extra GPUs alone do not enable sharding. |
| Model/tokenizer cannot load | Check the full export, all weight shards, tokenizer, and chat template; adapter-only or optimizer checkpoints are insufficient. |
| Suite fingerprints differ | Restore the original pinned suite; do not edit its files or manifest. |
| Missing/duplicate/modified predictions | Keep run folders separate; recover the complete original generation artifacts. |
| Generation failed | Keep diagnostics, fix the cause, and submit a fresh run. Partial generation cannot receive a complete score. |
| Docker connection error | Start Docker for manual Harbor grading; automated NRP task pods do not need it. |
| Incomplete summary / null accuracy | Inspect verifier errors and Harbor logs. Infrastructure failures are not ordinary wrong answers. |
| Many zeros / cutoff responses | Inspect full responses, extracted answers, chat template, thinking mode, and output budget. |
| Existing grading attempt | Use the grading-only retry procedure above; preserve the previous attempt. |

The [pipeline explanation](../architecture.md) covers scoring, provenance,
module roles, and integrity checks. The
[benchmark source record](../benchmark-sources.md) identifies the exact releases.
