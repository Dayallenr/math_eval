# Math evaluation with laptop-managed Harbor and NRP

This branch, `laptop-orchestration`, runs the coordinator and Harbor on your
laptop using your existing NRP login. Model generation runs in an NRP GPU Job.
After that Job finishes and releases its GPU, Harbor grades the saved answers in
isolated NRP CPU task pods. Seven benchmark scores are saved both locally and on
the lab's shared storage.

**When coordinating on your laptop, keep it awake, connected, and the command running.** If interrupted,
resume the same evaluation ID; a completed generation is reused. The separate
fully NRP-managed workflow is preserved on `nrp-controller-on-hold` while the
team decides whether to install its service-account roles.

```text
Laptop coordinator (your kubectl login)
    -> temporary CPU pod stages the request and copies the pinned suite
    -> NRP GPU Job loads the model once and saves its responses
    -> GPU Job finishes
    -> laptop fetches saved responses and runs Harbor
    -> Harbor launches up to 2 NRP CPU task pods at a time
    -> laptop saves seven scores and uploads reports/diagnostics to shared-pvc
```

No new Roles, RoleBindings, or dedicated ServiceAccounts are created or required
by this workflow. Your user account still needs existing permission to create,
inspect, exec into, and delete its Jobs/pods, and access the specified PVCs and
image-pull Secret. Credentials stay in your local kubeconfig; no user token is
copied into a pod.

## Benchmarks and baseline models

| Benchmark | Selection | Questions |
| --- | --- | ---: |
| MATH-500 | Test | 500 |
| BeyondAIME | Test | 100 |
| OlymMATH | English HARD | 100 |
| HMMT February 2025 | MathArena release | 30 |
| HMMT February 2026 | MathArena release | 33 |
| AIME 2025 | MathArena release, I + II | 30 |
| AIME 2026 | MathArena release, I + II | 30 |
| Total | | 823 |

The fourteen-question pilot has two questions from each benchmark. The pinned
suite archive is already on `wang-research-lab/shared-pvc`. The laptop workflow
copies it automatically through a short-lived CPU pod; you do not manually
prepare or download datasets. This laptop-coordinated mode does keep a local
suite copy so Harbor can upload task instructions/tests. It never downloads
model weights to the laptop. See [storage](docs/nrp-shared-suites.md) and
[dataset provenance](docs/benchmark-sources.md).

Evaluate the official `Qwen/Qwen3-8B` unchanged, SFT with labels, SFT without
labels, and the team's GRPO checkpoint with identical datasets and decoding
settings. Each must be a completed Hugging Face model export on an accessible
NRP PVC. "Unchanged" means no additional lab training; the official checkpoint
is already pretrained and post-trained. Confirm which SFT checkpoint the GRPO
run starts from. Model training is outside this repository.

## 1. Install the laptop tools

You need Git, `uv`, Python 3.12+, `kubectl`, and int128's OIDC `kubelogin`. On
macOS, Homebrew can install the CLIs:

```bash
brew install git uv kubectl kubelogin
```

From this repository on branch `laptop-orchestration`:

```bash
uv python install 3.12
uv sync --python 3.12 --locked --extra grading
uv run --no-sync math-eval nrp-laptop --help
```

Local Torch/Transformers, a GPU, and Docker are not required. Both published
images are reused. The grading extra is small; the large inference dependencies
stay in the published NRP runner image. Use macOS/Linux or WSL2 for this branch.

### Running the coordinator in remote tmux

To run the coordinator inside an NRP CPU pod so your laptop can sleep or shut
down, follow the [remote tmux instructions](docs/nrp-remote-tmux.md). The helper
creates a CPU-only Deployment, installs the tools, and uses your existing NRP
user login through the standard browser flow. It does not install administrator
roles. Model files and benchmark suites are already staged on NRP.

Remote tmux survives a laptop disconnect while its pod stays running. The
persistent workspace supports resuming after pod replacement. Local tmux does
not keep a sleeping laptop executing. The laptop instructions below remain
available for local coordination.

## 2. Connect to NRP

Keep an existing working kubeconfig. Check your access:

```bash
kubectl --context nautilus -n wang-research-lab get pvc
kubectl --context nautilus -n wang-research-lab get resourcequota
```

If you have not configured NRP, first complete [NRP access setup](https://nrp.ai/documentation/userdocs/start/getting-started/).
Use a separate configuration file to avoid overwriting another cluster:

```bash
mkdir -p .cache/nrp
curl -fSL https://nrp.ai/config -o .cache/nrp/kubeconfig
chmod 600 .cache/nrp/kubeconfig
export KUBECONFIG="$PWD/.cache/nrp/kubeconfig"
kubectl --context nautilus -n wang-research-lab get pvc
```

The first request may open an institutional login in your browser. Re-export
`KUBECONFIG` in a new terminal if using that separate file. The coordinator and
Harbor explicitly use the configured context; they do not switch your global
current context. No `math-eval nrp setup` command is needed on this branch.

## 3. Configure the existing images and model location

This checkout already has an ignored `nrp.toml` with the published images. For a
new checkout, copy the template once:

```bash
cp nrp.example.toml nrp.toml
```

Preserve an existing configured file. The template pins both validated image
digests, the current seven-benchmark archive and its hashes, namespace
`wang-research-lab`, suite/output PVC `shared-pvc`, and the namespace pull Secret
`nrp-registry-dayallen`. No rebuild or push is needed for a new model. For another
namespace, arrange its storage and image access rather than reusing these names
blindly. For private-image authentication errors, use the
[hidden credential prompt](docs/manual-evaluation.md#private-image-only-configure-pull-credentials).

Obtain each model's **PVC name and path relative to that PVC's root**. For
example, `model-pvc:exports/sft-with-labels` means the GPU pod loads that directory
from `model-pvc`. Include full model weights, config, tokenizer, and chat template;
merge/export adapter-only or raw optimizer checkpoints first.

The current requested baseline is **your Qwen3-8B checkpoint trained with SFT
without dataset labels**. It is already staged at
`math-eval-models:Qwen3-8B-SFT-no-labels`; see its
[verified source and storage location](docs/nrp-shared-suites.md#requested-sft-baseline-without-dataset-labels).
You do not need to create its folder or download it again. The unchanged official
Qwen3-8B release is also available at `math-eval-models:Qwen3-8B` as a control
without additional lab training. For another model, supply its completed export's PVC and folder path.
The evaluation loads that existing directory directly, without an inference
endpoint.

The Qwen3-8B comparison settings remain: thinking off, temperature 0.7, top-p 0.8,
top-k 20, min-p 0, seed 42, batch size 1, at most 8,192 input tokens and 12,288 new
tokens. Input plus output must fit the model's context. Keep these settings fixed
across the four model versions.

The GPU Job requests 1 generic NVIDIA GPU with advertised VRAM strictly above
40 GiB, 4 CPUs, 48 GiB host RAM, and 20 GiB temporary storage. Eligible generic
L40/L40S GPUs can satisfy it. Other NRP pools use distinct resource names; set
`gpu_resource` to `nvidia.com/rtxa6000` for A6000 or `nvidia.com/a100` for A100,
with the appropriate namespace quota. Actual model fit and CUDA compatibility
still need a pilot. See [NRP GPU instructions](https://nrp.ai/documentation/userdocs/running/gpu-pods/).

## 4. Run the pilot, then the full suite

Replace the model PVC/path below with a real completed export. Preview first:

```bash
uv run --no-sync math-eval nrp-laptop run \
  --model MODEL-PVC:exports/MODEL --pilot --dry-run
```

The preview contacts no cluster and submits no work. It shows one GPU generation
Job and three short-lived CPU transfer pods. Harbor will create its CPU task pods
later. No controller/CPU Harbor Job or Role/ServiceAccount manifests are included.

Run the fourteen-question pilot:

```bash
uv run --no-sync math-eval nrp-laptop run \
  --model MODEL-PVC:exports/MODEL --pilot
```

The command prints an evaluation ID and advances through staging, generation,
fetching, grading, and upload. Keep it running; on macOS you can wrap the command
with `caffeinate -i` to prevent idle sleep. GPU generation continues on NRP if the
laptop disconnects, but the laptop must reconnect/resume to start grading and
finish report upload. Helper pods are removed after each transfer, rather than
remaining idle while generation runs.

Inspect the pilot's reports and model responses before submitting the full suite:

```bash
uv run --no-sync math-eval nrp-laptop run \
  --model MODEL-PVC:exports/MODEL
```

Repeat with a different completed model path for each of the four comparison
models. Every run gets its own directory and seven separate benchmark scores.
No model is trained or updated during evaluation. Gold answers and oracle
`solve.sh` scripts are not included in model prompts.

## 5. Results, interruptions, and cleanup

```bash
uv run --no-sync math-eval nrp-laptop status EVALUATION-ID
uv run --no-sync math-eval nrp-laptop results EVALUATION-ID
```

The default local workspace is `runs/nrp-laptop/EVALUATION-ID/`. It contains
`request.json`, laptop `pipeline.json`, `report.json`, `benchmarks.csv`, the
verified suite, and `run/` with responses, per-question `summary.json`, and Harbor
logs. After successful upload, matching reports and compressed grading
logs also live under `shared-pvc/math-eval/evaluations/EVALUATION-ID/`, alongside
the original GPU generation files. These local files are ignored by Git and
must be retained to resume. Use `--workspace PATH` before the subcommand to keep
them elsewhere.

If the terminal closes or your connection drops, use the same workspace and ID:

```bash
uv run --no-sync math-eval nrp-laptop resume EVALUATION-ID
```

This uses the saved immutable configuration, reconnects to an existing matching
GPU Job, and reuses completed generation. Changed profile settings do not silently
change an existing evaluation. If grading was interrupted, the command preserves
its files and asks for an explicit retry:

```bash
uv run --no-sync math-eval nrp-laptop resume EVALUATION-ID --retry-grading
```

That preserves the previous grading attempt and retries the saved answers. A
failed or cancelled GPU Job is not automatically regenerated. Choose a new
evaluation after fixing its cause. Failed/missing verifier runs remain incomplete;
they are not converted into model accuracy zero or silently graded locally.

To stop, press Ctrl+C in the coordinating terminal, then cancel from that terminal
or another one after the coordinator has exited:

```bash
uv run --no-sync math-eval nrp-laptop cancel EVALUATION-ID
```

Cancellation deletes only this evaluation's generation Job and task/helper pods.
It preserves PVC data, model weights, datasets, and local diagnostics. It cannot
cancel a different request using the same Job name. Completed reports remain
available. Kubernetes generation Jobs otherwise expire seven days after finishing;
local and PVC results remain. A terminal disconnect does not itself cancel the
GPU Job.

For a pending/failed generation Job, inspect its logs and events:

```bash
kubectl --context nautilus -n wang-research-lab logs job/EVALUATION-ID-generate
kubectl --context nautilus -n wang-research-lab get pods -l math-eval/evaluation=EVALUATION-ID
```

## Validation and the workflow on hold

The published runner passed 31 tests in an NRP CPU Job. The published task image
passed seven live Harbor grading checks, including symbolic equivalence and
wrong/missing answers. This branch also passed a live fourteen-response scripted workflow, including
resume, Harbor grading, and report upload, plus recovery/transfer/ordering tests; see
[the validation record](docs/nrp-verification.md). Scripted fixture results are
harness checks, not scores for any of the four real models. A real GPU pilot is
still required once a model export is available.

To run the local regression suite:

```bash
uv run --no-sync python -m unittest discover -s tests -v
```

The tiny real Qwen test skips without the optional inference dependencies; the
published runner has already passed it. No large local inference installation is
needed for laptop-managed evaluation.

The NRP-controller implementation remains available through `math-eval nrp` and
on branch `nrp-controller-on-hold`. It is not used by `nrp-laptop`; no roles are
installed automatically. Its [architecture description](docs/checkpoint-evaluation.md)
and [role manifest](docs/nrp-rbac.json) are retained for the team's later decision.
The optional [GitLab builder](docs/gitlab-runner-build.md) is only for future
image publication; this workflow reuses the existing images.
