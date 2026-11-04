# Math evaluation with Harbor on NRP

Provide a completed model export on NRP storage with one command. A controller
running on NRP starts GPU generation, then a CPU Harbor grading job. Harbor
creates isolated Kubernetes task pods, runs the replay agent and each verifier,
and collects the results. Seven separate benchmark scores and diagnostics remain
on shared storage. Your laptop only submits the request and displays results.

```text
Model export on PVC -> NRP controller
                         |
                    GPU generation
                         |
                    Harbor CPU job
                         |
               isolated task pods + test.sh
                         |
               seven benchmark scores on PVC
```

The controller continues after you close your terminal. GPU generation finishes
before CPU grading starts, so the grading phase does not reserve a GPU. The
installed Harbor version is pinned to 0.23.0. NRP grading records
`grading_backend: harbor` and `harbor_environment: math_eval.harbor_nrp:NRPEnvironment`.

| Benchmark | Selection | Questions |
| --- | --- | ---: |
| MATH-500 | Test | 500 |
| BeyondAIME | Test | 100 |
| OlymMATH (en-hard) | English HARD only | 100 |
| HMMT February 2025 | MathArena release | 30 |
| HMMT February 2026 | MathArena release | 33 |
| AIME 2025 | MathArena release, I + II | 30 |
| AIME 2026 | MathArena release, I + II | 30 |
| Total | | 823 |

The full suite, fourteen-question pilot, and pinned archive are already on
`wang-research-lab/shared-pvc`. Do not prepare, upload, or download benchmark files
for the normal NRP workflow. See [the storage record](docs/nrp-shared-suites.md).
The benchmark selection matches the seven names in the supplied screenshot, using
all questions rather than its 20-question samples.
See [dataset sources and pinned revisions](docs/benchmark-sources.md). OlympiadBench and OlymMATH
English EASY are excluded. Old five-benchmark scores are not comparable to this suite.
Each question is an independent prompt; model weights stay fixed. Reference
answers and oracle scripts are never passed to Qwen.

**Deployment status:** both images are published. The runner passed all 31
regression tests on an NRP CPU pod with no skips, including tiny Qwen3 generation.
The published math-task image passed seven live Harbor grading checks on NRP,
covering symbolic equivalence, integer answers, wrong answers, and missing answers.
Those checks used scripted responses, not model benchmark scores. Production use
still needs the namespace roles installed and a real GPU pilot that validates
generation and all scorers. See the [verification record](docs/nrp-verification.md).

For each new model after setup, follow [Submit a completed model](#5-submit-a-completed-model).
The first four sections cover prerequisites and reusable setup.

## 1. Install local tools

Use Bash/zsh on macOS or Linux, or WSL2 on Windows. You need Git, `uv`, Python
3.12+, `kubectl`, and the int128 OIDC `kubelogin` plugin. Docker is needed by the
person building the two images once; users of already-published images need no
local Docker daemon for NRP evaluation.

On macOS with Homebrew, `brew install git uv kubectl kubelogin` installs the CLI
tools. Install/start [Docker Desktop](https://docs.docker.com/desktop/) if building
images. Other platforms can follow [uv installation](https://docs.astral.sh/uv/getting-started/installation/),
[kubectl installation](https://kubernetes.io/docs/tasks/tools/), and
[int128/kubelogin](https://github.com/int128/kubelogin).

Clone/pull the branch containing this pipeline, then run commands from its root:

```bash
git clone https://github.com/Dayallenr/math_eval.git
cd math_eval
uv python install 3.12
uv sync --python 3.12 --locked
uv run --no-sync math-eval nrp --help
mkdir -p .cache/nrp
```

An existing checkout can be used instead of cloning. Use the same source/lockfile
version used to build the evaluation image. Local generated suites and model
weights are not needed to submit an NRP evaluation.

## 2. Connect to NRP and choose your settings

Log in at [nrp.ai](https://nrp.ai/) and have your lab administrator add you to
`wang-research-lab`. You need permission to submit Jobs and inspect pods/logs.
The one-time role installation also needs permission to create ServiceAccounts,
Roles, and RoleBindings. See [NRP access setup](https://nrp.ai/documentation/userdocs/start/getting-started/).

If `kubectl` already works, keep its configuration. Otherwise:

```bash
curl -fSL https://nrp.ai/config -o .cache/nrp/kubeconfig
chmod 600 .cache/nrp/kubeconfig
export KUBECONFIG="$PWD/.cache/nrp/kubeconfig"
kubectl config use-context nautilus
kubectl --context nautilus -n wang-research-lab get pvc
kubectl --context nautilus -n wang-research-lab get resourcequota
```

The first authenticated request may open a browser. Re-export `KUBECONFIG` in a
new terminal when using that separate file. Do not overwrite another cluster's
config. The default GPU job requests one generic NVIDIA GPU, 4 CPUs, 48 GiB host RAM,
and 20 GiB temporary storage. Its node affinity requires more than 40 GiB of
GPU memory (`nvidia.com/gpu.memory > 40960`) and an amd64 node. Host RAM and GPU
VRAM are separate. L40/L40S and other compatible generic GPUs above the threshold
can satisfy this request; 40 GB-class GPUs are excluded.

Some GPU pools have distinct NRP resource names, so a single generic request
cannot automatically use every GPU type. In `nrp.toml`, use:

| Target pool | `gpu_resource` |
| --- | --- |
| Generic eligible GPUs, including L40/L40S | `nvidia.com/gpu` (default) |
| RTX A6000 | `nvidia.com/rtxa6000` |
| A100 above the memory threshold | `nvidia.com/a100` |

`min_gpu_memory_mib = 40960` keeps the strict memory constraint for all pools.
Confirm namespace quota/access for the selected resource; A100 access needs
specific quota. See [NRP GPU resource instructions](https://nrp.ai/documentation/userdocs/running/gpu-pods/).
Actual model/KV-cache fit, CUDA drivers, and supported dtype still need a pilot;
the memory constraint alone cannot guarantee every model fits. The locked Linux
Torch build uses CUDA 12.8.

## 3. Publish two images once

Both images are already published and pinned in `nrp.example.toml`. For the
existing lab namespace, reuse those image digests and `nrp-registry-dayallen`;
no image build or push is needed for a new model. The ignored local `nrp.toml`
can be copied from that template. See [direct NRP verification](docs/nrp-verification.md)
for the completed checks and how to verify a future runner image.

For future image builds, the [remote GitLab builder](docs/gitlab-runner-build.md)
is optional and starts only when explicitly requested in the GitLab UI. Pushing
code no longer creates a pipeline, and there is no GitLab `verify-runner` job.
The Docker commands below are another option for a builder with enough space.
The math-task image is a separate publication step.

Skip building if a maintainer supplies compatible published runner and task image
URLs/digests. Otherwise, choose a registry project you can push into. The lab
currently uses `gitlab-registry.nrp-nautilus.io`; its training image is under
`lucas_sjy/harbor_bespoke`. That does not automatically grant your account write
access. Publish separate evaluation image names, preserving training images.
[NRP GitLab registry instructions](https://nrp.ai/documentation/userdocs/development/gitlab/)
explain creating a project and locating its Container Registry URL.

To create your own destination, sign in at
[NRP GitLab](https://gitlab.nrp-nautilus.io/), select **New project → Create blank
project**, name it `math-eval-images`, and select your username (or an existing
lab group where you can create projects). Choose the desired visibility,
initialize with a README, and create it. Open **Deploy → Container Registry** for
its registry URL. The GitLab project namespace can be your username; it does not
need to match Kubernetes namespace `wang-research-lab`. Code can remain on GitHub.
A private project's images need a matching namespace pull Secret.

Replace the image URLs below with your own project and immutable version tag:

```bash
export MATH_EVAL_RUNNER_IMAGE='YOUR-REGISTRY/YOUR-PROJECT/math-eval:nrp-v2'
export MATH_EVAL_TASK_IMAGE='YOUR-REGISTRY/YOUR-PROJECT/math-task:nrp-v2'
docker login YOUR-REGISTRY
docker build --platform linux/amd64 -f docker/inference.Dockerfile \
  -t "$MATH_EVAL_RUNNER_IMAGE" .
docker run --rm --platform linux/amd64 "$MATH_EVAL_RUNNER_IMAGE" nrp --help
docker build --platform linux/amd64 -f docker/task.Dockerfile \
  -t "$MATH_EVAL_TASK_IMAGE" .
docker push "$MATH_EVAL_RUNNER_IMAGE"
docker push "$MATH_EVAL_TASK_IMAGE"
docker buildx imagetools inspect "$MATH_EVAL_RUNNER_IMAGE"
docker buildx imagetools inspect "$MATH_EVAL_TASK_IMAGE"
```

The runner contains the locked model/Harbor code and kubectl 1.34.1, compatible
with the checked NRP 1.34 API. Revisit this pin if the cluster changes. The smaller
task image matches the existing math-task Dockerfile and its pinned scorer.
Neither image contains model weights or benchmark data. Apple Silicon builds
need `linux/amd64`; an x86 Linux builder can avoid slow emulation. The CUDA runner
requires several GB of downloads and additional build/unpack storage. Use a
builder with ample free space (allow at least 20 GB); do not start this build on
an almost-full laptop. NRP GitLab also provides image builders as described in
the linked registry guide. Record image
digests and avoid overwriting a tag used for an experiment.

For private images, use a namespace pull Secret authorized to read **both** URLs.
The existing `nrp-registry` Secret may work for the lab project; otherwise create
a dedicated Secret using the credential-prompt recipe in
[private registry setup](docs/manual-evaluation.md#private-image-only-configure-pull-credentials).
Docker login on your laptop does not configure NRP's image pulls.

## 4. Configure the team profile and controller permissions once

```bash
cp nrp.example.toml nrp.toml
```

If you already have `nrp.toml`, preserve its image and storage settings and update
`suite_archive`, `archive_sha256`, `full_suite`, `full_sha256`, `pilot_suite`, and
`pilot_sha256` from the new example. Old profiles still select the old suite;
changing the code alone does not update an existing ignored profile.

Edit `nrp.toml`:

- Set `image` to the runner image and `task_image` to the math task image.
- Set `image_pull_secret` to the correct Secret, or `""` for public images.
- Keep namespace `wang-research-lab` and suite/output PVC `shared-pvc` for the
  installed shared setup. Output goes into `math-eval/evaluations/`.
- Optionally set `model_pvc` when most exports live on the same claim.
- Select generation settings and timeouts appropriate to the model. Defaults are
  the Qwen3-8B non-thinking profile: temperature 0.7, top-p 0.8, top-k 20,
  min-p 0, seed 42, batch size 1, thinking off, 8,192 maximum input tokens,
  and 12,288 new tokens. Keep settings fixed when comparing checkpoints.
  For thinking mode set `enable_thinking = true`, `temperature = 0.6`, and
  `top_p = 0.95`; keep top-k 20, min-p 0, and the same output budget.
  These sampling profiles follow [Qwen3-8B guidance](https://huggingface.co/Qwen/Qwen3-8B#best-practices).
  The 12,288 output-token limit is the lab budget, including reasoning and final
  answer; Qwen recommends longer budgets generally. Input plus output must fit
  the model context. Length-limited responses are recorded with `finish_reason: length`.
  Generation currently uses Transformers; a vLLM server is not deployed by this workflow.

The file is ignored by Git and contains image/claim/settings references, not
registry tokens. Full paths, archive fingerprints, and settings are documented
in [nrp.example.toml](nrp.example.toml). All claims must exist in the namespace and
be Bound. **Output storage must be ReadWriteMany** because the controller and
worker mount it concurrently. Existing `shared-pvc` supports this. A model claim
may be ReadWriteOnce; only the GPU worker mounts it. Arrange access if training
still holds that claim on another node. No package installation occurs on CephFS.

Review and install the namespace ServiceAccounts and roles:

```bash
uv run --no-sync math-eval nrp setup --dry-run > .cache/nrp/math-eval-rbac.json
kubectl --context nautilus apply --dry-run=server -f .cache/nrp/math-eval-rbac.json
uv run --no-sync math-eval nrp setup
```

The controller account can create/get Jobs. The separate Harbor worker account
can create/get/delete task pods and execute commands in them. Task pods receive
no Kubernetes token, model/output PVCs, or host mounts. Each has a deadline and
an owner reference so cancellation cleans up its task pods. This grants trusted
controller/Harbor code namespace workload permissions; it does not grant agents
those credentials. An administrator may need to install the roles if your account
cannot grant these permissions. No cluster-admin role is requested.

**Checked lab access:** the current user's account can submit the Jobs but was
forbidden from reading/managing Roles and RoleBindings during the server dry run.
Ask a namespace administrator to install the permissions once. For the unchanged
lab namespace/account names, the ready-to-review [RBAC manifest](docs/nrp-rbac.json)
can be applied directly:

```bash
kubectl --context nautilus apply -f docs/nrp-rbac.json
```

If you customize account names or namespace, send the generated
`.cache/nrp/math-eval-rbac.json` instead. Do not grant broad administrator privileges
to work around this requirement.

## 5. Submit a completed model

First finish/export the checkpoint into NRP storage. A usable Hugging Face export
contains `config.json`, full weights (all shards), tokenizer files, and a chat
template. Merge/export LoRA or raw training checkpoints first. For the lab GRPO
repo this is its export step; preserve `export_manifest.json` if present. Keep the
export unchanged during evaluation. This command does not upload local weights
or perform an adapter merge.

Use `PVC:relative/path` to identify the export. A training mount `/data` with an
export at `/data/exports/model-v1` means the relative path is `exports/model-v1`.
The claim and export must be accessible to the GPU worker in the same namespace.
No model endpoint, API key, or manually chosen model label is required.

Start with a fourteen-question pilot, replacing the example model locator:

```bash
uv run --no-sync math-eval nrp run \
  --model qwen-grpo-data:exports/model-v1 --pilot --dry-run \
  > .cache/nrp/pilot-preview.json
uv run --no-sync math-eval nrp run \
  --model qwen-grpo-data:exports/model-v1 --pilot
```

The preview shows controller, GPU, and Harbor worker Jobs; it submits nothing.
The real command checks claims and accounts, validates the controller manifest
with the server, submits one controller Job, and prints an evaluation ID. The
controller then handles all remaining generation and grading stages on NRP.
Save the printed ID. You can close the terminal after submission.

```bash
uv run --no-sync math-eval nrp status YOUR-EVALUATION-ID
uv run --no-sync math-eval nrp results YOUR-EVALUATION-ID
```

`results` displays a report only after successful grading. Low accuracy alone
does not mean the harness failed; inspect extraction, truncation, and diagnostics.
Pilot reports are marked `subset: true` and are not full benchmark results.

After the pilot works, run the full suite:

```bash
uv run --no-sync math-eval nrp run --model qwen-grpo-data:exports/model-v1
```

For every later checkpoint, repeat that command with its new export location.
If `model_pvc` is saved in `nrp.toml`, `--model exports/model-v2` is sufficient.
The image and datasets are reused. The runner currently loads one model onto one
GPU; requesting more GPUs does not add sharding. There is no background watcher
or training-finish trigger: each completed model is explicitly submitted once.

## 6. Results, failures, and cleanup

On the output PVC, every submission gets:

```text
math-eval/evaluations/EVALUATION-ID/
  request.json                Model locator, pinned suite/image references, settings
  pipeline.json               Stage, child Job names, completion/failure
  generation.json             Pointer to the immutable generation run
  report.json                 Seven scores + model/protocol/environment provenance
  benchmarks.csv              Seven benchmark rows for comparing checkpoint results
  runs/RUN-ID/
    run.json                  Model identity, settings, generation/grading state
    predictions.jsonl         Responses, extracted answers, token counts, finish reasons
    summary.json              Full benchmark and per-question results
    grading-diagnostics.tgz   Harbor trial/verifier logs and rewards
```

The archive is copied from the suite PVC to temporary container disk and checked
against its pinned SHA-256. Task inventories and hashes are verified there. The
GPU worker writes responses to persistent storage; after it completes, Harbor
runs the replay agent and `test.sh` in separate task pods. Harbor's logs are
archived before saving to CephFS to avoid thousands of small shared writes.
The controller saves/prints the final report after every question is accounted
for. Infrastructure errors produce failed/incomplete results, never fake zeros
or a completed report. Wrong/missing model answers legitimately score zero.

For live logs or a failed stage:

```bash
kubectl --context nautilus -n wang-research-lab logs -f job/YOUR-EVALUATION-ID -c controller
kubectl --context nautilus -n wang-research-lab logs job/YOUR-EVALUATION-ID-generate
kubectl --context nautilus -n wang-research-lab logs job/YOUR-EVALUATION-ID-grade
kubectl --context nautilus -n wang-research-lab get pods -l math-eval/evaluation=YOUR-EVALUATION-ID
kubectl --context nautilus -n wang-research-lab describe job YOUR-EVALUATION-ID
```

Check pod events for pending GPUs, quota, image pulls, claim attachment, or driver
problems. Inspect `pipeline.json` and retained diagnostics for execution/scoring
errors. Default stage limits are 48 hours for generation and 6 hours for grading;
change them deliberately for slower models. Incomplete generation cannot be
resumed or graded as complete. Automatic retries are disabled to avoid silently
spending GPU time again; fix the cause and submit a new evaluation ID. Existing
runs are preserved. No failed grading stage is silently switched to a local
backend.

To cancel or remove your own evaluation and its child Jobs/pods:

```bash
uv run --no-sync math-eval nrp cancel YOUR-EVALUATION-ID
```

This also stops an active evaluation. It does not delete output data, model
weights, or shared benchmarks. Finished Kubernetes Jobs have a seven-day TTL;
the persistent results remain after Jobs/logs expire. `nrp results` reads the
controller's logs, so after TTL expiry read `report.json` directly from mounted
NRP output storage. Downloading a report/diagnostic archive for inspection is
optional; completing evaluation never requires a laptop copy of the datasets.

For a new coding/terminal benchmark, add its adapter, verifier, and published task
image. The provider accepts an explicit task `docker_image` or a Dockerfile-hash
entry in `[task_images]`; it rejects unmapped Dockerfiles. It supports single
Linux CPU containers, exec and file transfer. Compose/multi-service, Windows,
GPU task environments, and restricted-network policies require additional backend
support. Interactive coding/terminal solving also needs an appropriate Harbor
agent/model connection; the existing math agent only replays saved predictions.

The current math protocol is one unaided response per question with custom
prompts/scorers, not an exact reproduction of each paper. AIME and BeyondAIME use integer
comparison; other datasets use exact matching and pinned Math-Verify 0.9.0.
Review automated-grader limitations and training overlap before interpreting scores.

Implementation: [controller and worker explanation](docs/checkpoint-evaluation.md).
Optional debugging: [manual generation and grading](docs/manual-evaluation.md).
To run the local regression tests:

```bash
uv sync --locked --extra grading --extra inference
uv run --no-sync python -m unittest discover -s tests -v
```

The tiny random Qwen test checks model plumbing, not model quality. The `local`
grading backend remains available for explicit debugging/manual use; the NRP
workflow uses Harbor. The teaching example remains in `tasks/aime-example/`.

## Repository layout

| Path | Purpose |
| --- | --- |
| `src/math_eval/` | Dataset adapters, inference, Harbor grading, reports, and NRP orchestration |
| `docker/` | Runner and task image definitions |
| `tests/` | Grading, model-loading, and orchestration regression checks |
| `tasks/aime-example/` | Small teaching task for understanding the oracle and verifier |
| `docs/` | Pipeline explanation, dataset sources/storage receipts, image build guide, permissions, and debugging |
| `nrp.example.toml` | Team configuration template; copy to ignored `nrp.toml` |
| `scripts/verify_nrp.py` | Verify a published runner directly in an NRP CPU Job |
| `.gitlab-ci.yml` | Optional image builder started explicitly in the GitLab UI |
| `pyproject.toml`, `uv.lock` | Package definition and reproducible dependencies |

Generated suites, runs, Jobs, caches, downloaded build artifacts, and local
configuration are ignored by Git. They are not part of the source tree to review
or upload. Keep the stored benchmark receipts for provenance, including the
clearly labeled historical five-benchmark receipt.
