> On branch `laptop-orchestration`, use `math-eval nrp-laptop` and the
> [README](../README.md). The cluster-controller design below is preserved for
> `nrp-controller-on-hold`; its roles are not needed by the laptop workflow.

# How the NRP evaluation pipeline works

The input is a finished Hugging Face model export on an NRP PVC. The output is
seven benchmark scores, individual question results, and diagnostics on persistent
NRP storage. The model's training method does not change this workflow: base,
SFT-only, SFT plus GRPO, and other compatible exports use the same interface.
Follow the [README](../README.md) for installation and commands.

## What each component does

| Component | Purpose |
| --- | --- |
| Dataset adapter (`suite.py`) | Converts pinned dataset rows into consistent questions and Harbor tasks. Already run for the shared suites. |
| Shared suite archive | Holds all prepared questions, references, verifiers, task definitions, manifests, and hashes. Read-only during evaluation. |
| `nrp.toml` | Reusable team settings: image URLs, namespace, claims, GPU type, decoding, deadlines, and suite fingerprints. |
| CLI (`nrp.py`) | Checks claims/accounts and submits one controller Job. Also provides status, results, and cancellation. |
| NRP controller (`nrp_workflow.py`) | Runs independently of your terminal; creates generation and grading Jobs in order and saves progress. |
| GPU worker (`inference.py`) | Loads the exported model once, prompts it with benchmark questions, and saves its responses. |
| CPU Harbor worker (`results.py`) | Validates those saved responses, launches Harbor, collects rewards, and computes separate benchmark scores. |
| NRP environment provider (`harbor_nrp.py`) | Implements Harbor's container lifecycle, command execution, and file transfer using Kubernetes task pods. |
| SavedPredictionAgent | Writes the extracted final answer from a saved model response to `/app/answer.txt`. Does not invent an answer or use an oracle. |
| Task verifier (`test.sh` and `grade.py`) | Reads the submitted answer, compares it to the reference, and writes reward/diagnostics. |
| Oracle (`solution/solve.sh`) | Writes a known-correct answer for testing the harness. It is not used to evaluate the trained model. |

```text
new model export + reusable nrp.toml
                |
    controller Job on NRP (CPU)
                |
    generation Job on NRP (GPU)
       question -> model -> response
                |
    saved responses on output PVC
                |
    grading Job on NRP (CPU, Harbor)
                |
    isolated task pods (CPU)
       replay response -> test.sh -> grade.py -> reward
                |
    per-benchmark scores + logs on output PVC
```

## The model answers first; Harbor grades afterward

The GPU worker reads question prompts and never passes reference answers, test
files, or oracle scripts to the model. It creates one response per question with
fixed decoding settings. Questions are independent; evaluating does not train or
update the weights. A truncated output is recorded, not silently retried.

The default lab profile is Qwen3-8B with thinking off, temperature 0.7, top-p 0.8,
top-k 20, min-p 0, and at most 12,288 generated tokens per question. For thinking
mode use thinking on, temperature 0.6 and top-p 0.95, keeping the other settings.
Both profiles follow [Qwen sampling guidance](https://huggingface.co/Qwen/Qwen3-8B#best-practices);
the shorter output budget is the lab convention, not Qwen's general recommendation.
Use one fixed profile for the base, SFT and GRPO checkpoints being compared.
The output budget includes reasoning and final answer and excludes input tokens.
Transformers is the implemented inference backend; vLLM serving remains separate work.

Harbor later uses the replay agent to write the extracted final answer from the
saved response into each task pod's answer file. Harbor uploads/runs the verifier,
which has the reference answer. This separation allows expensive GPU generation
to end before CPU grading begins. It also makes the exact response available for
inspection. Seeing a reference in a task's tests does not mean the model saw it.

`solve.sh` is only an oracle harness check. During real evaluation the command
selects `math_eval.replay:SavedPredictionAgent`, not Harbor's oracle agent.
The new NRP workflow always uses Harbor; it does not silently switch to the
optional direct `local` grading backend if Harbor fails.

## How Harbor works without a Docker daemon on NRP

Harbor 0.23.0 supports custom environment providers. `NRPEnvironment` implements
its environment interface using `kubectl create`, `exec`, `cp`, and `delete`.
Harbor still manages task execution, agent replay, verifier execution, trial
results, and rewards. Kubernetes starts the task containers from prebuilt OCI
images instead of requiring Docker-in-Docker or access to a host Docker socket.
Docker is used by a maintainer to build the images before evaluation.

There are two published images:

- The runner image contains this repo, the locked Python/Torch dependencies,
  Harbor, graders, and kubectl. It is used for controller and worker Jobs.
- The smaller math-task image contains Python and the pinned math scorer. Harbor
  uses a separate task pod for each trial. It contains no datasets or weights.

Existing task Dockerfiles remain byte-for-byte unchanged. The provider maps each
Dockerfile SHA-256 to a published image, with the current math Dockerfile mapped
to `task_image`. Unknown Dockerfiles fail explicitly. Additional mappings can be
added to `[task_images]`, and tasks with an explicit `docker_image` are supported.
This prevents silently grading a future coding task in the wrong math image.
Maintainers must build the mapped image from the actual task environment and pin
its image digest for reproducible experiments.

The current provider supports a single Linux CPU container per task, Bash,
`tar` for kubectl copy, `timeout` for command limits, file transfer, and named
container users where `su` is available. Multi-service/Compose, GPU task pods,
Windows, and restricted-network tasks require more environment support. Adding
terminal/coding evaluations also requires a compatible interactive Harbor agent
and model connection: replaying a final math response does not solve terminal
tasks. Harbor remains the common task harness as those adapters are added.

## Storage, permissions, and lifecycle

The controller mounts only output storage. The GPU worker mounts model and suite
claims read-only plus output storage read-write. The CPU Harbor worker mounts the
suite read-only plus outputs; no model mount is needed. Output storage must be
ReadWriteMany because the controller and workers mount it at the same time.

Inference defaults to one generic NVIDIA GPU on an amd64 node with
`nvidia.com/gpu.memory > 40960` MiB. This admits compatible 48 GB-class and larger
cards in that resource pool, including L40/L40S. A6000 uses the separate
`nvidia.com/rtxa6000` resource; A100 uses `nvidia.com/a100` with namespace quota.
Configure `gpu_resource` to select that pool. NRP's distinct resource classes
cannot be expressed as alternatives in one ordinary pod resource request.
The memory constraint remains active for each pool. This selects GPU capacity;
the pilot still verifies that the actual model, dtype, drivers, and token budget
fit on the allocated device.

Each worker copies a single archive from CephFS to temporary container disk,
checks its archive SHA-256, extracts the selected pilot/full suite, and validates
all task hashes using the existing loader. Python environments and grading work
are on container disk. Harbor's many small trial files are compressed before
copying to CephFS. The uploaded source suite is never rewritten.

The controller ServiceAccount can create/get namespace Jobs. The separate Harbor
ServiceAccount can create/get/delete pods and use pods/exec. The worker builds a
temporary kubeconfig referring to the projected CA and rotating token file; it
does not embed raw credentials in images, logs, or persistent output. Neither the
GPU worker nor task pods mount a Kubernetes service-account token. Task pods
receive no PVCs, model weights, or host mounts. These are workload boundaries,
not a claim that namespace-wide workload permissions are unprivileged; only
trusted controller/worker code should receive those accounts.

Child Jobs have the controller Job as owner. Task pods have the grading Job as
owner. Harbor deletes its task pods after use. A task pod also has a finite
`activeDeadlineSeconds` and sleep lifetime. Canceling the controller cascades to
its Jobs and task pods; it preserves shared datasets, model weights, and output
files. Jobs have a seven-day TTL after completion. Download a report if desired
or read the persistent report after Kubernetes logs expire.

The controller's HTTPS API uses the projected ServiceAccount token and CA and
retries transient API errors. A matching existing Job is reused after an
ambiguous creation response; it does not submit a second GPU Job. Different
requests with a colliding Job name are rejected. Execution retries are disabled
(`backoffLimit: 0`) so failures do not quietly spend another GPU run. A restarted
controller refuses to overwrite an existing pipeline state. Fix the cause and
submit a new evaluation ID. A failed GPU stage never launches grading.

## Results and interpreting scores

Each evaluation has its own directory, named with a UTC timestamp and random
suffix unless the user supplies an ID. It contains:

```text
request.json                 Resolved input/profile and model locator
pipeline.json                Stage names, state, and any controller failure
generation.json              Request-bound pointer to the generated run
report.json                  Compact successful evaluation report
benchmarks.csv               One row per benchmark
runs/RUN-ID/run.json          Model identity, generation settings and progress
runs/RUN-ID/predictions.jsonl Actual model responses and token/finish metadata
runs/RUN-ID/summary.json      Per-question scores and benchmark totals
runs/RUN-ID/grading-diagnostics.tgz  Harbor trials, verifier logs and rewards
```

Benchmarks remain separate. A single model load processes all questions;
Harbor then grades benchmark groups and reports one score for each group. The
scores are not combined into a new benchmark. The fourteen-question pilot has two
questions per benchmark; use the full 823-question suite for reported results.

AIME and BeyondAIME use integer comparison; only AIME has a 0–999 answer limit. Other math tasks use exact matching followed by
Math-Verify 0.9.0 with pinned LaTeX/SymPy dependencies. Wrong answers and missing
final answers score zero with diagnostic reasons. Infrastructure/verifier errors
fail the evaluation instead of being counted as ordinary wrong answers.
Diagnostics and a partial summary are retained when grading raises an error.
Inspect `summary.json` for truncation, parser failures, and individual outcomes.
This custom single-response protocol is not an exact reproduction of each
paper's published scores. Symbolic grading has limitations, and training overlap
must be considered when interpreting results.

Model identity is recorded automatically so later scores can be traced to the
export that generated them. No manually invented model name is required. Keep
exports immutable; metadata/file identity is not a cryptographic hash of every
weight byte. Compare runs with the same suite fingerprint, runner/task image
digests, generation settings, and scorer. Never mix oracle/fixture checks with
real model evaluations.

## Validation and remaining deployment work

Regression tests cover adapters, model generation plumbing, grading, orchestration,
PVC/role manifests, archive integrity, failed-generation handling, and the real
Harbor environment interface against a recorded API. The controller fixture test
uses synthetic generation and patches the Harbor invocation; it is a local
orchestration test rather than a cluster model score.

A live CPU-only check ran Harbor against seven scripted tasks using the published
math-task image pinned in `nrp.example.toml`. It accepted equivalent symbolic and
zero-padded integer answers and rejected deliberately wrong or missing answers,
returning all seven expected rewards with zero infrastructure errors and zero
retries. Harbor coordinated on the laptop's CPU for this diagnostic; the isolated
task containers ran on NRP and were removed afterward. These are harness checks,
not real model benchmark scores or a test of the cluster controller permissions.

The published runner separately passed all 31 regression tests in an NRP CPU Job,
including tiny random Qwen3 generation. Both image digests are pinned in the team
template. Before production use, have a namespace administrator install the
[namespace roles](nrp-rbac.json), then run the pilot using a real lab model export.
Local builds and fixtures do not establish CUDA/VRAM compatibility
or real-checkpoint behavior. See the current README deployment status.
