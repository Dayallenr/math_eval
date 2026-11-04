# Verify the existing runner directly on NRP

This replaces GitLab's `verify-runner` job. Use an already-published image; no
rebuild, local Docker installation, dataset download, or model is required.
The check runs in `wang-research-lab`, where lab members can inspect pod events.

## Verified runner

The existing runner passed on NRP on November 3, 2026 (UTC): **31 tests passed,
zero skips**, including real tiny Qwen3 save/load/generation on CPU. Source files
matched commit `b31ae3442a4bbce816ace113cb7da08ef4605a26`. Job
`math-eval-verify-20261103070714-6355` completed with exit code 0. The command below
uses the immutable image digest observed in that successful pod.

This checkout saves the full log and receipt under ignored
`.cache/nrp/runner-verification.log` and `.cache/nrp/runner-verification.json`.
Reusing this exact image does not require rerunning verification for each model.
A new runner image should be verified again.

## Submit the check

Run from the repository root after completing the README's NRP login setup.
The helper uses only Python's standard library, Git, and `kubectl`.

```bash
python3 scripts/verify_nrp.py submit \
  --image gitlab-registry.nrp-nautilus.io/dayallen/model-eval/math-eval@sha256:386537c9ee80c892ce19984508b48f443c57481ef650b91c5428a1d5e9ca7b42 \
  --commit b31ae3442a4bbce816ace113cb7da08ef4605a26 \
  --pull-secret nrp-registry-dayallen
```

`nrp-registry-dayallen` must authorize image pulls from `dayallen/model-eval`.
The Secret name alone does not establish access. If necessary, follow the
[hidden credential prompt](manual-evaluation.md#private-image-only-configure-pull-credentials)
and pass the Secret name you created. Never put a token in the command or Git.
Pass `--pull-secret ''` only for an image that is publicly pullable.

For a future image, replace **both** `--image` and `--commit` with the matching
published image and source commit available in this checkout. A digest is
preferred over a mutable tag. Uncommitted edits are not uploaded: the helper
packages tests from the selected Git commit and checks that the image's installed
Python source matches that commit. This prevents testing different code by mistake.

Add `--dry-run` to print the ConfigMap and Job manifests without cluster access.
Optional `--context` and `--namespace` go before `submit` or `status`.

## Inspect the result

Submission returns a unique name beginning `math-eval-verify-` and exits. Use
that actual name below:

```bash
python3 scripts/verify_nrp.py status YOUR-VERIFICATION-JOB
```

This prints the job/pod state, image digest once pulled, recent test logs, and
pod events. An initial `Pending` phase can mean the image is still downloading;
events identify image-pull, scheduling, and container errors. A registry
`unauthorized` error requires correcting pull credentials, then submitting a
new check. Do not repeatedly restart a pod that is still downloading normally.

To follow the full test log after the container starts:

```bash
kubectl --context nautilus -n wang-research-lab logs \
  -f job/YOUR-VERIFICATION-JOB
```

Success requires the Job to complete successfully and its log to contain
`VERIFICATION PASSED: ... tests; no skips`. An image that starts successfully
but fails a test is not verified. Preserve the complete log and actual image
digest for any failure report:

```bash
mkdir -p .cache/nrp
kubectl --context nautilus -n wang-research-lab logs \
  job/YOUR-VERIFICATION-JOB > .cache/nrp/runner-verification.log
```

After preserving the result, remove only this verifier's Job and owned ConfigMap:

```bash
python3 scripts/verify_nrp.py cleanup YOUR-VERIFICATION-JOB
```

The job also expires automatically seven days after finishing. It has a one-hour
deadline including startup; its pending time is not limited by GitLab's shared
runner `poll_timeout`.

## What is checked

- The installed Python source matches the requested image-build commit.
- The CLI and bundled `kubectl` start, CUDA-enabled Torch imports, and Harbor is
  the pinned version.
- All committed regression tests pass without skips, including a tiny randomly
  initialized Qwen3 model that loads and generates a padded batch on CPU.

The image does not contain the tests, so the helper mounts a small immutable
ConfigMap containing the tests, teaching grader fixture, and verification script.
The tests exercise the image's installed code. Nothing is installed at runtime.
Model/dataset Hub access is disabled during the check; the tiny model is created
locally inside the container and has no meaningful benchmark score.

This is one CPU Job requesting 2 CPUs, 4 GiB RAM, and 25 GiB temporary storage.
It selects AMD64 CPU nodes and requests no GPU. It mounts no PVC, model, dataset,
or Kubernetes API token. The image-pull Secret is used by Kubernetes to download
the image, not mounted into the test process. This follows
[NRP's CPU workload guidance](https://nrp.ai/documentation/userdocs/running/cpu-only/).

A pass validates the runner on CPU. A real fourteen-question GPU pilot remains
a deployment step; dedicated controller permissions apply only to the held
NRP-controller workflow. This check does not establish GPU compatibility or benchmark accuracy. The separate published
math-task image check is recorded below.
Continue with the [README setup](../README.md#3-configure-the-existing-images-and-model-location).

The GitLab builder remains an optional future image-building method. After the
updated `.gitlab-ci.yml` is committed and pushed, normal pushes do not create a
pipeline; explicit UI builds are still available. Old running jobs are not
automatically cancelled by this local change.

## Published grading image check

The math-task image also passed seven live Harbor checks on NRP on November 3,
2026 (UTC), using the digest pinned in `nrp.example.toml`:

```text
gitlab-registry.nrp-nautilus.io/dayallen/model-eval/math-task@sha256:0893dbcc6500a1f19957e43e5a9cf1528c9b62470ffd091702d504312c1f9c1d
```

The diagnostic used scripted saved responses: one correct, one wrong, and one
missing answer each for symbolic math and AIME-style integers, plus one correct
large BeyondAIME integer. All seven rewards matched expectations (three ones,
four zeros), with no infrastructure errors or retries. The symbolic check
accepted `0.5` for `1/2`. The integer checks accepted leading zeros.

Harbor coordinated on the laptop CPU for this diagnostic and launched at most two
isolated task pods at a time on NRP. No model or GPU was used. All test pods were
removed. The local log, receipt, and compressed task diagnostics are saved under
ignored `.cache/nrp/task-image-verification.*`. These results validate the
published image and real task-pod grading path; they are not model accuracy
measurements or a test of the entire cluster controller.

The ignored `nrp.toml` and the team template now use both published image digests
and the existing `nrp-registry-dayallen` pull Secret. The shared benchmark PVC is
still Bound with ReadWriteMany access.

## Controller workflow on hold

The administrator step below applies only to the fully NRP-managed workflow on
`nrp-controller-on-hold`. The active `laptop-orchestration` workflow uses the
user's local kubeconfig and does not require these dedicated accounts or roles.

Resuming the controller workflow requires a namespace administrator installing
the two dedicated service accounts and scoped roles. The checked-in manifest matches the generated
profile. From a checkout of this repository, the administrator runs:

```bash
kubectl --context nautilus apply -f docs/nrp-rbac.json
```

The requesting user's account can create task pods but cannot create Roles in
`wang-research-lab`; the dedicated `math-eval-controller` and `math-eval-harbor`
service accounts are not installed yet. A new model export location and a real
fourteen-question GPU pilot are needed after the permissions are installed.

## Laptop coordinator regression checks

The laptop branch adds eleven regression tests for transfer validation, separate
benchmark reports, GPU-before-Harbor ordering, request identity, interrupted
grading, cancellation, and recovery after a lost create/upload response. The
local suite passes 41 tests; the tiny real Qwen test is skipped locally because
Torch/Transformers are not installed. The published runner separately passed
that test as recorded above. No new image build is needed for this branch: new
coordinator code runs locally, and temporary transfer pods use a standard-library
helper supplied in their command.

A real model export and GPU pilot are still required before claiming the complete
model evaluation is validated. Scripted CPU fixtures validate orchestration only.

## Live laptop workflow check

On November 3, 2026 (Pacific time; November 4 UTC), evaluation
`math-eval-laptop-check-20261104042901-a8ba` passed the laptop workflow on NRP:
14 scripted responses, two per benchmark, seven separate reports, and zero
infrastructure errors. The check exercised archive staging and fingerprint
validation, saved-response download, Harbor task-pod execution, report upload to
`shared-pvc`, and an interrupted grading retry that reused the saved responses.
No GPU was requested, no model weights were loaded, and no Roles, RoleBindings,
or dedicated ServiceAccounts were installed. Scripted fixture files replaced
inference; their metadata is explicitly marked `generation_kind = test_fixture`.
Their zero accuracy scores are diagnostic outputs, not a model baseline.

The check initially exposed duplicate volume definitions for a shared PVC; the
laptop manifests now reuse one claim volume while preserving read-only dataset
and model mounts. To avoid repeated image downloads during the diagnostic,
retried Harbor task pods used the existing NRP provider with only a test-specific
node selector for a cached CPU node. Normal evaluation scheduling is unchanged.
The earlier CUDA-image fixture attempt was stopped while downloading its image.
All diagnostic Jobs/pods were deleted; fixture reports remain in their own
uniquely named evaluation directory.

The complete log, receipt, local regression log, and compressed Harbor diagnostics
are saved under ignored `.cache/nrp/laptop-verification.*` and
`.cache/nrp/laptop-tests.log`. This verifies the transfer/grading/upload path with
scripted answers. Loading a real exported model on an eligible GPU and running
its fourteen-question pilot remains necessary.
