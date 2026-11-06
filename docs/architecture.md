# Architecture

The supported operating workflow is a remote tmux coordinator on NRP. The Mac
submits setup commands and provides the browser login, then can disconnect.
The command-line implementation retains the name `nrp-laptop` for compatibility;
`evalctl` runs that same coordinator inside the remote CPU pod.

## Components and responsibilities

| Component | Role | Compute |
| --- | --- | --- |
| Remote coordinator | Persists request/state, submits generation, runs Harbor, uploads reports | CPU |
| GPU worker | Loads one completed checkpoint and generates all responses | One GPU |
| Transfer pods | Copy the pinned suite, saved predictions, and reports | Short-lived CPU pods |
| Harbor task pods | Run the saved-answer agent and deterministic verifier | CPU, up to two concurrently |
| Persistent volumes | Retain checkpoints, benchmarks, predictions, and results | Shared NRP storage |

`remote_coordinator.py` creates the CPU Deployment and a public bootstrap
ConfigMap. It copies an explicit source snapshot rather than a `.git` directory,
local environment, model weights, or credentials. `remote-coordinator-start.sh`
installs the locked grading environment, tmux, kubectl, and kubelogin on ephemeral
container storage. The run workspace is on the output PVC.

The browser login uses an authorization-code flow with a temporary loopback
port-forward. NRP's device-code interface rejected the federated UCSC identity,
so it is not used. User tokens stay in the remote host's private ephemeral home;
worker/task pods do not receive them or a default ServiceAccount token. The
coordinator uses the user's existing namespace permissions.

## Evaluation lifecycle

1. Parse `PVC:relative/export/path`, validate the profile and Bound PVCs, assign
   an evaluation ID, and save an immutable request.
2. Copy the pinned benchmark archive through a CPU helper. Verify the archive,
   manifest, task inventory, hashes, and question counts.
3. Submit one GPU Job. The worker mounts model weights and benchmark storage
   read-only, loads the model once, and saves each response to JSONL.
4. Wait for successful generation completion. The finished GPU Job releases its
   GPU; no GPU is held while grading runs.
5. Fetch and validate the saved responses. Require exactly one matching response
   per question, with matching prompt, suite, extraction, and artifact hashes.
6. Run Harbor for each benchmark. Its replay agent uploads the saved answer to
   `/app/answer.txt`; it makes no LLM calls and does not choose the grade.
7. The task's `tests/test.sh` invokes the Python verifier. The verifier reads the
   reference answer and writes reward 1 or 0 plus diagnostic reasons.
8. Aggregate all question results and upload the report, CSV, run metadata, and
   compressed Harbor diagnostics. Mark the evaluation complete only after upload
   succeeds.

The model sees the question and output-format instructions. Gold answers and
oracle `solution/solve.sh` scripts are excluded from model prompts. The oracle is
only a task-packaging test; actual evaluation uses saved model responses.

## Harbor without a Docker daemon

`math_eval.harbor_nrp:NRPEnvironment` implements Harbor's environment interface
using Kubernetes pods and prebuilt OCI images. Kubernetes launches each task
container; the coordinator does not need Docker-in-Docker or a Docker socket.
The current provider supports a single Linux container per task. Future terminal
or coding benchmarks need suitable images, graders, agents, and environment
capabilities; arbitrary multi-container tasks are not supported automatically.

## Modules

| Module | Responsibility |
| --- | --- |
| `suite.py` | Pinned sources, normalization, Harbor task generation, suite verification |
| `inference.py` | Model identity, tokenizer/chat template, generation, saved response records |
| `grading.py` | Final-answer extraction, integer and symbolic comparison |
| `replay.py` | Harbor agent that transports a saved model answer |
| `harbor_nrp.py` | Isolated Kubernetes task environments for Harbor |
| `results.py` | Prediction validation, Harbor result parsing, benchmark scoring |
| `laptop.py` | User-authenticated coordinator used by remote `evalctl` |
| `_transfer.py` | Minimal helper-pod transfer and upload integrity checks |
| `nrp.py`, `nrp_workflow.py` | Shared request/configuration and cluster-worker primitives |

The shared modules also retain compatibility with earlier controller images and
saved requests. That older controller is not part of the standard setup and its
administrator-role instructions are not part of this guide. Published worker
images and request schemas must remain compatible when changing these modules.

## Recovery and integrity

The coordinator locks each workspace so two processes cannot mutate the same
run. Resume checks request identity, reuses a matching existing GPU Job, and
reuses completed generation. A lost create response must not create a duplicate
GPU Job. A lost upload acknowledgement can be retried without regenerating or
regrading answers. Different reports cannot overwrite an existing completed run.

Per-question infrastructure errors keep the report incomplete. They are not
silently counted as wrong answers or graded with a different backend. Completed
responses and valid grades can be retained during recovery. The explicit broad
grading retry preserves the previous attempt and regrades saved answers; it does
not perform new inference.

Large numbers of small files on shared Ceph storage can make verification and
Harbor recovery slow. GPU queue time, generation, suite validation, and CPU
grading are separate phases; pod AGE includes waiting time.

See [the operations guide](nrp-remote-tmux.md) for commands and
[the benchmark record](benchmark-sources.md) for scoring details.
