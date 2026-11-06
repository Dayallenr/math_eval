# Math Evaluation

Evaluate exported language-model checkpoints on seven math benchmarks using
Harbor and the NRP Nautilus cluster. A GPU worker loads the model once and saves
its answers. Harbor then grades those answers in isolated CPU pods and produces
one score per benchmark.

The standard workflow runs entirely on NRP, including the coordinator in remote
tmux. After starting a run and detaching, your laptop may sleep or shut down.

```text
Mac: connect, log in, start a run
  → NRP CPU coordinator in tmux
  → NRP GPU worker generates and saves answers
  → GPU is released
  → Harbor runs up to two CPU grading pods concurrently
  → reports, CSV, responses, and diagnostics saved on shared storage
```

## Benchmarks

| Benchmark | Selection | Questions |
| --- | --- | ---: |
| MATH-500 | Test | 500 |
| BeyondAIME | Test | 100 |
| OlymMATH | English HARD | 100 |
| HMMT February 2025 | MathArena release | 30 |
| HMMT February 2026 | MathArena release | 33 |
| AIME 2025 | AIME I + II | 30 |
| AIME 2026 | AIME I + II | 30 |
| **Total** | | **823** |

An optional pilot uses two questions per benchmark, fourteen total. Dataset
revisions, task hashes, and model settings are recorded for reproducibility.
Accuracy includes every question, including responses that reach the token
limit. This is the repository's evaluation protocol; matching benchmark names
alone does not reproduce another paper's scores.

## Setup

Use Python 3.12+, Git, uv, kubectl, and int128's kubelogin. On macOS:

```bash
brew install git uv kubectl kubelogin
git clone https://github.com/Dayallenr/math_eval.git
cd math_eval
uv python install 3.12
uv sync --python 3.12 --locked --extra grading
```

For an existing checkout, run the `uv` commands from its root. Docker and local
Torch/Transformers are not required for the standard NRP workflow.

Configure your account using [NRP's access instructions](https://nrp.ai/documentation/userdocs/start/getting-started/),
then verify access to the lab namespace:

```bash
kubectl --context nautilus -n wang-research-lab get pvc
```

Copy the profile only if you do not already have `nrp.toml`:

```bash
cp nrp.example.toml nrp.toml
```

The template pins the published GPU and CPU images, benchmark archive, hashes,
namespace, and image-pull Secret. The lab's datasets and baseline model are
already on NRP; a new model evaluation does not require another dataset upload
or image build. For another namespace, configure its PVCs and image access first.
See [storage and checkpoint requirements](docs/nrp-shared-suites.md) and
[private-image credentials](docs/maintenance/manual-evaluation.md#private-image-only-configure-pull-credentials).

Create the CPU coordinator once, then inspect startup:

```bash
.venv/bin/python scripts/remote_coordinator.py create
.venv/bin/python scripts/remote_coordinator.py status
.venv/bin/python scripts/remote_coordinator.py logs
```

If it already exists, use `status` instead of creating another one. Once it is
ready, log in and verify remote permissions:

```bash
.venv/bin/python scripts/remote_coordinator.py login
.venv/bin/python scripts/remote_coordinator.py check
```

Open the displayed login URL on the same laptop while the command is running.
The browser callback uses a temporary localhost port-forward. For UCSC, choose
CILogon/UCSC if prompted, then complete CruzID/Duo login. The remote host uses your
existing namespace permissions; this workflow installs no Roles, RoleBindings,
or ServiceAccounts. Credentials remain outside the shared PVC and Git.

## Run an evaluation

Attach to remote tmux:

```bash
.venv/bin/python scripts/remote_coordinator.py shell
```

Inside that remote session, start the prepared SFT-without-labels baseline:

```bash
# Optional 14-question setup check:
evalctl run --model math-eval-models:Qwen3-8B-SFT-no-labels --pilot

# Full 823-question evaluation, as a separate run:
evalctl run --model math-eval-models:Qwen3-8B-SFT-no-labels
```

Run one command at a time. Each submission gets its own evaluation ID. For a new
checkpoint, substitute `--model YOUR-PVC:exports/MODEL`; the path must contain a
complete Hugging Face model export. Adapter-only checkpoints must be merged or
exported first. This repository evaluates models; it does not train them.

Detach with **Ctrl-b, then d**. Your Mac can then sleep or shut down. Reconnect
with the same `shell` command. Do not use Ctrl-C merely to disconnect.

The comparison profile uses thinking off, temperature 0.7, top-p 0.8, top-k 20,
min-p 0, seed 42, batch size 1, an 8,192-token input limit, and a 12,288-token
output limit. Keep the profile fixed when comparing checkpoints. GPU queue time
is separate from inference runtime.

## Results and recovery

Inside the remote session:

```bash
evalctl status EVALUATION-ID
evalctl results EVALUATION-ID
```

Each run produces:

| Artifact | Contents |
| --- | --- |
| `benchmarks.csv` | Per-benchmark accuracy, correct/total counts, errors, and truncation counts |
| `report.json` | Final status, scores, model identity, protocol, and suite fingerprint |
| `run/predictions.jsonl` | Prompts, full responses, extracted answers, token counts, and stopping reasons |
| `run/summary.json` | Per-question grading results and benchmark summaries |
| `run/run.json` | Model information and inference settings |
| `run/grading/` | Harbor trial logs and verifier diagnostics |

The persistent workspace is
`/state/math-eval/remote/math-eval-remote-dayallen/runs/EVALUATION-ID/`.
Final reports also live on `shared-pvc` under
`math-eval/evaluations/EVALUATION-ID/`. Reports are available after grading and
upload complete; failed verifier runs are reported as infrastructure errors,
not incorrect model answers.

If the coordinator was interrupted, reuse the saved ID:

```bash
evalctl resume EVALUATION-ID
```

Grading retries preserve saved answers. A broad `--retry-grading` repeats the
whole grading attempt; for an isolated infrastructure failure, see the
[targeted recovery notes](docs/nrp-remote-tmux.md#interruptions-and-cleanup)
before launching unnecessary grading work.

Tmux survives laptop disconnection while its pod stays running. If the pod is
replaced, log in again and resume from the persistent workspace. NRP's interactive
Deployments are subject to its two-week cleanup policy. When finished with the
host, follow the [stop and cleanup instructions](docs/nrp-remote-tmux.md).

## Project guide

| Location | Purpose |
| --- | --- |
| `src/math_eval/` | Dataset adapters, generation, Harbor integration, orchestration, and reporting |
| `scripts/` | Remote-host setup and published-image verification |
| `tasks/aime-example/` | Small reference Harbor task used by the regression tests |
| `tests/` | Grading, provenance, recovery, and infrastructure tests |
| `docker/` | GPU inference and CPU task image definitions |
| `nrp.example.toml` | Shared configuration template; local `nrp.toml` is ignored |
| `docs/` | Operation, storage, architecture, benchmark sources, and validation |
| `docs/maintenance/` | Image publishing and optional manual debugging |

Start with the [remote operations guide](docs/nrp-remote-tmux.md). See
[architecture](docs/architecture.md), [benchmark provenance](docs/benchmark-sources.md),
[storage](docs/nrp-shared-suites.md), and [validation](docs/nrp-verification.md)
for details. [Maintenance instructions](docs/maintenance/README.md) cover tests
and image publication. Generated data, local environments, credentials, and
run artifacts are excluded from Git.
