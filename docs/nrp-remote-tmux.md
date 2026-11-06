# Run the evaluation in remote tmux on NRP

This uses the existing `nrp-laptop` coordinator **inside an NRP CPU pod**. The CLI
name is retained for compatibility: no evaluation process needs to keep running on your laptop.
The CPU coordinator submits GPU generation, waits for it to finish, then runs
Harbor grading in isolated NRP CPU pods and uploads the reports. It adds no Roles,
RoleBindings, or ServiceAccounts. It authenticates using your own NRP account.

The setup is a CPU-only Deployment named `math-eval-remote-dayallen` in
`wang-research-lab`. NRP requires a Deployment for an interactive/idle CPU host;
such Deployments are subject to NRP's two-week cleanup policy. See
[NRP interactive pod policy](https://nrp.ai/documentation/userdocs/running/long-idle/).
Remove it when your evaluations are finished.

## 1. Prepare or inspect the host from your laptop

Run these commands from the repository root, with
`nrp.toml` configured and your normal working `kubectl` login. The helper uses
Python 3.12 or later; the existing `.venv/bin/python` meets this requirement.

```bash
.venv/bin/python scripts/remote_coordinator.py status
.venv/bin/python scripts/remote_coordinator.py logs
```

If the host does not exist, create it once:

```bash
.venv/bin/python scripts/remote_coordinator.py create
```

Creation copies a small source snapshot, locked dependencies, and public OIDC
configuration into a ConfigMap. It installs tmux, kubectl 1.34.1, kubelogin
1.36.4, uv 0.12.23, and the project's grading dependencies on ephemeral container
storage. The two downloaded Kubernetes tools are checked against their release
SHA256 files. No image build, Docker daemon, local GPU, or new administrator role
is needed. Watch `status` until `READY` is `1/1`; `logs` explains startup failures.
`create --dry-run` previews the resources without creating them.

The host deliberately does not use its default ServiceAccount token. Only public
OIDC settings are copied from your laptop. Login tokens are created by your
browser login and kept in the host's private ephemeral home directory, outside
shared PVC storage. The model weights are not mounted in this CPU host.

## 2. Log in on the remote host

```bash
.venv/bin/python scripts/remote_coordinator.py login
```

Keep the command running and open the displayed login URL in a browser **on this
same laptop**. Sign in using the same NRP account that can access
`wang-research-lab`. For UCSC, choose CILogon if prompted, select University of
California, Santa Cruz, then sign in with CruzID and complete campus MFA. This is
the standard NRP
[browser login flow](https://nrp.ai/documentation/userdocs/start/getting-started/).
Do not paste tokens or passwords into Git, chat, or shared storage. If the login
expires, rerun `login` to obtain a fresh URL.

The helper temporarily forwards `127.0.0.1:18000` on your laptop to the same
loopback port in the pod for the browser callback. It closes this connection
after login; no public service is exposed. Leave local port 18000 free while
logging in. The callback logs you into the remote pod, rather than only into the
NRP website. NRP's separate device-code page rejected this federated UCSC
account with “Interface can only be accessed by internal users,” so this setup
uses the standard browser flow instead.

Then verify remote access and the pilot configuration:

```bash
.venv/bin/python scripts/remote_coordinator.py check
```

This checks user permissions and prepares a pilot manifest without starting a
GPU job. A fresh login is needed if the coordinator container is replaced; its
credentials are intentionally ephemeral.

## 3. Attach and run the trained baseline pilot

```bash
.venv/bin/python scripts/remote_coordinator.py shell
```

You are now **inside remote tmux**. Run:

```bash
evalctl run --model math-eval-models:Qwen3-8B-SFT-no-labels --pilot
```

This target is the requested Qwen3-8B SFT model trained without dataset labels,
from `Dayallen/WANG_Lab_Baseline_Model`, already downloaded to `math-eval-models`.
It is not the separately stored unchanged official Qwen3-8B. See
[the model verification details](nrp-shared-suites.md#requested-sft-baseline-without-dataset-labels).

The pilot has fourteen questions, two from each of the seven benchmarks. Keep
its printed evaluation ID. Press **Ctrl-b, release both keys, then d** to detach.
After detaching, your laptop may sleep or shut down. The NRP host and GPU/CPU
workers continue independently. Reconnect later with the same `shell` command.
Do not press Ctrl-c merely to disconnect: that interrupts the coordinator.

## 4. Inspect and continue

Inside the remote shell:

```bash
evalctl status EVALUATION-ID
evalctl results EVALUATION-ID
```

If the coordinator command is still occupying the terminal, create a second
remote tmux window with **Ctrl-b then c** to run status commands. `results` is
available once report upload finishes. Each run contains seven separate benchmark
scores, saved model responses, and grading diagnostics.

The remote workspace is:

```text
/state/math-eval/remote/math-eval-remote-dayallen/runs/EVALUATION-ID/
```

It is persisted on `shared-pvc`. Files include `pipeline.json`, `run/predictions.jsonl`,
`report.json`, and `benchmarks.csv`. The final uploaded copies also reside under
`shared-pvc:math-eval/evaluations/EVALUATION-ID/`. `evalctl` sets the config and
persistent workspace arguments automatically.

After the pilot has completed without infrastructure errors, inspect answers
and scores. Start the full 823-question suite with:

```bash
evalctl run --model math-eval-models:Qwen3-8B-SFT-no-labels
```

Keep thinking off, temperature 0.7, top-p 0.8, top-k 20, min-p 0, seed 42, and
12,288 maximum new tokens for comparisons. Each new run gets its own evaluation
ID. Running a full suite creates a new generation job; it does not extend the
pilot. For other checkpoints, change only `--model PVC:folder` as appropriate.

## Interruptions and cleanup

A network disconnect from tmux does not interrupt evaluation. If the coordinator
process was interrupted, resume the same evaluation ID rather than submitting a
new run:

```bash
evalctl resume EVALUATION-ID
# Broad retry: preserves the prior attempt and regrades all saved answers.
evalctl resume EVALUATION-ID --retry-grading
```

Tmux cannot survive deletion or replacement of its own pod. Recreate the host if
needed, log in again, attach, and resume using the same workspace. GPU job and
saved-answer recovery prevent a second GPU run for the same saved request.
For an isolated infrastructure error in an otherwise completed grading run,
Harbor supports a targeted retry:

```bash
python -m harbor.cli.main jobs resume \
  -p /state/math-eval/remote/math-eval-remote-dayallen/runs/EVALUATION-ID/run/grading/BENCHMARK \
  -f RuntimeError
```

This removes matching failed trials, so first preserve their directories and
stop other coordinators for that evaluation. It retains completed Harbor trials.
It does **not** rebuild `summary.json`, update coordinator state, or upload the
final reports by itself. A maintainer must validate saved predictions, aggregate
all Harbor results, and finish the coordinator's reporting/upload phase; do not
mark an evaluation complete by editing JSON. The completed baseline's recovery
receipt records that procedure. Use the broad retry when a full preserved regrade
is acceptable.

An interrupted generation job
that actually failed needs investigation; resume does not silently replace it.

To stop an evaluation intentionally, interrupt its coordinator with Ctrl-c, then
run `evalctl cancel EVALUATION-ID`. This removes that evaluation's active workers
and retains saved data. To remove the remote host when finished, run **on your
laptop**:

```bash
.venv/bin/python scripts/remote_coordinator.py delete
```

Delete the host only after evaluations finish or after intentionally stopping
them. Removing the host alone does not cancel a running GPU job. PVCs, model
files, and saved results are retained. `create` refuses to replace an existing
host; after changing source or configuration, finish or stop active work before
recreating it. Do not delete the shared PVCs as part of coordinator cleanup.

## Terminal and progress tips

Use **Ctrl-b then w** to select a tmux window, **Ctrl-b then c** to create one,
and **Ctrl-b then [** to enter scrollback (press **q** to leave). To enable mouse
scrolling, run `tmux set -g mouse on` in a separate idle window. Do not type
shell commands into the window occupied by an active coordinator.

The coordinator prints phase changes rather than every model answer. To inspect
generation progress from the laptop:

```bash
kubectl --context nautilus -n wang-research-lab logs job/EVALUATION-ID-generate --tail=10
kubectl --context nautilus -n wang-research-lab get pods -l math-eval/evaluation=EVALUATION-ID
```

Pod AGE includes scheduling wait. Container start time describes actual runtime.
A connection/attachment error alone does not mean the evaluation failed: inspect
`status` and worker logs before submitting a second run. If the session was
interrupted, resume its existing evaluation ID.
