# Build the evaluation runner on NRP GitLab

The destination project is <https://gitlab.nrp-nautilus.io/Dayallen/model-eval>.
Its registry prefix is `gitlab-registry.nrp-nautilus.io/dayallen/model-eval`.
The runner image will be published below that prefix as `math-eval:COMMIT-SHA`.
This workflow builds only the evaluation runner; the separate math-task image is
a later step. It does not start an evaluation or allocate a GPU.

## Upload the code once

The project initially has only a README. The remote builder needs this repo's
current code, lockfile, Dockerfile, and `.gitlab-ci.yml`. Commit local changes
before uploading so the remote branch contains the current seven benchmarks.
Run these commands from this evaluation repository after that commit:

```bash
git remote add nrp-gitlab https://gitlab.nrp-nautilus.io/Dayallen/model-eval.git
git push nrp-gitlab HEAD:refs/heads/evaluation-setup
```

If `nrp-gitlab` already exists, inspect its URL with `git remote get-url
nrp-gitlab` and reuse it only if it points to the intended project. The push
creates a separate GitLab branch. It preserves GitLab's existing default branch
and your GitHub `origin`. Do not force-push over the initial README history.
No model weights, generated suites, local configuration, or credentials need to
be uploaded. If HTTPS Git asks for authentication, use GitLab's supported Git
credentials; never place a token in the remote URL or send it in chat.

## Start one runner build

1. Open **Build → Pipelines** in the GitLab project.
2. Select the pipeline for branch `evaluation-setup` and the commit you uploaded.
3. Select **Run/Play** next to the manual `build-runner` job.
4. Watch the build log. A successful job prints the published image reference
   and saves `runner-image.txt` and `runner-digest.txt` as job artifacts.
5. Let `verify-runner` complete. It pulls that exact commit's image, checks the
   CLI and CUDA-enabled Torch installation, and runs the regression tests on CPU.

Uploading code alone does not launch the build because the build job is manual.
A blocked/manual pipeline at this point is expected. If no pipeline appears,
use **Build → New pipeline** and select `evaluation-setup`. The default branch
does not need to be changed for a branch pipeline to run.

The jobs use GitLab-provided `CI_REGISTRY_USER` and `CI_REGISTRY_PASSWORD` at
runtime; no registry token is committed. NRP documents preconfigured shared
runners and the maintained Kaniko fork in its
[GitLab build guide](https://nrp.ai/documentation/userdocs/development/gitlab/).
The job uses the documented `GODEBUG=http2client=0` setting for GitLab pushes.
It requests 30 GiB temporary storage, a 40 GiB limit, and up to 8 GiB RAM through
GitLab Runner override variables. Actual overrides depend on the shared runner's
configuration. If the job is pending, check runner availability; if it runs out
of memory or storage, inspect the runner limits rather than building on an
almost-full laptop. This build needs no permissions in `wang-research-lab`.

## Use the published reference after validation

Only after **both** jobs pass, download `runner-image.txt` from `build-runner`.
It contains a reference such as:

```text
gitlab-registry.nrp-nautilus.io/dayallen/model-eval/math-eval@sha256:ACTUAL-DIGEST
```

Use that actual reference for `image` in your ignored `nrp.toml`. Do not copy the
placeholder digest above. Each build uses the full source commit as its tag;
the digest identifies the image bytes, including when a commit's build is retried.
The pipeline also stores build-cache layers in the same project's `build-cache`
repository. No `latest` tag is overwritten.

Passing this pipeline validates the runner image on a CPU runner. A real NRP GPU
pilot, the math-task image, pull access, and namespace ServiceAccounts/roles are
still needed before a complete evaluation can be considered deployed.
