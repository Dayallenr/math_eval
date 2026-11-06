# Publish evaluation images

This is a maintainer task. Existing evaluators reuse the digests in
`nrp.example.toml`; changing model weights does not require new images.

The registry project is
[Dayallen/model-eval](https://gitlab.nrp-nautilus.io/Dayallen/model-eval).
GitHub hosts the source repository, while GitLab provides the container registry
and an optional manual builder. These are services used by one codebase, not two
runtime implementations.

## Build with Docker

From the repository root, after committing and checking the working tree:

```bash
docker login gitlab-registry.nrp-nautilus.io -u Dayallen
export MATH_EVAL_IMAGE_TAG="$(git rev-parse HEAD)"
export MATH_EVAL_REGISTRY='gitlab-registry.nrp-nautilus.io/dayallen/model-eval'
docker buildx build --platform linux/amd64 \
  -f docker/inference.Dockerfile \
  -t "$MATH_EVAL_REGISTRY/math-eval:$MATH_EVAL_IMAGE_TAG" --push .
docker buildx build --platform linux/amd64 \
  -f docker/task.Dockerfile \
  -t "$MATH_EVAL_REGISTRY/math-task:$MATH_EVAL_IMAGE_TAG" --push .
```

Use the registry token at the hidden Docker login prompt. Never put a token in
a command, Git remote URL, or tracked file. Building the inference image requires
substantial disk space and network transfer; it is not needed to run evaluations.

Pin the resulting image digests, rather than a mutable `latest` tag. Verify the
runner with the build's source commit using the
[verification helper](README.md#published-image-verification). Verify the task
image through Harbor, then run an appropriate real-model pilot for an updated
inference image before changing the shared profile.

## Optional GitLab runner build

`.gitlab-ci.yml` supports an explicitly requested web pipeline and a manual
`build-runner` job. Normal pushes do not launch it. Publish the intended source
commit to GitLab, select that branch/commit under **Build → Pipelines → New
pipeline**, then start `build-runner`. It builds the GPU runner only; publish the
CPU task image separately when its Dockerfile changes.

The job uses GitLab-provided registry credentials and the maintained Kaniko
builder. Success produces `runner-image.txt` and `runner-digest.txt` artifacts.
Use the resulting digest only after direct NRP verification. Git authentication
and container-registry authentication are separate: Git pushes need repository
write access; image publication needs registry write access.

Published runtime images may contain compatibility commands from earlier
versions. Keep their request schema and current coordinator compatible, or
publish and verify a coordinated replacement. Do not delete registry images
referenced by existing runs or the pinned shared configuration.
