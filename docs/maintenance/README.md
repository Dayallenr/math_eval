# Maintenance

Normal evaluators should follow the [README](../../README.md) and
[remote operations guide](../nrp-remote-tmux.md). No image build or local dataset
preparation is needed for another completed model export.

## Regression checks

```bash
uv sync --python 3.12 --locked --extra grading
uv run --no-sync python -m unittest discover -s tests -v
```

The optional tiny Qwen smoke test requires the inference extra and is skipped
without it. Install that extra only on a suitable development machine or use
the published runner verification below. Unit tests use temporary fixtures and
do not submit GPU work or require access to the NRP namespace.

## Published-image verification

After an image change, commit the source and publish the image before verifying
it. Supply the exact **build source commit**, not an unrelated current HEAD:

```bash
.venv/bin/python scripts/verify_nrp.py submit \
  --image REGISTRY/PROJECT/math-eval@sha256:ACTUAL-DIGEST \
  --commit BUILD-SOURCE-COMMIT
.venv/bin/python scripts/verify_nrp.py status VERIFICATION-JOB-NAME
.venv/bin/python scripts/verify_nrp.py cleanup VERIFICATION-JOB-NAME
```

Verification runs in an NRP CPU Job. It checks installed-source hashes, versions,
CUDA-enabled Torch, CLI availability, and tests, including tiny model inference.
It does not run the full baseline or allocate a GPU. A new CUDA image also needs
an appropriate GPU pilot before its production use.

## Other references

- [Image publication](images.md): local Docker builds and the optional GitLab builder.
- [Manual debugging](manual-evaluation.md): prepare fixtures/suites, generate
  responses separately, inspect graders, and configure private-image pulls.
- [Architecture](../architecture.md): module responsibilities and integrity checks.
- [Validation record](../nrp-verification.md): completed model run and prior checks.

The old controller-based deployment is retained in Git history. The standard
workflow uses remote tmux and the existing user identity. Compatibility helpers
and saved request fields must remain compatible with published worker images;
removing them requires coordinated image and request-schema changes.
