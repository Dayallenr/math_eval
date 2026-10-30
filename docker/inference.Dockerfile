FROM ghcr.io/astral-sh/uv:0.9.5 AS uv
FROM registry.k8s.io/kubectl:v1.34.1 AS kubectl
FROM python:3.12-slim
COPY --from=uv /uv /usr/local/bin/uv
COPY --from=kubectl /bin/kubectl /usr/local/bin/kubectl
WORKDIR /opt/math-eval
COPY pyproject.toml uv.lock README.md ./
# Install the locked dependencies in their own layer. Plain RUN works with
# Docker/BuildKit and NRP's Kaniko builder; no BuildKit cache mount is required.
RUN UV_HTTP_TIMEOUT=300 uv sync --frozen --extra inference --extra grading \
    --no-dev --no-install-project --no-cache
COPY src ./src
RUN UV_HTTP_TIMEOUT=300 uv sync --frozen --extra inference --extra grading \
    --no-dev --no-editable --no-cache
ENV PATH="/opt/math-eval/.venv/bin:$PATH" \
    HF_HOME=/tmp/huggingface \
    PYTHONUNBUFFERED=1
ENTRYPOINT ["math-eval"]
