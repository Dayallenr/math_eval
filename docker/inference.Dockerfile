FROM ghcr.io/astral-sh/uv:0.9.5 AS uv
FROM python:3.12-slim
COPY --from=uv /uv /usr/local/bin/uv
WORKDIR /opt/math-eval
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --extra inference --no-dev
ENV PATH="/opt/math-eval/.venv/bin:$PATH" \
    HF_HOME=/tmp/huggingface \
    PYTHONUNBUFFERED=1
ENTRYPOINT ["math-eval"]
