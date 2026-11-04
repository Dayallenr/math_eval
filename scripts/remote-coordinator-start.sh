#!/bin/sh
set -eu
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y --no-install-recommends ca-certificates curl tmux unzip >/dev/null
rm -rf /var/lib/apt/lists/*
pip install --no-cache-dir uv==0.12.23 >/dev/null
curl -fsSL https://dl.k8s.io/release/v1.34.1/bin/linux/amd64/kubectl -o /tmp/kubectl
curl -fsSL https://dl.k8s.io/release/v1.34.1/bin/linux/amd64/kubectl.sha256 -o /tmp/kubectl.sha256
(cd /tmp && echo "$(cat kubectl.sha256)  kubectl" | sha256sum -c -)
install -m 755 /tmp/kubectl /usr/local/bin/kubectl
curl -fsSL https://github.com/int128/kubelogin/releases/download/v1.36.4/kubelogin_linux_amd64.zip -o /tmp/kubelogin.zip
curl -fsSL https://github.com/int128/kubelogin/releases/download/v1.36.4/kubelogin_linux_amd64.zip.sha256 -o /tmp/kubelogin.sha256
python - <<'PY'
import hashlib
from pathlib import Path
assert hashlib.sha256(Path('/tmp/kubelogin.zip').read_bytes()).hexdigest() == Path('/tmp/kubelogin.sha256').read_text().split()[0]
PY
unzip -qo /tmp/kubelogin.zip kubelogin -d /tmp
install -m 755 /tmp/kubelogin /usr/local/bin/kubectl-oidc_login
mkdir -p /workspace /root/.kube
chmod 700 /root/.kube
install -m 600 /bootstrap/kubeconfig.json /root/.kube/config
python - <<'PY'
import tarfile
with tarfile.open('/bootstrap/source.tgz') as archive:
    archive.extractall('/workspace', filter='data')
PY
install -m 644 /bootstrap/nrp.toml /workspace/nrp.toml
cd /workspace
uv sync --python 3.12 --locked --extra grading
mkdir -p "$MATH_EVAL_WORKSPACE"
cat > /usr/local/bin/evalctl <<'EOF'
#!/bin/sh
cd /workspace
exec /opt/math-eval-venv/bin/math-eval nrp-laptop --config /workspace/nrp.toml --workspace "$MATH_EVAL_WORKSPACE" "$@"
EOF
chmod 755 /usr/local/bin/evalctl
cat > /root/.bashrc <<'EOF'
export PS1='nrp-eval:\w\$ '
echo 'Use evalctl run/resume/status/results/cancel. Detach tmux: Ctrl-b then d.'
EOF
chmod 700 /root/.kube/cache 2>/dev/null || true
tmux new-session -d -s eval -c /workspace /bin/bash
touch /tmp/coordinator-ready
printf 'REMOTE COORDINATOR READY; authenticate with the remote helper login command. No GPU requested.\n'
exec sleep infinity
