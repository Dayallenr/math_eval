#!/bin/bash
set -eu

# Resolve this script's directory, rather than assuming a working directory.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec python "${SCRIPT_DIR}/grade.py" "$@"
