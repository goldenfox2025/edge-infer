#!/bin/bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$REPO_ROOT/build}"
if [[ "$BUILD_DIR" != /* ]]; then BUILD_DIR="$REPO_ROOT/$BUILD_DIR"; fi
export BUILD_DIR
export PYTHONPATH="$BUILD_DIR:$REPO_ROOT/bindings/python${PYTHONPATH:+:$PYTHONPATH}"

cd -- "$REPO_ROOT"
exec "$PYTHON_BIN" "$REPO_ROOT/frontend/chat.py" "$@"
