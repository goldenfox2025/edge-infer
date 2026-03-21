#!/bin/bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"

exec "${PYTHON_BIN}" frontend/chat.py "$@"
