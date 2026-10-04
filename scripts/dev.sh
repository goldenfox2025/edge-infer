#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${IMAGE:-llm-infer-dev}"
exec docker run --rm -it --gpus all \
  -v "$REPO_ROOT:/workspace" -w /workspace "$IMAGE" bash
