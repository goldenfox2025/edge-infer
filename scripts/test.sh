#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$REPO_ROOT/build-cpu-tests}"
if [[ "$BUILD_DIR" != /* ]]; then BUILD_DIR="$REPO_ROOT/$BUILD_DIR"; fi

# Only the standalone workspace tests: no CUDA, Python packages or model files.
BUILD_DIR="$BUILD_DIR" bash "$REPO_ROOT/scripts/build.sh" --cpu-tests --debug
ctest --test-dir "$BUILD_DIR" --output-on-failure "$@"
