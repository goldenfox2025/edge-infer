#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE_DIR="${PROFILE_DIR:-$REPO_ROOT/data}"

if (($# == 0)); then
  echo "Usage: bash scripts/profile.sh {ncu|nsys} [chat options]" >&2
  exit 2
fi
PROFILER="$1"
shift
case "$PROFILER" in
  ncu|nsys) ;;
  *) echo "Unknown profiler: $PROFILER (choose ncu or nsys)" >&2; exit 2 ;;
esac
if ! command -v "$PROFILER" >/dev/null 2>&1; then
  echo "$PROFILER is not installed or is not on PATH" >&2
  exit 1
fi

mkdir -p -- "$PROFILE_DIR"
REPORT="$PROFILE_DIR/chat_$(date +%Y%m%d_%H%M%S)"
if [[ "$PROFILER" == ncu ]]; then
  exec ncu --target-processes all --set full --export "$REPORT" \
    bash "$REPO_ROOT/scripts/run.sh" "$@"
fi
exec nsys profile --trace=cuda,nvtx,osrt --output "$REPORT" \
  bash "$REPO_ROOT/scripts/run.sh" "$@"
