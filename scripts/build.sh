#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_TYPE="${BUILD_TYPE:-Release}"
JOBS="${JOBS:-2}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
ENGINE=ON
PYTHON_BINDINGS=ON
CLEAN_BUILD=0
CMAKE_ARGS=()

while (($#)); do
  case "$1" in
    --cpu-tests) ENGINE=OFF; PYTHON_BINDINGS=OFF ;;
    --native) ENGINE=ON; PYTHON_BINDINGS=OFF ;;
    --debug) BUILD_TYPE=Debug ;;
    --release) BUILD_TYPE=Release ;;
    --clean) CLEAN_BUILD=1 ;;
    --no-clean) CLEAN_BUILD=0 ;;
    --) shift; CMAKE_ARGS=("$@"); break ;;
    *) echo "Usage: bash scripts/build.sh [--cpu-tests|--native] [--debug|--release] [--clean] [-- CMake options]" >&2; exit 2 ;;
  esac
  shift
done

if [[ "$ENGINE" == OFF ]]; then
  BUILD_DIR="${BUILD_DIR:-$REPO_ROOT/build-cpu-tests}"
  if [[ "$BUILD_DIR" != /* ]]; then BUILD_DIR="$REPO_ROOT/$BUILD_DIR"; fi
else
  BUILD_DIR="${BUILD_DIR:-$REPO_ROOT/build}"
  if [[ "$BUILD_DIR" != /* ]]; then BUILD_DIR="$REPO_ROOT/$BUILD_DIR"; fi
  if [[ "$PYTHON_BINDINGS" == ON ]]; then
    PYBIND11_DIR="$("$PYTHON_BIN" -m pybind11 --cmakedir)"
    CMAKE_ARGS=(-Dpybind11_DIR="$PYBIND11_DIR" -DPython_EXECUTABLE="$(command -v "$PYTHON_BIN")" "${CMAKE_ARGS[@]}")
  fi
fi

cmake -S "$REPO_ROOT" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
  -DEDGE_INFER_BUILD_RUNTIME="$ENGINE" \
  -DEDGE_INFER_BUILD_PYTHON="$PYTHON_BINDINGS" \
  -DBUILD_TESTING=ON \
  "${CMAKE_ARGS[@]}"

if [[ "$CLEAN_BUILD" == 1 ]]; then
  cmake --build "$BUILD_DIR" --target clean
fi
cmake --build "$BUILD_DIR" --parallel "$JOBS"
