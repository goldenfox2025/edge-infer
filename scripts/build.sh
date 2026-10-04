#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_TYPE="${BUILD_TYPE:-Release}"
JOBS="${JOBS:-2}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
ENGINE=ON
CLEAN_BUILD=0
CMAKE_ARGS=()

while (($#)); do
  case "$1" in
    --cpu-tests) ENGINE=OFF ;;
    --debug) BUILD_TYPE=Debug ;;
    --release) BUILD_TYPE=Release ;;
    --clean) CLEAN_BUILD=1 ;;
    --no-clean) CLEAN_BUILD=0 ;;
    --) shift; CMAKE_ARGS=("$@"); break ;;
    *) echo "Usage: bash scripts/build.sh [--cpu-tests] [--debug|--release] [--clean] [-- CMake options]" >&2; exit 2 ;;
  esac
  shift
done

if [[ "$ENGINE" == OFF ]]; then
  BUILD_DIR="${BUILD_DIR:-$REPO_ROOT/build-cpu-tests}"
  if [[ "$BUILD_DIR" != /* ]]; then BUILD_DIR="$REPO_ROOT/$BUILD_DIR"; fi
else
  BUILD_DIR="${BUILD_DIR:-$REPO_ROOT/build}"
  if [[ "$BUILD_DIR" != /* ]]; then BUILD_DIR="$REPO_ROOT/$BUILD_DIR"; fi
  CUTLASS_DIR="${CUTLASS_DIR:-$REPO_ROOT/cutlass}"
  if [[ ! -f "$CUTLASS_DIR/include/cute/tensor.hpp" ]]; then
    if [[ "$CUTLASS_DIR" != "$REPO_ROOT/cutlass" ]]; then
      echo "CUTLASS_DIR does not contain include/cute/tensor.hpp: $CUTLASS_DIR" >&2
      exit 1
    fi
    # Use the gitlink recorded by this checkout, never an unpinned latest clone.
    git -C "$REPO_ROOT" submodule update --init --recursive -- cutlass
  fi
  if [[ ! -f "$CUTLASS_DIR/include/cute/tensor.hpp" ]]; then
    echo "CUTLASS headers are missing. Initialize the submodule before building." >&2
    exit 1
  fi
  PYBIND11_DIR="$("$PYTHON_BIN" -m pybind11 --cmakedir)"
  CMAKE_ARGS=(-Dpybind11_DIR="$PYBIND11_DIR" -DPython_EXECUTABLE="$(command -v "$PYTHON_BIN")" -DCUTLASS_DIR="$CUTLASS_DIR" "${CMAKE_ARGS[@]}")
fi

cmake -S "$REPO_ROOT" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
  -DLLM_INFER_BUILD_ENGINE="$ENGINE" \
  -DBUILD_TESTING=ON \
  "${CMAKE_ARGS[@]}"

if [[ "$CLEAN_BUILD" == 1 ]]; then
  cmake --build "$BUILD_DIR" --target clean
fi
cmake --build "$BUILD_DIR" --parallel "$JOBS"
