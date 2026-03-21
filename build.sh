#!/bin/bash
set -euo pipefail

BUILD_DIR="${BUILD_DIR:-build}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
JOBS="${JOBS:-$(nproc)}"
CLEAN_BUILD=1

for arg in "$@"; do
  case "$arg" in
    --no-clean)
      CLEAN_BUILD=0
      ;;
    --debug)
      BUILD_TYPE=Debug
      ;;
    --release)
      BUILD_TYPE=Release
      ;;
    *)
      echo "Unknown option: $arg" >&2
      exit 1
      ;;
  esac
done

if [ ! -d "cutlass" ]; then
  echo "📥 Cloning CUTLASS..."
  git clone --recursive https://github.com/NVIDIA/cutlass.git --depth=1
fi

if [ "$CLEAN_BUILD" -eq 1 ]; then
  echo "🧹 Cleaning ${BUILD_DIR}..."
  rm -rf "${BUILD_DIR}"
fi

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

echo "🛠️  Running CMake (${BUILD_TYPE})..."
cmake .. \
  -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
  -Dpybind11_DIR="$(python3 -c 'import pybind11; print(pybind11.get_cmake_dir())')"

echo "🔨 Building with ${JOBS} jobs..."
cmake --build . -j"${JOBS}"

echo "✅ Build Done"
