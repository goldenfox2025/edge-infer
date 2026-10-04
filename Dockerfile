# Desktop Linux development image. Jetson requires a JetPack-matched image.
FROM nvidia/cuda:12.6.3-devel-ubuntu24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake python3 python3-dev python3-venv \
    && rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv "$VIRTUAL_ENV"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

WORKDIR /app
COPY requirements-build.txt requirements-runtime.txt ./
# PyTorch is used here to read weights/tokenize input on the host. The native
# inference module owns CUDA execution; quantization tooling is not included.
RUN python -m pip install --no-cache-dir -r requirements-build.txt -r requirements-runtime.txt \
    && python -m pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu

COPY CMakeLists.txt ./
COPY core/ core/
COPY runtime/ runtime/
COPY operators/ operators/
COPY bindings/ bindings/
COPY frontend/ frontend/
COPY scripts/ scripts/
# Initialize the recorded CUTLASS submodule on the host before docker build.
COPY cutlass/ cutlass/

ARG CUDA_ARCHITECTURES=89
ARG BUILD_JOBS=2
RUN cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHITECTURES" \
    -Dpybind11_DIR="$(python -m pybind11 --cmakedir)" \
    && cmake --build build --parallel "$BUILD_JOBS"

ENV PYTHONPATH=/app/build:/app/bindings/python
CMD ["bash"]
