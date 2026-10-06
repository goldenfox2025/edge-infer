# Desktop Linux development image. Jetson requires a JetPack-matched image.
FROM nvidia/cuda:12.0.1-devel-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates gcc-12 g++-12 make cmake python3 python3-dev python3-venv \
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

ARG CUDA_ARCHITECTURES=89
ARG BUILD_JOBS=2
RUN cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER=/usr/bin/g++-12 \
    -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12 \
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHITECTURES" \
    -DBUILD_TESTING=ON \
    -DPython_EXECUTABLE="$VIRTUAL_ENV/bin/python" \
    -Dpybind11_DIR="$(python -m pybind11 --cmakedir)" \
    && cmake --build build --parallel "$BUILD_JOBS" \
    && ctest --test-dir build -LE gpu --output-on-failure

ENV PYTHONPATH=/app/build:/app/bindings/python
CMD ["bash"]
