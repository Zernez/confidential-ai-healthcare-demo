#!/bin/bash
set -e

# ╔════════════════════════════════════════════════╗
# ║  WASM C++ ML Docker Run                        ║
# ╚════════════════════════════════════════════════╝

# Image and container names
IMAGE_NAME=wasmwebgpu-ml-demo
CONTAINER_NAME=wasmwebgpu-ml-container

echo "╔════════════════════════════════════════════════╗"
echo "║  WASM C++ ML Benchmark - Docker Execution      ║"
echo "╚════════════════════════════════════════════════╝"
echo ""

cd "$(dirname "$0")"

# Run attestation on host
python3 attestation/attestation.py
if [ $? -ne 0 ]; then
    echo "Attestazione fallita sull'host. Blocco esecuzione."
    exit 1
fi

# Build Docker image
echo "Building Docker image..."
docker build -f docker/Dockerfile.wasmwebgpu -t $IMAGE_NAME .
if [ $? -ne 0 ]; then
    echo "❌ Docker build failed"
    exit 1
fi
echo "✓ Docker image built"
echo ""

# Remove existing container if present
if [ "$(docker ps -aq -f name=$CONTAINER_NAME)" ]; then
    echo "Removing existing container $CONTAINER_NAME"
    docker rm -f $CONTAINER_NAME
fi

echo "Starting container with GPU support..."
echo ""

# Map NVIDIA devices (if available)
EXTRA_DEVICES=""
for DEV in /dev/nvidia*; do
    if [ -e "$DEV" ]; then
        EXTRA_DEVICES="$EXTRA_DEVICES --device $DEV"
    fi
done

# Check if nvidia-docker is available
if command -v nvidia-docker &> /dev/null; then
    DOCKER_CMD="nvidia-docker"
    echo "Using nvidia-docker"
elif docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
    DOCKER_CMD="docker"
    GPU_FLAG="--gpus all"
    echo "Using docker with --gpus all"
else
    DOCKER_CMD="docker"
    GPU_FLAG=""
    echo "⚠ Warning: GPU support may not be available"
fi

# Run benchmark in container
$DOCKER_CMD run $GPU_FLAG --name $CONTAINER_NAME \
    --rm \
    -v $(pwd):/app \
    -w /app \
    -e WASI_WEBGPU_ENABLED=1 \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    $EXTRA_DEVICES \
    $IMAGE_NAME \
    bash -c "
        echo '=== Docker Container Environment ===' && \
        echo 'WASI SDK:' && ${CXX} --version && \
        echo 'CMake:' && cmake --version && \
        echo 'wasmtime:' && wasmtime --version && \
        echo 'Python:' && python3 --version && \
        echo '' && \
        echo '=== Exporting Dataset ===' && \
        python3 export_diabetes_for_wasm.py && \
        echo '' && \
        echo '=== Building C++ WASM ===' && \
        cd wasmwebgpu-ml && \
        mkdir -p build && \
        cd build && \
        cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE} -DBUILD_WASM=ON && \
        cmake --build . --config Release -j\$(nproc) && \
        cd .. && \
        echo '' && \
        echo '=== Running Attestation ===' && \
        python3 ../attestation/attestation.py && \
        echo '' && \
        echo '=== Running Benchmark ===' && \
        wasmtime run --dir=. --dir=../wasm-ml/data --env WASI_WEBGPU_ENABLED=1 build/wasmwebgpu-ml-benchmark.wasm
    "

RESULT=$?

if [ $RESULT -ne 0 ]; then
    echo ""
    echo "❌ Benchmark failed with exit code $RESULT"
    exit $RESULT
fi

echo ""
echo "╔════════════════════════════════════════════════╗"
echo "║  ✓ Docker Benchmark Complete!                 ║"
echo "╚════════════════════════════════════════════════╝"
