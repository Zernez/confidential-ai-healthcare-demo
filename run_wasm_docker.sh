#!/bin/bash
set -e

# ╔════════════════════════════════════════════════╗
# ║  WASM ML Docker Run                            ║
# ╚════════════════════════════════════════════════╝

# Image and container names
IMAGE_NAME=wasm-ml-demo
CONTAINER_NAME=wasm-ml-container

echo -e "╔════════════════════════════════════════════════╗"
echo -e "║  WASM ML Benchmark - Docker Execution          ║"
echo -e "╚════════════════════════════════════════════════╝"
echo ""

cd "$(dirname "$0")"

python3 attestation/attestation.py
if [ $? -ne 0 ]; then
    echo "Attestazione fallita sull'host. Blocco esecuzione."
    exit 1
fi

# Build Docker image
echo -e "Building Docker image..."
docker build -f docker/Dockerfile.wasm -t $IMAGE_NAME .
if [ $? -ne 0 ]; then
    echo -e "Docker build failed"
    exit 1
fi
echo -e "Docker image built"
echo ""

# Remove existing container if present
if [ "$(docker ps -aq -f name=$CONTAINER_NAME)" ]; then
    echo -e "Removing existing container $CONTAINER_NAME"
    docker rm -f $CONTAINER_NAME
fi

echo -e "Starting container with GPU support..."
echo ""

# Map NVIDIA devices (if available)
EXTRA_DEVICES=""
for DEV in /dev/nvidia*; do
    if [ -e "$DEV" ]; then
        EXTRA_DEVICES="$EXTRA_DEVICES --device $DEV"
    fi
done

# Optional: Run attestation on host before container
# python3 attestation/attestation.py || { echo "Attestation failed. Blocking execution."; exit 1; }

# Run benchmark in container
docker run --gpus all --name $CONTAINER_NAME \
    --rm \
    -v $(pwd):/app \
    -w /app \
    $EXTRA_DEVICES \
    $IMAGE_NAME \
    bash -c "
        cd wasm-ml && \
        python3 ../export_diabetes_for_wasm.py && \
        cargo build --release --bin wasm-ml-benchmark && \
        ./target/release/wasm-ml-benchmark
    "

if [ $? -ne 0 ]; then
    echo -e "Benchmark failed"
    exit 1
fi

echo ""
echo -e "╔════════════════════════════════════════════════╗"
echo -e "║  Docker Benchmark Complete!                    ║"
echo -e "╚════════════════════════════════════════════════╝"
