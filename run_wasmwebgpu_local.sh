#!/bin/bash
set -e

# ╔════════════════════════════════════════════════╗
# ║  WASM C++ ML Local Run (no Docker)             ║
# ╚════════════════════════════════════════════════╝

echo "╔════════════════════════════════════════════════╗"
echo "║  WASM C++ ML Benchmark - Local Execution       ║"
echo "╚════════════════════════════════════════════════╝"
echo ""

cd "$(dirname "$0")"

# Check if binary exists
BINARY_PATH="wasmwebgpu-ml/build/wasmwebgpu-ml-benchmark.wasm"

if [ ! -f "$BINARY_PATH" ]; then
    echo "Binary not found. Building first..."
    ./run_wasmwebgpu_benchmark.sh
    exit $?
fi

# Run attestation
python3 attestation/attestation.py
if [ $? -ne 0 ]; then
    echo "Attestazione fallita sull'host. Blocco esecuzione."
    exit 1
fi

# Check if data exists
if [ ! -f "wasm-ml/data/diabetes_train.csv" ] || [ ! -f "wasm-ml/data/diabetes_test.csv" ]; then
    echo "Dataset not found. Exporting..."
    python3 export_diabetes_for_wasm.py
    if [ $? -ne 0 ]; then
        echo "Failed to export dataset"
        exit 1
    fi
fi

# Enable GPU if available
export WASI_WEBGPU_ENABLED=1

# Check for WASM runtime
if command -v wasmtime &> /dev/null; then
    RUNTIME="wasmtime"
    RUNTIME_CMD="wasmtime run --dir=. --dir=./wasm-ml/data --env WASI_WEBGPU_ENABLED=1"
elif command -v wasmer &> /dev/null; then
    RUNTIME="wasmer"
    RUNTIME_CMD="wasmer run --dir=. --dir=./wasm-ml/data --env WASI_WEBGPU_ENABLED=1"
else
    echo "❌ No WASM runtime found (wasmtime or wasmer required)"
    echo "Install with:"
    echo "  curl https://wasmtime.dev/install.sh -sSf | bash"
    exit 1
fi

echo "Using $RUNTIME runtime..."
echo ""

# Run benchmark
echo "Running C++ WASM benchmark..."
echo ""

cd wasmwebgpu-ml
$RUNTIME_CMD build/wasmwebgpu-ml-benchmark.wasm

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Benchmark failed"
    exit 1
fi

echo ""
echo "✅ Benchmark completed!"
