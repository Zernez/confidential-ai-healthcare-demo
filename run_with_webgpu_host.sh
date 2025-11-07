#!/bin/bash
set -e

# ╔════════════════════════════════════════════════╗
# ║  Run WASM with Custom WebGPU Host             ║
# ╚════════════════════════════════════════════════╝

echo "╔════════════════════════════════════════════════╗"
echo "║  C++ WASM ML with wasi:webgpu (Beta)          ║"
echo "╚════════════════════════════════════════════════╝"
echo ""

cd "$(dirname "$0")"

# Check if host is built
HOST_BINARY="wasmtime-webgpu-host/target/release/wasmtime-webgpu-host"

if [ ! -f "$HOST_BINARY" ]; then
    echo "Custom wasmtime host not found. Building..."
    ./build_webgpu_host.sh
    if [ $? -ne 0 ]; then
        exit 1
    fi
fi

# Check if WASM binary exists
WASM_BINARY="wasmwebgpu-ml/build/wasmwebgpu-ml-benchmark.wasm"

if [ ! -f "$WASM_BINARY" ]; then
    echo "WASM binary not found. Building..."
    ./run_wasmwebgpu_benchmark.sh
    if [ $? -ne 0 ]; then
        exit 1
    fi
fi

# Check for dataset
if [ ! -f "wasm-ml/data/diabetes_train.csv" ] || [ ! -f "wasm-ml/data/diabetes_test.csv" ]; then
    echo "Dataset not found. Exporting..."
    python3 export_diabetes_for_wasm.py
fi

echo "Running C++ WASM with GPU support..."
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Run with custom host
"$HOST_BINARY" "$WASM_BINARY" \
    --dir wasmwebgpu-ml \
    --dir wasm-ml/data

EXIT_CODE=$?

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    echo "╔════════════════════════════════════════════════╗"
    echo "║  ✓ Benchmark Complete with GPU!               ║"
    echo "╚════════════════════════════════════════════════╝"
else
    echo "❌ Benchmark failed with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi
