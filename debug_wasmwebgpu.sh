#!/bin/bash

# Quick test script to debug WASM execution

echo "=== WASM Debugging Test ==="
echo ""

cd "$(dirname "$0")"

echo "1. Checking binary..."
if [ -f "wasmwebgpu-ml/build/wasmwebgpu-ml-benchmark.wasm" ]; then
    echo "   ✓ Binary exists"
    echo "   Size: $(du -h wasmwebgpu-ml/build/wasmwebgpu-ml-benchmark.wasm | cut -f1)"
else
    echo "   ❌ Binary not found"
    exit 1
fi

echo ""
echo "2. Preparing data files..."

# Remove symlink if exists
if [ -L "wasmwebgpu-ml/data" ]; then
    echo "   Removing symlink..."
    rm -f wasmwebgpu-ml/data
fi

# Create directory and copy files
mkdir -p wasmwebgpu-ml/data

if [ -f "wasm-ml/data/diabetes_train.csv" ]; then
    cp wasm-ml/data/diabetes_train.csv wasmwebgpu-ml/data/
    echo "Training CSV copied ($(wc -l < wasmwebgpu-ml/data/diabetes_train.csv) lines)"
else
    echo "Source training CSV not found"
    exit 1
fi

if [ -f "wasm-ml/data/diabetes_test.csv" ]; then
    cp wasm-ml/data/diabetes_test.csv wasmwebgpu-ml/data/
    echo "Test CSV copied ($(wc -l < wasmwebgpu-ml/data/diabetes_test.csv) lines)"
else
    echo "Source test CSV not found"
    exit 1
fi

echo ""
echo "3. Checking wasmtime..."
if command -v wasmtime &> /dev/null; then
    echo "   ✓ wasmtime found: $(wasmtime --version)"
else
    echo "   ❌ wasmtime not found"
    exit 1
fi

echo ""
echo "4. Testing WASM execution..."
echo ""
echo "   Command: wasmtime run --dir=. build/wasmwebgpu-ml-benchmark.wasm"
echo ""
echo "   Output:"
echo "   ─────────────────────────────────────────────"

cd wasmwebgpu-ml
wasmtime run --dir=. build/wasmwebgpu-ml-benchmark.wasm

EXIT_CODE=$?
cd ..

echo "   ─────────────────────────────────────────────"
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    echo "Execution successful!"
else
    echo "Execution failed with exit code: $EXIT_CODE"
fi

echo ""
echo "=== Debug Complete ==="
