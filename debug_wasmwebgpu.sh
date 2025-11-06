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
echo "2. Checking data files..."
if [ -f "wasm-ml/data/diabetes_train.csv" ]; then
    echo "   ✓ Training CSV exists ($(wc -l < wasm-ml/data/diabetes_train.csv) lines)"
else
    echo "   ❌ Training CSV not found"
    exit 1
fi

if [ -f "wasm-ml/data/diabetes_test.csv" ]; then
    echo "   ✓ Test CSV exists ($(wc -l < wasm-ml/data/diabetes_test.csv) lines)"
else
    echo "   ❌ Test CSV not found"
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
echo "4. Testing WASM execution with debug output..."
echo ""
echo "   Command: wasmtime run --dir=data::wasm-ml/data wasmwebgpu-ml/build/wasmwebgpu-ml-benchmark.wasm"
echo ""
echo "   Output:"
echo "   ─────────────────────────────────────────"

cd wasmwebgpu-ml
WASMTIME_LOG=info wasmtime run \
    --dir=data::../wasm-ml/data \
    --allow-stdio \
    build/wasmwebgpu-ml-benchmark.wasm 2>&1

EXIT_CODE=$?
cd ..

echo "   ─────────────────────────────────────────"
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    echo "   ✓ Execution successful"
else
    echo "   ❌ Execution failed with exit code: $EXIT_CODE"
    echo ""
    echo "   Trying alternative directory mapping..."
    echo ""
    
    cd wasmwebgpu-ml
    wasmtime run \
        --dir=. \
        --dir=../wasm-ml/data \
        build/wasmwebgpu-ml-benchmark.wasm 2>&1
    
    EXIT_CODE2=$?
    cd ..
    
    if [ $EXIT_CODE2 -eq 0 ]; then
        echo "   ✓ Alternative mapping worked!"
    else
        echo "   ❌ Alternative mapping also failed (exit code: $EXIT_CODE2)"
    fi
fi

echo ""
echo "=== Debug Complete ==="
