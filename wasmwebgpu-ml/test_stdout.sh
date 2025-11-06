#!/bin/bash

echo "Testing basic WASM stdout..."
cd "$(dirname "$0")"

# Activate environment
if [ -f "env.sh" ]; then
    source env.sh
else
    echo "Environment not found"
    exit 1
fi

echo ""
echo "1. Compiling hello world..."
$CXX test_hello.cpp -o test_hello.wasm \
    --target=wasm32-wasi \
    -O0 \
    -mexec-model=reactor

if [ $? -ne 0 ]; then
    echo "   ❌ Compilation failed"
    exit 1
fi

echo "   ✓ Compiled"
echo ""

echo "2. Running with wasmtime..."
wasmtime run test_hello.wasm

echo ""
echo "3. Exit code: $?"
