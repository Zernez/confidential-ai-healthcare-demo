#!/bin/bash
set -e

# ╔════════════════════════════════════════════════╗
# ║  WASM C++ Benchmark - Build & Run              ║
# ╚════════════════════════════════════════════════╝

echo "╔════════════════════════════════════════════════╗"
echo "║  WASM C++ ML Benchmark - Build & Run          ║"
echo "╚════════════════════════════════════════════════╝"
echo ""

cd "$(dirname "$0")"

# Step 1: Export diabetes dataset from Python (shared with Rust version)
echo "[1/4] Checking dataset..."
if [ ! -f "wasm-ml/data/diabetes_train.csv" ] || [ ! -f "wasm-ml/data/diabetes_test.csv" ]; then
    echo "  Dataset not found, exporting from Python..."
    python3 export_diabetes_for_wasm.py
    if [ $? -ne 0 ]; then
        echo "  ❌ Failed to export dataset"
        exit 1
    fi
    echo "  ✓ Dataset exported"
else
    echo "  ✓ Dataset already exists"
fi

# Copy CSV files to C++ project (symlinks don't work in WASM)
echo "  Copying CSV files to wasmwebgpu-ml/data..."
mkdir -p wasmwebgpu-ml/data
cp wasm-ml/data/diabetes_train.csv wasmwebgpu-ml/data/
cp wasm-ml/data/diabetes_test.csv wasmwebgpu-ml/data/
echo "CSV files copied"

echo ""

# Step 2: Verify CSV files
echo "[2/4] Verifying CSV files..."
TRAIN_CSV="wasmwebgpu-ml/data/diabetes_train.csv"
TEST_CSV="wasmwebgpu-ml/data/diabetes_test.csv"

if [ ! -f "$TRAIN_CSV" ]; then
    echo "Training CSV not found: $TRAIN_CSV"
    exit 1
fi
if [ ! -f "$TEST_CSV" ]; then
    echo "  ❌ Test CSV not found: $TEST_CSV"
    exit 1
fi

TRAIN_LINES=$(wc -l < "$TRAIN_CSV")
TEST_LINES=$(wc -l < "$TEST_CSV")

echo "  Training samples: $((TRAIN_LINES - 1))"
echo "  Test samples: $((TEST_LINES - 1))"
echo ""

# Step 3: Build C++ WASM binary
echo "[3/4] Building C++ WASM binary..."

# Activate WASI environment
if [ -f "wasmwebgpu-ml/env.sh" ]; then
    source wasmwebgpu-ml/env.sh
else
    echo "  ❌ Environment file not found. Run setup_wasi_cpp.sh first!"
    exit 1
fi

cd wasmwebgpu-ml
chmod +x build.sh
./build.sh

if [ $? -ne 0 ]; then
    echo "  ❌ Build failed"
    exit 1
fi

cd ..
echo ""

# Step 4: Run the benchmark
echo "[4/4] Running C++ WASM benchmark..."

BINARY_PATH="wasmwebgpu-ml/build/wasmwebgpu-ml-benchmark.wasm"

if [ ! -f "$BINARY_PATH" ]; then
    echo "  ❌ Binary not found: $BINARY_PATH"
    exit 1
fi

echo "  Binary size: $(du -h $BINARY_PATH | cut -f1)"
echo ""

# Check for WASM runtime
if command -v wasmtime &> /dev/null; then
    echo "  Using wasmtime runtime..."
    echo ""
    echo "  ─────────────────────────────────────────────"
    
    # Run with proper directory mapping
    cd wasmwebgpu-ml
    wasmtime run --dir=. build/wasmwebgpu-ml-benchmark.wasm
    
    EXIT_CODE=$?
    cd ..
    
    echo "  ─────────────────────────────────────────────"
    echo ""
    
    if [ $EXIT_CODE -ne 0 ]; then
        echo "WASM execution failed with exit code: $EXIT_CODE"
        exit $EXIT_CODE
    fi
    
elif command -v wasmer &> /dev/null; then
    echo "  Using wasmer runtime..."
    echo ""
    echo "  ─────────────────────────────────────────────"
    
    cd wasmwebgpu-ml
    wasmer run --dir=. build/wasmwebgpu-ml-benchmark.wasm
    
    EXIT_CODE=$?
    cd ..
    
    echo "  ─────────────────────────────────────────────"
    echo ""
    
    if [ $EXIT_CODE -ne 0 ]; then
        echo "WASM execution failed with exit code: $EXIT_CODE"
        exit $EXIT_CODE
    fi
else
    echo "  ❌ No WASM runtime found (wasmtime or wasmer required)"
    echo "  Install with:"
    echo "    curl https://wasmtime.dev/install.sh -sSf | bash"
    echo "  or:"
    echo "    curl https://get.wasmer.io -sSfL | sh"
    exit 1
fi

echo ""
echo "╔════════════════════════════════════════════════╗"
echo "║  ✓ C++ Benchmark Complete!                    ║"
echo "╚════════════════════════════════════════════════╝"
