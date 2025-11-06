#!/bin/bash
set -e

# ╔════════════════════════════════════════════════╗
# ║  WASM ML Local Run (no Docker)                 ║
# ╚════════════════════════════════════════════════╝

echo -e "╔════════════════════════════════════════════════╗"
echo -e "║  WASM ML Benchmark - Local Execution           ║"
echo -e "╚════════════════════════════════════════════════╝"
echo ""

cd "$(dirname "$0")"

# Check if binary exists
BINARY_PATH="wasm-ml/target/release/wasm-ml-benchmark"

if [ ! -f "$BINARY_PATH" ]; then
    echo -e "Binary not found. Building first..."
    ./run_wasm_benchmark.sh
    exit $?
fi

python3 attestation/attestation.py
if [ $? -ne 0 ]; then
    echo "Attestazione fallita sull'host. Blocco esecuzione."
    exit 1
fi

# Check if data exists
if [ ! -f "wasm-ml/data/diabetes_train.csv" ] || [ ! -f "wasm-ml/data/diabetes_test.csv" ]; then
    echo -e "Dataset not found. Exporting..."
    python3 export_diabetes_for_wasm.py
    if [ $? -ne 0 ]; then
        echo -e "Failed to export dataset"
        exit 1
    fi
fi

# Run benchmark
echo -e "Running WASM benchmark..."
echo ""

cd wasm-ml
./target/release/wasm-ml-benchmark

echo ""
echo -e "Benchmark completed!"
