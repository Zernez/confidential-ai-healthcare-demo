#!/bin/bash
set -e

# ╔════════════════════════════════════════════════╗
# ║  WASM ML Benchmark - Build & Run              ║
# ╚════════════════════════════════════════════════╝

echo -e "╔════════════════════════════════════════════════╗"
echo -e "║  WASM ML Benchmark - Build & Run               ║"
echo -e "╚════════════════════════════════════════════════╝"
echo ""

cd "$(dirname "$0")"

# Step 1: Export diabetes dataset from Python
echo -e "[1/4] Exporting diabetes dataset from Python..."
python3 export_diabetes_for_wasm.py
if [ $? -ne 0 ]; then
    echo -e "Failed to export dataset"
    exit 1
fi
echo -e "Dataset exported"
echo ""

# Step 2: Check if CSV files exist
echo -e "[2/4] Verifying CSV files..."
TRAIN_CSV="wasm-ml/data/diabetes_train.csv"
TEST_CSV="wasm-ml/data/diabetes_test.csv"

if [ ! -f "$TRAIN_CSV" ]; then
    echo -e "Training CSV not found: $TRAIN_CSV"
    exit 1
fi
if [ ! -f "$TEST_CSV" ]; then
    echo -e "Test CSV not found: $TEST_CSV"
    exit 1
fi

TRAIN_LINES=$(wc -l < "$TRAIN_CSV")
TEST_LINES=$(wc -l < "$TEST_CSV")

echo -e "Training samples: $((TRAIN_LINES - 1))"
echo -e "Test samples: $((TEST_LINES - 1))"
echo ""

# Step 3: Build WASM binary
echo -e "[3/4] Building WASM binary..."
cd wasm-ml

echo -e "  Building with Cargo (this may take a while)..."
cargo build --release --bin wasm-ml-benchmark
if [ $? -ne 0 ]; then
    echo -e "Build failed"
    exit 1
fi

echo -e "Build completed"
cd ..
echo ""

# Step 4: Run the benchmark
echo -e "[4/4] Running WASM benchmark..."
echo ""

BINARY_PATH="wasm-ml/target/release/wasm-ml-benchmark"

if [ ! -f "$BINARY_PATH" ]; then
    echo -e "Binary not found: $BINARY_PATH"
    exit 1
fi

# Run the benchmark
cd wasm-ml
./target/release/wasm-ml-benchmark
cd ..

echo ""
echo -e "\033[36m╔════════════════════════════════════════════════╗\033[0m"
echo -e "\033[36m║  Benchmark Complete!                           ║\033[0m"
echo -e "\033[36m╚════════════════════════════════════════════════╝\033[0m"
