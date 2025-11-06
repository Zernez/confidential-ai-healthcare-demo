#!/bin/bash
set -e

# ╔════════════════════════════════════════════════╗
# ║  Python ML Benchmark (RAPIDS)                  ║
# ╚════════════════════════════════════════════════╝

echo -e "╔════════════════════════════════════════════════╗"
echo -e "║  Python ML Benchmark (RAPIDS)                  ║"
echo -e "╚════════════════════════════════════════════════╝"
echo ""

cd "$(dirname "$0")"

# Check if model file exists from previous run
if [ -f "model_diabetes_gpu.pkl" ]; then
    echo -e "[INFO] Removing old model file..."
    rm -f model_diabetes_gpu.pkl
fi

# Run the Python benchmark
echo -e "[RUN] Starting Python ML benchmark..."
echo ""

python3 main.py

if [ $? -ne 0 ]; then
    echo ""
    echo -e "Benchmark failed"
    exit 1
fi

echo ""
echo -e "╔════════════════════════════════════════════════╗"
echo -e "║  Python Benchmark Complete!                    ║"
echo -e "╚════════════════════════════════════════════════╝"
