#!/bin/bash
set -e

# ╔════════════════════════════════════════════════╗
# ║  Complete ML Benchmark Suite                   ║
# ║  Python + Rust + C++ GPU Comparison            ║
# ╚════════════════════════════════════════════════╝

RESULTS_DIR="benchmark_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

echo "╔════════════════════════════════════════════════╗"
echo "║  Complete ML Benchmark Suite                   ║"
echo "║  Comparing: Python (RAPIDS) | Rust (wgpu) | C++║"
echo "╚════════════════════════════════════════════════╝"
echo ""
echo "Results will be saved to: $RESULTS_DIR"
echo ""

cd "$(dirname "$0")"

# ═══════════════════════════════════════════════
# System Information
# ═══════════════════════════════════════════════

echo "=== System Information ===" | tee "$RESULTS_DIR/system_info.txt"
echo "" | tee -a "$RESULTS_DIR/system_info.txt"
echo "Date: $(date)" | tee -a "$RESULTS_DIR/system_info.txt"
echo "Hostname: $(hostname)" | tee -a "$RESULTS_DIR/system_info.txt"
echo "OS: $(uname -a)" | tee -a "$RESULTS_DIR/system_info.txt"
echo "" | tee -a "$RESULTS_DIR/system_info.txt"

# GPU Info
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Information:" | tee -a "$RESULTS_DIR/system_info.txt"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | tee -a "$RESULTS_DIR/system_info.txt"
else
    echo "GPU: Not available or nvidia-smi not found" | tee -a "$RESULTS_DIR/system_info.txt"
fi

echo "" | tee -a "$RESULTS_DIR/system_info.txt"

# CPU Info
echo "CPU Information:" | tee -a "$RESULTS_DIR/system_info.txt"
lscpu | grep -E "Model name|CPU\(s\):|Thread" | tee -a "$RESULTS_DIR/system_info.txt"
echo ""

# ═══════════════════════════════════════════════
# 1. Python (RAPIDS) Benchmark
# ═══════════════════════════════════════════════

echo "╔════════════════════════════════════════════════╗"
echo "║  [1/3] Python (RAPIDS) Benchmark               ║"
echo "╚════════════════════════════════════════════════╝"
echo ""

if [ -f "run_python_benchmark.sh" ]; then
    echo "Running Python benchmark..." | tee "$RESULTS_DIR/python_results.txt"
    echo "" | tee -a "$RESULTS_DIR/python_results.txt"
    
    ./run_python_benchmark.sh 2>&1 | tee -a "$RESULTS_DIR/python_results.txt"
    
    if [ $? -eq 0 ]; then
        echo "✅ Python benchmark completed" | tee -a "$RESULTS_DIR/python_results.txt"
    else
        echo "❌ Python benchmark failed" | tee -a "$RESULTS_DIR/python_results.txt"
    fi
else
    echo "⚠️  Python benchmark script not found, skipping..." | tee "$RESULTS_DIR/python_results.txt"
fi

echo ""
sleep 2

# ═══════════════════════════════════════════════
# 2. Rust (wgpu) Benchmark
# ═══════════════════════════════════════════════

echo "╔════════════════════════════════════════════════╗"
echo "║  [2/3] Rust (wgpu) Benchmark                   ║"
echo "╚════════════════════════════════════════════════╝"
echo ""

if [ -f "run_wasm_docker.sh" ]; then
    echo "Running Rust WASM benchmark..." | tee "$RESULTS_DIR/rust_results.txt"
    echo "" | tee -a "$RESULTS_DIR/rust_results.txt"
    
    ./run_wasm_docker.sh 2>&1 | tee -a "$RESULTS_DIR/rust_results.txt"
    
    if [ $? -eq 0 ]; then
        echo "✅ Rust benchmark completed" | tee -a "$RESULTS_DIR/rust_results.txt"
    else
        echo "❌ Rust benchmark failed" | tee -a "$RESULTS_DIR/rust_results.txt"
    fi
else
    echo "⚠️  Rust benchmark script not found, skipping..." | tee "$RESULTS_DIR/rust_results.txt"
fi

echo ""
sleep 2

# ═══════════════════════════════════════════════
# 3. C++ (wasi:webgpu) Benchmark
# ═══════════════════════════════════════════════

echo "╔════════════════════════════════════════════════╗"
echo "║  [3/3] C++ (wasi:webgpu) Benchmark             ║"
echo "╚════════════════════════════════════════════════╝"
echo ""

if [ -f "run_wasmwebgpu_docker.sh" ]; then
    echo "Running C++ WASM benchmark..." | tee "$RESULTS_DIR/cpp_results.txt"
    echo "" | tee -a "$RESULTS_DIR/cpp_results.txt"
    
    ./run_wasmwebgpu_docker.sh 2>&1 | tee -a "$RESULTS_DIR/cpp_results.txt"
    
    if [ $? -eq 0 ]; then
        echo "✅ C++ benchmark completed" | tee -a "$RESULTS_DIR/cpp_results.txt"
    else
        echo "❌ C++ benchmark failed" | tee -a "$RESULTS_DIR/cpp_results.txt"
    fi
else
    echo "⚠️  C++ benchmark script not found, skipping..." | tee "$RESULTS_DIR/cpp_results.txt"
fi

echo ""

# ═══════════════════════════════════════════════
# Results Summary
# ═══════════════════════════════════════════════

echo "╔════════════════════════════════════════════════╗"
echo "║  Benchmark Results Summary                     ║"
echo "╚════════════════════════════════════════════════╝"
echo ""

SUMMARY_FILE="$RESULTS_DIR/SUMMARY.txt"

echo "=== BENCHMARK RESULTS SUMMARY ===" > "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"
echo "Date: $(date)" >> "$SUMMARY_FILE"
echo "Results Directory: $RESULTS_DIR" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

# Extract MSE
echo "┌─────────────────────────────────────────────┐" >> "$SUMMARY_FILE"
echo "│  Mean Squared Error (MSE)                   │" >> "$SUMMARY_FILE"
echo "└─────────────────────────────────────────────┘" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

for impl in python rust cpp; do
    if [ -f "$RESULTS_DIR/${impl}_results.txt" ]; then
        MSE=$(grep -i "Mean Squared Error" "$RESULTS_DIR/${impl}_results.txt" | tail -1 || echo "N/A")
        echo "${impl^^}: $MSE" >> "$SUMMARY_FILE"
    fi
done

echo "" >> "$SUMMARY_FILE"

# Extract Training Time
echo "┌─────────────────────────────────────────────┐" >> "$SUMMARY_FILE"
echo "│  Training Time                              │" >> "$SUMMARY_FILE"
echo "└─────────────────────────────────────────────┘" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

for impl in python rust cpp; do
    if [ -f "$RESULTS_DIR/${impl}_results.txt" ]; then
        TRAIN_TIME=$(grep -iE "Training (completed|time)" "$RESULTS_DIR/${impl}_results.txt" | tail -1 || echo "N/A")
        echo "${impl^^}: $TRAIN_TIME" >> "$SUMMARY_FILE"
    fi
done

echo "" >> "$SUMMARY_FILE"

# Extract Inference Time
echo "┌─────────────────────────────────────────────┐" >> "$SUMMARY_FILE"
echo "│  Inference Time                             │" >> "$SUMMARY_FILE"
echo "└─────────────────────────────────────────────┘" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

for impl in python rust cpp; do
    if [ -f "$RESULTS_DIR/${impl}_results.txt" ]; then
        INFER_TIME=$(grep -iE "Inference (time|completed)" "$RESULTS_DIR/${impl}_results.txt" | tail -1 || echo "N/A")
        echo "${impl^^}: $INFER_TIME" >> "$SUMMARY_FILE"
    fi
done

echo "" >> "$SUMMARY_FILE"

# Binary Sizes
echo "┌─────────────────────────────────────────────┐" >> "$SUMMARY_FILE"
echo "│  Binary Sizes                               │" >> "$SUMMARY_FILE"
echo "└─────────────────────────────────────────────┘" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

if [ -f "wasm-ml/target/release/wasm-ml-benchmark.wasm" ]; then
    RUST_SIZE=$(ls -lh wasm-ml/target/release/wasm-ml-benchmark.wasm | awk '{print $5}')
    echo "RUST: $RUST_SIZE" >> "$SUMMARY_FILE"
fi

if [ -f "wasmwebgpu-ml/build/wasmwebgpu-ml-benchmark.wasm" ]; then
    CPP_SIZE=$(ls -lh wasmwebgpu-ml/build/wasmwebgpu-ml-benchmark.wasm | awk '{print $5}')
    echo "C++: $CPP_SIZE" >> "$SUMMARY_FILE"
fi

echo "" >> "$SUMMARY_FILE"

# ═══════════════════════════════════════════════
# Display Summary
# ═══════════════════════════════════════════════

cat "$SUMMARY_FILE"

echo ""
echo "╔════════════════════════════════════════════════╗"
echo "║  ✅ All Benchmarks Complete!                   ║"
echo "╚════════════════════════════════════════════════╝"
echo ""
echo "Full results saved to: $RESULTS_DIR/"
echo ""
echo "Files:"
echo "  - system_info.txt     System and GPU information"
echo "  - python_results.txt  Python (RAPIDS) benchmark"
echo "  - rust_results.txt    Rust (wgpu) benchmark"
echo "  - cpp_results.txt     C++ (wasi:webgpu) benchmark"
echo "  - SUMMARY.txt         Comparison summary"
echo ""
echo "To view summary:"
echo "  cat $RESULTS_DIR/SUMMARY.txt"
echo ""
