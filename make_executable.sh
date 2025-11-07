#!/bin/bash
# Make all benchmark scripts executable

cd "$(dirname "$0")"

echo "Making scripts executable..."

chmod +x setup_wasi_cpp.sh
chmod +x setup_conda_rapids.sh

chmod +x run_python_benchmark.sh
chmod +x run_local.sh

chmod +x run_wasm_benchmark.sh
chmod +x run_wasm_local.sh
chmod +x run_wasm_docker.sh

chmod +x run_wasmwebgpu_benchmark.sh
chmod +x run_wasmwebgpu_local.sh
chmod +x run_wasmwebgpu_docker.sh

chmod +x run_all_benchmarks.sh

chmod +x wasmwebgpu-ml/build.sh 2>/dev/null || true

echo "✅ All scripts are now executable"
echo ""
echo "You can now run:"
echo "  ./run_all_benchmarks.sh  - Run complete benchmark suite"
echo "  ./run_python_benchmark.sh - Python only"
echo "  ./run_wasm_docker.sh - Rust only"
echo "  ./run_wasmwebgpu_docker.sh - C++ only"
