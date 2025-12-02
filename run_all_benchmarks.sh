#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# Unified Benchmark Runner
# Runs all three implementations: Python, Rust WASM, C++ WASM
# ═══════════════════════════════════════════════════════════════════════════

set -e

cd "$(dirname "$0")"

RUNTIME="./wasmtime-gpu-host/target/release/wasmtime-gpu-host"
RUST_WASM="./wasm-ml/target/wasm32-wasip1/release/wasm-ml-benchmark.wasm"
CPP_WASM="./wasmwebgpu-ml/build/wasmwebgpu-ml-benchmark.wasm"
DATA_DIR="$(pwd)/data"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_header() {
    echo ""
    echo "╔══════════════════════════════════════════════════════════════════════╗"
    echo "║                    Confidential AI Healthcare Demo                    ║"
    echo "║                      Unified Benchmark Suite                          ║"
    echo "╚══════════════════════════════════════════════════════════════════════╝"
    echo ""
}

run_python() {
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}[1/3] Running Python Baseline${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    
    cd python-baseline
    python3 src/main.py
    cd ..
}

run_rust_wasm() {
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}[2/3] Running Rust WASM${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    
    if [ ! -f "$RUST_WASM" ]; then
        echo "[ERROR] Rust WASM not found: $RUST_WASM"
        echo "        Build with: cd wasm-ml && cargo build --target wasm32-wasip1 --release"
        return 1
    fi
    
    "$RUNTIME" --backend cuda "$RUST_WASM" --dir "$DATA_DIR"
}

run_cpp_wasm() {
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}[3/3] Running C++ WASM${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    
    if [ ! -f "$CPP_WASM" ]; then
        echo "[ERROR] C++ WASM not found: $CPP_WASM"
        echo "        Build with: cd wasmwebgpu-ml && ./build.sh"
        return 1
    fi
    
    "$RUNTIME" --backend cuda "$CPP_WASM" --dir "$DATA_DIR"
}

collect_results() {
    echo ""
    echo -e "${YELLOW}╔══════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${YELLOW}║                         BENCHMARK SUMMARY                            ║${NC}"
    echo -e "${YELLOW}╚══════════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo "To extract JSON results, run each benchmark and grep for BENCHMARK_JSON:"
    echo ""
    echo "  ./run_all_benchmarks.sh 2>&1 | grep -A1 'BENCHMARK_JSON' | grep -v 'BENCHMARK_JSON' | grep -v '^--$'"
    echo ""
}

# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

print_header

case "${1:-all}" in
    python)
        run_python
        ;;
    rust)
        run_rust_wasm
        ;;
    cpp)
        run_cpp_wasm
        ;;
    all)
        run_python
        run_rust_wasm
        run_cpp_wasm
        collect_results
        ;;
    *)
        echo "Usage: $0 [python|rust|cpp|all]"
        echo ""
        echo "  python  - Run Python baseline only"
        echo "  rust    - Run Rust WASM only"
        echo "  cpp     - Run C++ WASM only"
        echo "  all     - Run all benchmarks (default)"
        exit 1
        ;;
esac
