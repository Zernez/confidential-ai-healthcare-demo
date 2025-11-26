#!/bin/bash
set -e

# ╔════════════════════════════════════════════════╗
# ║  Run WASM with TEE Attestation                ║
# ║  Uses wasmtime-webgpu-host runtime            ║
# ╚════════════════════════════════════════════════╝

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
RUNTIME="$PROJECT_ROOT/wasmtime-webgpu-host/target/release/wasmtime-webgpu-host"
# Default data directory (check multiple locations)
DATA_DIR=""
for dir in "$PROJECT_ROOT/data" "$PROJECT_ROOT/wasm-ml/data"; do
    if [ -d "$dir" ]; then
        DATA_DIR="$dir"
        break
    fi
done

# Default WASM module
WASM_MODULE=""

# ─────────────────────────────────────────────────
# Parse arguments
# ─────────────────────────────────────────────────
show_help() {
    echo ""
    echo "╔════════════════════════════════════════════════╗"
    echo "║  Run WASM with TEE Attestation                ║"
    echo "╚════════════════════════════════════════════════╝"
    echo ""
    echo "Usage: $0 [OPTIONS] [WASM_FILE]"
    echo ""
    echo "Options:"
    echo "  --rust        Run Rust WASM module (wasm-ml)"
    echo "  --cpp         Run C++ WASM module (wasmwebgpu-ml)"
    echo "  --data-dir    Specify data directory (default: ./data)"
    echo "  --help        Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --rust                    # Run Rust ML module"
    echo "  $0 --cpp                     # Run C++ ML module"
    echo "  $0 path/to/module.wasm       # Run custom WASM file"
    echo ""
    exit 0
}

for arg in "$@"; do
    case $arg in
        --rust)
            WASM_MODULE="$PROJECT_ROOT/wasm-ml/target/wasm32-wasip1/release/wasm-ml-benchmark.wasm"
            ;;
        --cpp)
            WASM_MODULE="$PROJECT_ROOT/wasmwebgpu-ml/build/wasmwebgpu-ml-benchmark.wasm"
            ;;
        --data-dir=*)
            DATA_DIR="${arg#*=}"
            ;;
        --help|-h)
            show_help
            ;;
        *.wasm)
            WASM_MODULE="$arg"
            ;;
    esac
done

# ─────────────────────────────────────────────────
# Check runtime exists
# ─────────────────────────────────────────────────
if [ ! -f "$RUNTIME" ]; then
    echo "❌ Runtime not found: $RUNTIME"
    echo ""
    echo "Build it first with:"
    echo "  ./build_webgpu_host.sh --release"
    echo ""
    echo "Or build everything with:"
    echo "  ./build_all.sh"
    exit 1
fi

# ─────────────────────────────────────────────────
# Auto-detect WASM module if not specified
# ─────────────────────────────────────────────────
if [ -z "$WASM_MODULE" ]; then
    # Try Rust first
    RUST_WASM="$PROJECT_ROOT/wasm-ml/target/wasm32-wasip1/release/wasm-ml-benchmark.wasm"
    CPP_WASM="$PROJECT_ROOT/wasmwebgpu-ml/build/wasmwebgpu-ml-benchmark.wasm"
    
    if [ -f "$RUST_WASM" ]; then
        WASM_MODULE="$RUST_WASM"
        echo "📦 Auto-detected: Rust WASM module"
    elif [ -f "$CPP_WASM" ]; then
        WASM_MODULE="$CPP_WASM"
        echo "📦 Auto-detected: C++ WASM module"
    else
        echo "❌ No WASM module found!"
        echo ""
        echo "Build one first:"
        echo "  ./build_wasm.sh --release           # For Rust"
        echo "  ./build_wasmwebgpu_ml.sh            # For C++"
        echo ""
        echo "Or specify a WASM file:"
        echo "  $0 path/to/module.wasm"
        exit 1
    fi
fi

# ─────────────────────────────────────────────────
# Check WASM module exists
# ─────────────────────────────────────────────────
if [ ! -f "$WASM_MODULE" ]; then
    echo "❌ WASM module not found: $WASM_MODULE"
    exit 1
fi

# ─────────────────────────────────────────────────
# Run!
# ─────────────────────────────────────────────────
echo ""
echo "╔════════════════════════════════════════════════╗"
echo "║  Running WASM with TEE Attestation            ║"
echo "╚════════════════════════════════════════════════╝"
echo ""
echo "  Runtime: $RUNTIME"
echo "  Module:  $WASM_MODULE"
echo "  Data:    $DATA_DIR"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Build command
CMD="$RUNTIME $WASM_MODULE"

# Add data directory if found
if [ -n "$DATA_DIR" ] && [ -d "$DATA_DIR" ]; then
    CMD="$CMD --dir $DATA_DIR"
else
    echo "⚠️  Warning: Data directory not found. Run ./setup_data.sh first."
fi

# Execute
exec $CMD
