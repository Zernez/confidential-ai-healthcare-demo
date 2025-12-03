#!/bin/bash
set -e

# ╔════════════════════════════════════════════════╗
# ║  Run WASM with TEE Attestation + GPU          ║
# ║  Uses wasmtime-gpu-host runtime (CUDA)        ║
# ╚════════════════════════════════════════════════╝

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"

# NEW: Use wasmtime-gpu-host (CUDA backend) instead of wasmtime-webgpu-host (Vulkan)
# The CUDA backend works headless on H100 without needing display/Vulkan surface extensions
RUNTIME_CUDA="$PROJECT_ROOT/wasmtime-gpu-host/target/release/wasmtime-gpu-host"
RUNTIME_WEBGPU="$PROJECT_ROOT/wasmtime-webgpu-host/target/release/wasmtime-webgpu-host"

# Select runtime (prefer CUDA)
if [ -f "$RUNTIME_CUDA" ]; then
    RUNTIME="$RUNTIME_CUDA"
    RUNTIME_TYPE="CUDA (cuBLAS/PTX)"
elif [ -f "$RUNTIME_WEBGPU" ]; then
    RUNTIME="$RUNTIME_WEBGPU"
    RUNTIME_TYPE="WebGPU (Vulkan)"
else
    RUNTIME=""
    RUNTIME_TYPE="none"
fi

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

# Force specific backend
FORCE_BACKEND=""

# ─────────────────────────────────────────────────
# Parse arguments
# ─────────────────────────────────────────────────
show_help() {
    echo ""
    echo "╔════════════════════════════════════════════════╗"
    echo "║  Run WASM with TEE Attestation + GPU          ║"
    echo "╚════════════════════════════════════════════════╝"
    echo ""
    echo "Usage: $0 [OPTIONS] [WASM_FILE]"
    echo ""
    echo "Options:"
    echo "  --rust          Run Rust WASM module (wasm-ml)"
    echo "  --cpp           Run C++ WASM module (wasmwebgpu-ml)"
    echo "  --data-dir=DIR  Specify data directory (default: ./data)"
    echo "  --cuda          Force CUDA backend (wasmtime-gpu-host)"
    echo "  --webgpu        Force WebGPU backend (wasmtime-webgpu-host)"
    echo "  --verbose       Enable verbose logging"
    echo "  --help          Show this help message"
    echo ""
    echo "Available runtimes:"
    if [ -f "$RUNTIME_CUDA" ]; then
        echo "  ✓ wasmtime-gpu-host (CUDA) - RECOMMENDED for H100"
    else
        echo "  ✗ wasmtime-gpu-host (CUDA) - not built"
    fi
    if [ -f "$RUNTIME_WEBGPU" ]; then
        echo "  ✓ wasmtime-webgpu-host (Vulkan)"
    else
        echo "  ✗ wasmtime-webgpu-host (Vulkan) - not built"
    fi
    echo ""
    echo "Examples:"
    echo "  $0 --rust                    # Run Rust ML module with CUDA"
    echo "  $0 --cpp                     # Run C++ ML module with CUDA"
    echo "  $0 --rust --webgpu           # Force WebGPU backend"
    echo "  $0 path/to/module.wasm       # Run custom WASM file"
    echo ""
    exit 0
}

VERBOSE=""

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
        --cuda)
            FORCE_BACKEND="cuda"
            ;;
        --webgpu)
            FORCE_BACKEND="webgpu"
            ;;
        --verbose|-v)
            VERBOSE="1"
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
# Select runtime based on --cuda/--webgpu flags
# ─────────────────────────────────────────────────
if [ "$FORCE_BACKEND" = "cuda" ]; then
    if [ -f "$RUNTIME_CUDA" ]; then
        RUNTIME="$RUNTIME_CUDA"
        RUNTIME_TYPE="CUDA (cuBLAS/PTX)"
    else
        echo "Error: CUDA runtime not found: $RUNTIME_CUDA"
        echo ""
        echo "Build it with:"
        echo "  cd wasmtime-gpu-host && cargo build --release"
        exit 1
    fi
elif [ "$FORCE_BACKEND" = "webgpu" ]; then
    if [ -f "$RUNTIME_WEBGPU" ]; then
        RUNTIME="$RUNTIME_WEBGPU"
        RUNTIME_TYPE="WebGPU (Vulkan)"
    else
        echo "Error: WebGPU runtime not found: $RUNTIME_WEBGPU"
        echo ""
        echo "Build it with:"
        echo "  ./build_webgpu_host.sh"
        exit 1
    fi
fi

# ─────────────────────────────────────────────────
# Check runtime exists
# ─────────────────────────────────────────────────
if [ -z "$RUNTIME" ] || [ ! -f "$RUNTIME" ]; then
    echo "╔════════════════════════════════════════════════╗"
    echo "║  ERROR: No runtime found!                     ║"
    echo "╚════════════════════════════════════════════════╝"
    echo ""
    echo "Build one of the runtimes first:"
    echo ""
    echo "  # RECOMMENDED: CUDA backend (works headless on H100)"
    echo "  cd wasmtime-gpu-host"
    echo "  unset CC  # Important: use system compiler, not wasi-sdk"
    echo "  cargo build --release"
    echo ""
    echo "  # Alternative: WebGPU backend (requires Vulkan surface)"
    echo "  ./build_webgpu_host.sh"
    echo ""
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
        echo "Auto-detected: Rust WASM module"
    elif [ -f "$CPP_WASM" ]; then
        WASM_MODULE="$CPP_WASM"
        echo "Auto-detected: C++ WASM module"
    else
        echo "No WASM module found!"
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
    echo "WASM module not found: $WASM_MODULE"
    exit 1
fi

# ─────────────────────────────────────────────────
# Run!
# ─────────────────────────────────────────────────
echo ""
echo "╔════════════════════════════════════════════════╗"
echo "║  Running WASM with TEE Attestation + GPU      ║"
echo "╚════════════════════════════════════════════════╝"
echo ""
echo "  Runtime: $RUNTIME"
echo "  Backend: $RUNTIME_TYPE"
echo "  Module:  $WASM_MODULE"
echo "  Data:    $DATA_DIR"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Build command
CMD="$RUNTIME"

# Add verbose flag if requested
if [ -n "$VERBOSE" ]; then
    CMD="$CMD --verbose"
    export RUST_LOG=debug
fi

# Add workdir - use project root so WASM can access data/diabetes_train.csv
CMD="$CMD --workdir $PROJECT_ROOT"

# Add WASM module
CMD="$CMD $WASM_MODULE"

# Execute
exec $CMD
