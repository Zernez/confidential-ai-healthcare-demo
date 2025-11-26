#!/bin/bash
set -e

# ╔════════════════════════════════════════════════════════════╗
# ║  BUILD ALL - Confidential AI Healthcare Demo              ║
# ║                                                            ║
# ║  Builds all components:                                    ║
# ║  1. wasmtime-webgpu-host (Runtime with TEE attestation)   ║
# ║  2. wasm-ml (Rust WASM module)                            ║
# ║  3. wasmwebgpu-ml (C++ WASM module)                       ║
# ╚════════════════════════════════════════════════════════════╝

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║  Confidential AI Healthcare Demo - Full Build             ║"
echo "║                                                            ║"
echo "║  Components:                                               ║"
echo "║  • wasmtime-webgpu-host (Runtime + TEE Attestation)       ║"
echo "║  • wasm-ml (Rust → WASM)                                  ║"
echo "║  • wasmwebgpu-ml (C++ → WASM)                             ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

# ─────────────────────────────────────────────────
# Parse arguments
# ─────────────────────────────────────────────────
BUILD_HOST=true
BUILD_WASM_RUST=true
BUILD_WASM_CPP=true
CLEAN=false
RELEASE="--release"
SKIP_CPP=false

for arg in "$@"; do
    case $arg in
        --host-only)
            BUILD_WASM_RUST=false
            BUILD_WASM_CPP=false
            ;;
        --wasm-only)
            BUILD_HOST=false
            ;;
        --rust-only)
            BUILD_HOST=false
            BUILD_WASM_CPP=false
            ;;
        --cpp-only)
            BUILD_HOST=false
            BUILD_WASM_RUST=false
            ;;
        --skip-cpp)
            SKIP_CPP=true
            BUILD_WASM_CPP=false
            ;;
        --clean)
            CLEAN=true
            ;;
        --debug)
            RELEASE=""
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --host-only    Build only the runtime host"
            echo "  --wasm-only    Build only WASM modules (Rust + C++)"
            echo "  --rust-only    Build only Rust WASM module"
            echo "  --cpp-only     Build only C++ WASM module"
            echo "  --skip-cpp     Skip C++ build (if WASI SDK not installed)"
            echo "  --clean        Clean all build artifacts first"
            echo "  --debug        Build in debug mode (faster compile)"
            echo "  --help         Show this help message"
            echo ""
            echo "Default: Build all components in release mode"
            echo ""
            exit 0
            ;;
    esac
done

# Track build status
HOST_STATUS="⏭️  Skipped"
RUST_STATUS="⏭️  Skipped"
CPP_STATUS="⏭️  Skipped"

START_TIME=$(date +%s)

# ─────────────────────────────────────────────────
# Clean if requested
# ─────────────────────────────────────────────────
if [ "$CLEAN" = true ]; then
    echo "🧹 Cleaning all build artifacts..."
    echo ""
    
    if [ -d "wasmtime-webgpu-host/target" ]; then
        echo "  Cleaning wasmtime-webgpu-host..."
        rm -rf wasmtime-webgpu-host/target
    fi
    
    if [ -d "wasm-ml/target" ]; then
        echo "  Cleaning wasm-ml..."
        rm -rf wasm-ml/target
    fi
    
    if [ -d "wasmwebgpu-ml/build" ]; then
        echo "  Cleaning wasmwebgpu-ml..."
        rm -rf wasmwebgpu-ml/build
    fi
    
    echo "✓ Clean complete"
    echo ""
fi

# ─────────────────────────────────────────────────
# Build Runtime Host
# ─────────────────────────────────────────────────
if [ "$BUILD_HOST" = true ]; then
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📦 [1/3] Building wasmtime-webgpu-host (Runtime)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    
    if ./build_webgpu_host.sh $RELEASE; then
        HOST_STATUS="✅ Success"
    else
        HOST_STATUS="❌ Failed"
        echo ""
        echo "❌ Runtime build failed. Aborting."
        exit 1
    fi
    echo ""
fi

# ─────────────────────────────────────────────────
# Build Rust WASM Module
# ─────────────────────────────────────────────────
if [ "$BUILD_WASM_RUST" = true ]; then
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📦 [2/3] Building wasm-ml (Rust → WASM)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    
    if ./build_wasm.sh $RELEASE; then
        RUST_STATUS="✅ Success"
    else
        RUST_STATUS="❌ Failed"
        echo ""
        echo "⚠️  Rust WASM build failed. Continuing..."
    fi
    echo ""
fi

# ─────────────────────────────────────────────────
# Build C++ WASM Module
# ─────────────────────────────────────────────────
if [ "$BUILD_WASM_CPP" = true ]; then
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📦 [3/3] Building wasmwebgpu-ml (C++ → WASM)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    
    if ./build_wasmwebgpu_ml.sh; then
        CPP_STATUS="✅ Success"
    else
        CPP_STATUS="⚠️  Failed (optional)"
        echo ""
        echo "⚠️  C++ WASM build failed. This is optional."
        echo "    You may need to install WASI SDK first."
    fi
    echo ""
elif [ "$SKIP_CPP" = true ]; then
    CPP_STATUS="⏭️  Skipped (--skip-cpp)"
fi

# ─────────────────────────────────────────────────
# Build Summary
# ─────────────────────────────────────────────────
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║  BUILD COMPLETE                                            ║"
echo "╠════════════════════════════════════════════════════════════╣"
echo "║                                                            ║"
printf "║  %-20s %s\n" "wasmtime-webgpu-host:" "$HOST_STATUS" | head -c 60; echo "║"
printf "║  %-20s %s\n" "wasm-ml (Rust):" "$RUST_STATUS" | head -c 60; echo "║"
printf "║  %-20s %s\n" "wasmwebgpu-ml (C++):" "$CPP_STATUS" | head -c 60; echo "║"
echo "║                                                            ║"
printf "║  Total time: %d seconds                                   ║\n" "$DURATION"
echo "║                                                            ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# ─────────────────────────────────────────────────
# Output locations
# ─────────────────────────────────────────────────
echo "📁 Output Locations:"
echo ""

if [ "$BUILD_HOST" = true ]; then
    HOST_BIN="wasmtime-webgpu-host/target/release/wasmtime-webgpu-host"
    if [ -f "$HOST_BIN" ]; then
        echo "  Runtime Host:"
        echo "    $HOST_BIN"
        echo ""
    fi
fi

if [ "$BUILD_WASM_RUST" = true ]; then
    RUST_WASM="wasm-ml/target/wasm32-wasip1/release/wasm-ml-benchmark.wasm"
    if [ -f "$RUST_WASM" ]; then
        echo "  Rust WASM Module:"
        echo "    $RUST_WASM"
        echo ""
    fi
fi

if [ "$BUILD_WASM_CPP" = true ]; then
    CPP_WASM="wasmwebgpu-ml/build/wasmwebgpu-ml-benchmark.wasm"
    if [ -f "$CPP_WASM" ]; then
        echo "  C++ WASM Module:"
        echo "    $CPP_WASM"
        echo ""
    fi
fi

# ─────────────────────────────────────────────────
# Quick start commands
# ─────────────────────────────────────────────────
echo ""
echo "🚀 Quick Start:"
echo ""
echo "  # Run Rust WASM module:"
echo "  ./wasmtime-webgpu-host/target/release/wasmtime-webgpu-host \\"
echo "      ./wasm-ml/target/wasm32-wasip1/release/wasm-ml-benchmark.wasm \\"
echo "      --dir ./data"
echo ""
echo "  # Run C++ WASM module:"
echo "  ./wasmtime-webgpu-host/target/release/wasmtime-webgpu-host \\"
echo "      ./wasmwebgpu-ml/build/wasmwebgpu-ml-benchmark.wasm \\"
echo "      --dir ./data"
echo ""
