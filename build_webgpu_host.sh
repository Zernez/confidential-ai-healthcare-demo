#!/bin/bash
set -e

# ╔════════════════════════════════════════════════╗
# ║  Wasmtime WebGPU Host + Attestation Build     ║
# ║  Runtime with wasi:webgpu + wasmtime:attestation
# ╚════════════════════════════════════════════════╝

echo "╔════════════════════════════════════════════════╗"
echo "║  Building Wasmtime TEE Host                   ║"
echo "║  • wasi:webgpu (GPU compute)                  ║"
echo "║  • wasmtime:attestation (VM + GPU)            ║"
echo "╚════════════════════════════════════════════════╝"
echo ""

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
HOST_DIR="$PROJECT_ROOT/wasmtime-webgpu-host"

# ─────────────────────────────────────────────────
# [1/4] Check Rust installation
# ─────────────────────────────────────────────────
echo "[1/4] Checking Rust installation..."
if ! command -v cargo &> /dev/null; then
    echo "❌ Rust not found. Install with:"
    echo "   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
    exit 1
fi
RUST_VERSION=$(rustc --version)
echo "✓ Rust installed: $RUST_VERSION"

# ─────────────────────────────────────────────────
# [2/4] Parse arguments
# ─────────────────────────────────────────────────
FEATURES="attestation-all"  # Default: TDX + NVIDIA GPU attestation
CLEAN=false
DEBUG=false

for arg in "$@"; do
    case $arg in
        --no-attestation)
            FEATURES=""
            echo "⚠️  Building WITHOUT attestation support"
            ;;
        --gpu-only)
            FEATURES="attestation-nvidia"
            echo "📦 Building with GPU attestation only"
            ;;
        --tdx-only)
            FEATURES="attestation-tdx"
            echo "📦 Building with TDX attestation only"
            ;;
        --clean)
            CLEAN=true
            ;;
        --debug)
            DEBUG=true
            ;;
        --help)
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --no-attestation   Build without TEE attestation support"
            echo "  --gpu-only         Build with NVIDIA GPU attestation only"
            echo "  --tdx-only         Build with Intel TDX attestation only"
            echo "  --clean            Clean before building"
            echo "  --debug            Build in debug mode (faster compile)"
            echo "  --help             Show this help message"
            echo ""
            echo "Default: Build with full attestation (TDX + GPU)"
            echo ""
            exit 0
            ;;
    esac
done

# ─────────────────────────────────────────────────
# [2/4] Clean if requested
# ─────────────────────────────────────────────────
if [ "$CLEAN" = true ]; then
    echo ""
    echo "[2/4] Cleaning build artifacts..."
    pushd "$HOST_DIR" > /dev/null
    cargo clean
    popd > /dev/null
    echo "✓ Clean complete"
else
    echo ""
    echo "[2/4] Skipping clean (use --clean to clean)"
fi

# ─────────────────────────────────────────────────
# [3/4] Build runtime
# ─────────────────────────────────────────────────
echo ""
echo "[3/4] Building wasmtime-webgpu-host..."

# Prepare build command
BUILD_CMD="cargo build"

if [ "$DEBUG" = false ]; then
    BUILD_CMD="$BUILD_CMD --release"
    BUILD_MODE="release"
else
    BUILD_MODE="debug"
fi

if [ -n "$FEATURES" ]; then
    BUILD_CMD="$BUILD_CMD --features $FEATURES"
    echo "  Features: $FEATURES"
else
    echo "  Features: none (no attestation)"
fi

echo "  Mode: $BUILD_MODE"
echo ""

pushd "$HOST_DIR" > /dev/null
$BUILD_CMD
BUILD_RESULT=$?
popd > /dev/null

if [ $BUILD_RESULT -ne 0 ]; then
    echo ""
    echo "❌ Build failed"
    exit 1
fi

# ─────────────────────────────────────────────────
# [4/4] Show summary
# ─────────────────────────────────────────────────
echo ""
echo "[4/4] Build Summary"
echo "─────────────────────────────────────────────────"

BINARY_PATH="$HOST_DIR/target/$BUILD_MODE/wasmtime-webgpu-host"

if [ -f "$BINARY_PATH" ]; then
    BINARY_SIZE=$(ls -lh "$BINARY_PATH" | awk '{print $5}')
    echo "✓ Binary: $BINARY_PATH"
    echo "  Size: $BINARY_SIZE"
else
    echo "❌ Binary not found at: $BINARY_PATH"
    exit 1
fi

echo ""
echo "╔════════════════════════════════════════════════╗"
echo "║  ✓ wasmtime-webgpu-host build complete!       ║"
echo "╚════════════════════════════════════════════════╝"
echo ""

# Show feature summary
echo "Enabled Features:"
if [ -n "$FEATURES" ]; then
    if [[ "$FEATURES" == *"attestation-tdx"* ]] || [[ "$FEATURES" == *"attestation-all"* ]]; then
        echo "  ✓ Intel TDX VM attestation"
    fi
    if [[ "$FEATURES" == *"attestation-nvidia"* ]] || [[ "$FEATURES" == *"attestation-all"* ]]; then
        echo "  ✓ NVIDIA GPU attestation (NRAS)"
    fi
else
    echo "  ⚠️  No attestation (use default build for TEE support)"
fi
echo "  ✓ wasi:webgpu (GPU compute)"
echo ""

echo "Usage:"
echo "  $BINARY_PATH <wasm_file> [--dir <directory>]"
echo ""
echo "Example:"
echo "  $BINARY_PATH ../wasm-ml/target/wasm32-wasi/release/wasm-ml-benchmark.wasm --dir ../data"
