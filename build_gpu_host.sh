#!/bin/bash
set -e

# ╔════════════════════════════════════════════════╗
# ║  Build wasmtime-gpu-host (CUDA Backend)       ║
# ║  • cuBLAS for matrix operations (Tensor Cores)║
# ║  • cuRAND for random number generation        ║
# ║  • PTX kernels for ML operations              ║
# ╚════════════════════════════════════════════════╝

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
HOST_DIR="$PROJECT_ROOT/wasmtime-gpu-host"

echo ""
echo "╔════════════════════════════════════════════════╗"
echo "║  Building wasmtime-gpu-host (CUDA Backend)    ║"
echo "╚════════════════════════════════════════════════╝"
echo ""

# ─────────────────────────────────────────────────
# Check directory exists
# ─────────────────────────────────────────────────
if [ ! -d "$HOST_DIR" ]; then
    echo "Error: wasmtime-gpu-host directory not found"
    echo "Expected: $HOST_DIR"
    exit 1
fi

cd "$HOST_DIR"

# ─────────────────────────────────────────────────
# CRITICAL: Unset CC to use system compiler
# wasi-sdk clang cannot compile native Linux binaries
# ─────────────────────────────────────────────────
echo "[1/4] Checking compiler environment..."

if [ -n "$CC" ]; then
    echo "  ⚠️  CC is set to: $CC"
    if [[ "$CC" == *"wasi"* ]]; then
        echo "  ⚠️  Detected wasi-sdk compiler - unsetting CC"
        unset CC
        unset CXX
    fi
fi

# Verify we have a system compiler
if command -v gcc &> /dev/null; then
    echo "  ✓ System compiler: $(gcc --version | head -1)"
elif command -v clang &> /dev/null; then
    echo "  ✓ System compiler: $(clang --version | head -1)"
else
    echo "  ✗ No system C compiler found!"
    echo "  Install with: sudo apt-get install build-essential"
    exit 1
fi

# ─────────────────────────────────────────────────
# Check dependencies
# ─────────────────────────────────────────────────
echo ""
echo "[2/4] Checking dependencies..."

# Check for CUDA
if command -v nvcc &> /dev/null; then
    echo "  ✓ CUDA toolkit: $(nvcc --version | grep release | awk '{print $5}' | tr -d ',')"
elif [ -d "/usr/local/cuda" ]; then
    echo "  ✓ CUDA found at /usr/local/cuda"
    export PATH="/usr/local/cuda/bin:$PATH"
    export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
else
    echo "  ⚠️  CUDA toolkit not found - will build without CUDA support"
    echo "     (WebGPU backend will still work)"
fi

# Check for pkg-config and OpenSSL
if ! command -v pkg-config &> /dev/null; then
    echo "  ⚠️  pkg-config not found"
    echo "     Install with: sudo apt-get install pkg-config"
fi

if ! pkg-config --exists openssl 2>/dev/null; then
    echo "  ⚠️  OpenSSL development headers not found"
    echo "     Install with: sudo apt-get install libssl-dev"
fi

# ─────────────────────────────────────────────────
# Build
# ─────────────────────────────────────────────────
echo ""
echo "[3/4] Building wasmtime-gpu-host..."
echo "  Mode: release"
echo "  Features: cuda (default)"
echo ""

# Clean previous build if requested
if [ "$1" = "--clean" ]; then
    echo "  Cleaning previous build..."
    cargo clean
fi

# Build with release optimizations
cargo build --release 2>&1 | while read line; do
    # Filter out some noise but show important messages
    if [[ "$line" == *"Compiling"* ]] || [[ "$line" == *"error"* ]] || [[ "$line" == *"warning:"* && "$line" != *"generated"* ]]; then
        echo "  $line"
    fi
done

# Check if build succeeded
if [ ! -f "target/release/wasmtime-gpu-host" ]; then
    echo ""
    echo "  ✗ Build failed!"
    echo ""
    echo "  Try building with verbose output:"
    echo "    cd $HOST_DIR"
    echo "    unset CC"
    echo "    cargo build --release"
    exit 1
fi

# ─────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────
echo ""
echo "[4/4] Build Summary"
echo "─────────────────────────────────────────────────"

BINARY="$HOST_DIR/target/release/wasmtime-gpu-host"
SIZE=$(du -h "$BINARY" | cut -f1)

echo "  ✓ Binary: $BINARY"
echo "  ✓ Size: $SIZE"

echo ""
echo "╔════════════════════════════════════════════════╗"
echo "║  ✓ wasmtime-gpu-host build complete!          ║"
echo "╚════════════════════════════════════════════════╝"
echo ""
echo "Features:"
echo "  ✓ CUDA backend (cuBLAS + PTX kernels)"
echo "  ✓ WebGPU fallback (wgpu/Vulkan)"
echo "  ✓ wasi:gpu interface implementation"
echo "  ✓ WASI P1 support (filesystem, stdio)"
echo ""
echo "Usage:"
echo "  $BINARY <wasm_file>"
echo "  $BINARY --workdir ./data <wasm_file>"
echo "  $BINARY --backend cuda <wasm_file>"
echo ""
echo "Or use the run script:"
echo "  ./run_with_attestation.sh --rust"
echo "  ./run_with_attestation.sh --cpp"
echo ""
