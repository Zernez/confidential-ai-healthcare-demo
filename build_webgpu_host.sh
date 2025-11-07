#!/bin/bash
set -e

# ╔════════════════════════════════════════════════╗
# ║  Build Custom Wasmtime Host with WebGPU       ║
# ╚════════════════════════════════════════════════╝

echo "╔════════════════════════════════════════════════╗"
echo "║  Building Wasmtime WebGPU Host                 ║"
echo "╚════════════════════════════════════════════════╝"
echo ""

cd "$(dirname "$0")/wasmtime-webgpu-host"

# Check Rust installation
if ! command -v cargo &> /dev/null; then
    echo "❌ Rust not found. Install with:"
    echo "   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
    exit 1
fi

echo "[1/2] Building wasmtime-webgpu-host..."
echo ""

cargo build --release

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Build failed"
    exit 1
fi

echo ""
echo "[2/2] Build complete!"
echo ""

BINARY_PATH="target/release/wasmtime-webgpu-host"
BINARY_SIZE=$(ls -lh "$BINARY_PATH" | awk '{print $5}')

echo "╔════════════════════════════════════════════════╗"
echo "║  ✓ Build Complete!                            ║"
echo "╚════════════════════════════════════════════════╝"
echo ""
echo "Binary: wasmtime-webgpu-host/$BINARY_PATH"
echo "Size: $BINARY_SIZE"
echo ""
echo "To run WASM with GPU:"
echo "  ./run_with_webgpu_host.sh"
