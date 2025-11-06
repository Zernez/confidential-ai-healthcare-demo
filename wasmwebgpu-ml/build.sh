#!/bin/bash
set -e

# ╔════════════════════════════════════════════════╗
# ║  WASM C++ Build Script                         ║
# ╚════════════════════════════════════════════════╝

echo "╔════════════════════════════════════════════════╗"
echo "║  Building wasmwebgpu-ml (C++ + wasi:webgpu)   ║"
echo "╚════════════════════════════════════════════════╝"
echo ""

cd "$(dirname "$0")"

# Check if environment is activated
if [ -z "$WASI_SDK_PATH" ]; then
    echo "WASI SDK environment not activated!"
    echo "Please run: source env.sh"
    exit 1
fi

echo "[1/4] Environment check..."
echo "  WASI SDK: $WASI_SDK_PATH"
echo "  Compiler: $CXX"
echo ""

# Create build directory
echo "[2/4] Creating build directory..."
mkdir -p build
cd build

# Configure with CMake
echo "[3/4] Configuring with CMake..."
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_TOOLCHAIN_FILE="$CMAKE_TOOLCHAIN_FILE" \
    -DBUILD_WASM=ON

if [ $? -ne 0 ]; then
    echo "CMake configuration failed"
    exit 1
fi

echo ""

# Build
echo "[4/4] Building..."
cmake --build . --config Release -j$(nproc 2>/dev/null || echo 4)

if [ $? -ne 0 ]; then
    echo "Build failed"
    exit 1
fi

echo ""
echo "╔════════════════════════════════════════════════╗"
echo "║    Build Complete!                            ║"
echo "╚════════════════════════════════════════════════╝"
echo ""

# Show binary info
BINARY="wasmwebgpu-ml-benchmark.wasm"
if [ -f "$BINARY" ]; then
    SIZE=$(ls -lh "$BINARY" | awk '{print $5}')
    echo "Binary: build/$BINARY"
    echo "Size: $SIZE"
    echo ""
    echo "To run: cd .. && ./run_benchmark.sh"
fi
