#!/bin/bash
set -e

# ╔════════════════════════════════════════════════╗
# ║  wasmwebgpu-ml Build Script (C++)             ║
# ║  Compiles C++ ML module to WebAssembly        ║
# ╚════════════════════════════════════════════════╝

echo "╔════════════════════════════════════════════════╗"
echo "║  Building wasmwebgpu-ml (C++ → WASM)          ║"
echo "╚════════════════════════════════════════════════╝"
echo ""

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
CPP_DIR="$PROJECT_ROOT/wasmwebgpu-ml"
BUILD_DIR="$CPP_DIR/build"

# WASI SDK path (adjust if needed)
WASI_SDK_PATH="${WASI_SDK_PATH:-/opt/wasi-sdk}"

# ─────────────────────────────────────────────────
# [1/6] Check WASI SDK installation
# ─────────────────────────────────────────────────
echo "[1/6] Checking WASI SDK installation..."

if [ ! -d "$WASI_SDK_PATH" ]; then
    echo "⚠️  WASI SDK not found at: $WASI_SDK_PATH"
    echo ""
    echo "Installing WASI SDK..."
    
    WASI_VERSION="24"
    WASI_RELEASE="wasi-sdk-${WASI_VERSION}.0"
    
    # Detect OS
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        WASI_FILE="${WASI_RELEASE}-x86_64-linux.tar.gz"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        WASI_FILE="${WASI_RELEASE}-x86_64-macos.tar.gz"
    else
        echo "❌ Unsupported OS: $OSTYPE"
        echo "Please install WASI SDK manually from:"
        echo "  https://github.com/WebAssembly/wasi-sdk/releases"
        exit 1
    fi
    
    WASI_URL="https://github.com/WebAssembly/wasi-sdk/releases/download/${WASI_RELEASE}/${WASI_FILE}"
    
    echo "Downloading: $WASI_URL"
    wget -q --show-progress "$WASI_URL" -O "/tmp/$WASI_FILE"
    
    echo "Extracting to /opt/wasi-sdk..."
    sudo mkdir -p /opt/wasi-sdk
    sudo tar -xzf "/tmp/$WASI_FILE" -C /opt --strip-components=1
    sudo mv "/opt/${WASI_RELEASE}" /opt/wasi-sdk 2>/dev/null || true
    
    rm "/tmp/$WASI_FILE"
    
    WASI_SDK_PATH="/opt/wasi-sdk"
    echo "✓ WASI SDK installed at: $WASI_SDK_PATH"
else
    echo "✓ WASI SDK found at: $WASI_SDK_PATH"
fi

# Verify clang exists
if [ ! -f "$WASI_SDK_PATH/bin/clang" ]; then
    echo "❌ WASI clang not found at: $WASI_SDK_PATH/bin/clang"
    exit 1
fi

# ─────────────────────────────────────────────────
# [2/6] Check CMake installation
# ─────────────────────────────────────────────────
echo ""
echo "[2/6] Checking CMake installation..."
if ! command -v cmake &> /dev/null; then
    echo "❌ CMake not found. Installing..."
    if command -v apt-get &> /dev/null; then
        sudo apt-get update && sudo apt-get install -y cmake
    elif command -v brew &> /dev/null; then
        brew install cmake
    else
        echo "Please install CMake manually"
        exit 1
    fi
fi
CMAKE_VERSION=$(cmake --version | head -n1)
echo "✓ $CMAKE_VERSION"

# ─────────────────────────────────────────────────
# [3/6] Parse arguments
# ─────────────────────────────────────────────────
CLEAN=false
BUILD_TYPE="Release"
BUILD_NATIVE=false

for arg in "$@"; do
    case $arg in
        --clean)
            CLEAN=true
            ;;
        --debug)
            BUILD_TYPE="Debug"
            ;;
        --native)
            BUILD_NATIVE=true
            ;;
        --help)
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --clean      Clean build directory before building"
            echo "  --debug      Build in debug mode"
            echo "  --native     Build native binary (for testing, no WASM)"
            echo "  --help       Show this help message"
            echo ""
            exit 0
            ;;
    esac
done

# ─────────────────────────────────────────────────
# [4/6] Clean if requested
# ─────────────────────────────────────────────────
if [ "$CLEAN" = true ]; then
    echo ""
    echo "[3/6] Cleaning build directory..."
    rm -rf "$BUILD_DIR"
    echo "✓ Clean complete"
else
    echo ""
    echo "[3/6] Skipping clean (use --clean to clean)"
fi

# ─────────────────────────────────────────────────
# [5/6] Configure and build
# ─────────────────────────────────────────────────
echo ""
echo "[4/6] Configuring CMake..."

mkdir -p "$BUILD_DIR"
pushd "$BUILD_DIR" > /dev/null

if [ "$BUILD_NATIVE" = true ]; then
    echo "  Mode: Native (for testing)"
    cmake .. \
        -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
        -DBUILD_WASM=OFF \
        -DBUILD_NATIVE=ON
else
    echo "  Mode: WASM (wasm32-wasi)"
    echo "  WASI SDK: $WASI_SDK_PATH"
    cmake .. \
        -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
        -DBUILD_WASM=ON \
        -DBUILD_NATIVE=OFF \
        -DCMAKE_TOOLCHAIN_FILE="$WASI_SDK_PATH/share/cmake/wasi-sdk.cmake" \
        -DWASI_SDK_PREFIX="$WASI_SDK_PATH"
fi

echo ""
echo "[5/6] Building..."
echo "  Build type: $BUILD_TYPE"
echo ""

cmake --build . --config $BUILD_TYPE -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
BUILD_RESULT=$?

popd > /dev/null

if [ $BUILD_RESULT -ne 0 ]; then
    echo ""
    echo "❌ Build failed"
    exit 1
fi

# ─────────────────────────────────────────────────
# [6/6] Show summary
# ─────────────────────────────────────────────────
echo ""
echo "[6/6] Build Summary"
echo "─────────────────────────────────────────────────"

if [ "$BUILD_NATIVE" = true ]; then
    BINARY="$BUILD_DIR/wasmwebgpu-ml-benchmark"
    if [ -f "$BINARY" ]; then
        BINARY_SIZE=$(ls -lh "$BINARY" | awk '{print $5}')
        echo "✓ Native Binary: $BINARY"
        echo "  Size: $BINARY_SIZE"
    fi
else
    WASM_FILE="$BUILD_DIR/wasmwebgpu-ml-benchmark.wasm"
    if [ -f "$WASM_FILE" ]; then
        WASM_SIZE=$(ls -lh "$WASM_FILE" | awk '{print $5}')
        echo "✓ WASM Module: $WASM_FILE"
        echo "  Size: $WASM_SIZE"
    else
        echo "⚠️  WASM file not found at expected location"
        echo "  Looking for .wasm files in build directory..."
        find "$BUILD_DIR" -name "*.wasm" -type f
    fi
fi

echo ""
echo "╔════════════════════════════════════════════════╗"
echo "║  ✓ wasmwebgpu-ml build complete!              ║"
echo "╚════════════════════════════════════════════════╝"
echo ""

if [ "$BUILD_NATIVE" = false ]; then
    echo "To run with attestation-enabled runtime:"
    echo "  ./wasmtime-webgpu-host/target/release/wasmtime-webgpu-host \\"
    echo "      $WASM_FILE \\"
    echo "      --dir ./data"
fi
