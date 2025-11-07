#!/bin/bash
set -e

# ╔════════════════════════════════════════════════╗
# ║  Complete wasi:webgpu Beta Setup & Test       ║
# ╚════════════════════════════════════════════════╝

echo "╔════════════════════════════════════════════════╗"
echo "║  wasi:webgpu Beta Implementation              ║"
echo "║  Complete Setup & Test                        ║"
echo "╚════════════════════════════════════════════════╝"
echo ""

cd "$(dirname "$0")"

# Track what we've done
STEPS_DONE=0
STEPS_TOTAL=6

step_complete() {
    STEPS_DONE=$((STEPS_DONE + 1))
    echo "✓ [$STEPS_DONE/$STEPS_TOTAL] $1"
    echo ""
}

# ═══════════════════════════════════════════════
# 1. Setup wasi-gfx and generate WIT bindings
# ═══════════════════════════════════════════════

echo "[1/$STEPS_TOTAL] Setting up wasi-gfx and generating WIT bindings..."
echo ""

if [ ! -f "setup_wasi_gfx.sh" ]; then
    echo "❌ setup_wasi_gfx.sh not found"
    exit 1
fi

chmod +x setup_wasi_gfx.sh
./setup_wasi_gfx.sh

if [ $? -ne 0 ]; then
    echo "❌ Failed to setup wasi-gfx"
    exit 1
fi

step_complete "wasi-gfx setup and WIT bindings generated"

# ═══════════════════════════════════════════════
# 2. Check Rust installation
# ═══════════════════════════════════════════════

echo "[2/$STEPS_TOTAL] Checking Rust installation..."
echo ""

if ! command -v cargo &> /dev/null; then
    echo "Rust not found. Installing..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source $HOME/.cargo/env
fi

echo "Rust version: $(rustc --version)"
echo "Cargo version: $(cargo --version)"

step_complete "Rust toolchain ready"

# ═══════════════════════════════════════════════
# 3. Build custom wasmtime host
# ═══════════════════════════════════════════════

echo "[3/$STEPS_TOTAL] Building custom wasmtime host..."
echo ""

chmod +x build_webgpu_host.sh
./build_webgpu_host.sh

if [ $? -ne 0 ]; then
    echo "❌ Failed to build wasmtime host"
    exit 1
fi

step_complete "Custom wasmtime host built"

# ═══════════════════════════════════════════════
# 4. Setup C++ environment
# ═══════════════════════════════════════════════

echo "[4/$STEPS_TOTAL] Setting up C++ environment..."
echo ""

if [ ! -f "setup_wasi_cpp.sh" ]; then
    echo "❌ setup_wasi_cpp.sh not found"
    exit 1
fi

chmod +x setup_wasi_cpp.sh

if [ ! -d "$HOME/wasi-tools/wasi-sdk" ]; then
    ./setup_wasi_cpp.sh
else
    echo "WASI SDK already installed"
fi

step_complete "C++ environment ready"

# ═══════════════════════════════════════════════
# 5. Build C++ WASM with WIT bindings
# ═══════════════════════════════════════════════

echo "[5/$STEPS_TOTAL] Building C++ WASM with WIT bindings..."
echo ""

# Activate WASI environment
if [ -f "wasmwebgpu-ml/env.sh" ]; then
    source wasmwebgpu-ml/env.sh
fi

cd wasmwebgpu-ml

# Copy WIT bindings to include path
if [ -d "wit-bindings" ]; then
    echo "Using WIT bindings from wit-bindings/"
    # Update CMakeLists.txt to include wit-bindings
    if ! grep -q "wit-bindings" CMakeLists.txt; then
        sed -i 's|include_directories(${CMAKE_SOURCE_DIR}/external)|include_directories(${CMAKE_SOURCE_DIR}/external)\ninclude_directories(${CMAKE_SOURCE_DIR}/wit-bindings)|' CMakeLists.txt
    fi
fi

# Build
chmod +x build.sh
./build.sh

if [ $? -ne 0 ]; then
    echo "❌ Failed to build C++ WASM"
    cd ..
    exit 1
fi

cd ..

step_complete "C++ WASM built with WIT bindings"

# ═══════════════════════════════════════════════
# 6. Prepare dataset
# ═══════════════════════════════════════════════

echo "[6/$STEPS_TOTAL] Preparing dataset..."
echo ""

if [ ! -f "wasm-ml/data/diabetes_train.csv" ]; then
    echo "Exporting dataset..."
    python3 export_diabetes_for_wasm.py
fi

# Create symlink if needed
if [ ! -L "wasmwebgpu-ml/data" ] && [ ! -d "wasmwebgpu-ml/data" ]; then
    cd wasmwebgpu-ml
    ln -s ../wasm-ml/data data
    cd ..
fi

step_complete "Dataset ready"

# ═══════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════

echo "╔════════════════════════════════════════════════╗"
echo "║  ✓ Complete Setup Done!                       ║"
echo "╚════════════════════════════════════════════════╝"
echo ""
echo "All $STEPS_TOTAL steps completed successfully!"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Created Components:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  1. wasi-gfx repository (wasmwebgpu-ml/wasi-gfx/)"
echo "  2. WIT bindings (wasmwebgpu-ml/wit-bindings/)"
echo "  3. Custom wasmtime host (wasmtime-webgpu-host/)"
echo "  4. C++ WASM binary (wasmwebgpu-ml/build/*.wasm)"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Ready to Run!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Test the implementation:"
echo ""
echo "  chmod +x run_with_webgpu_host.sh"
echo "  ./run_with_webgpu_host.sh"
echo ""
echo "This will run the C++ ML benchmark with:"
echo "  • Real wasi:webgpu WIT bindings"
echo "  • Custom wasmtime host"
echo "  • wgpu GPU backend"
echo "  • WGSL compute shaders"
echo ""
echo "Expected output:"
echo "  • GPU initialization messages"
echo "  • Training on CPU/GPU"
echo "  • MSE: ~2875 (same as other implementations)"
echo ""
echo "For more info:"
echo "  cat WASI_WEBGPU_BETA_GUIDE.md"
echo ""
