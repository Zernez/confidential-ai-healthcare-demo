#!/bin/bash
set -e

# ╔════════════════════════════════════════════════╗
# ║  WASM-ML Build Script (Rust)                  ║
# ║  Compiles Rust ML module to WebAssembly       ║
# ╚════════════════════════════════════════════════╝

echo "╔════════════════════════════════════════════════╗"
echo "║  Building wasm-ml (Rust → WASM)               ║"
echo "╚════════════════════════════════════════════════╝"
echo ""

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
WASM_DIR="$PROJECT_ROOT/wasm-ml"
TARGET="wasm32-wasi"

# ─────────────────────────────────────────────────
# [1/5] Check Rust installation
# ─────────────────────────────────────────────────
echo "[1/5] Checking Rust installation..."
if ! command -v rustc &> /dev/null; then
    echo "❌ Rust not found. Installing..."
    curl --proto '=https' --tlsv1.2 https://sh.rustup.rs -sSf | sh -s -- -y
    source "$HOME/.cargo/env"
else
    RUST_VERSION=$(rustc --version)
    echo "✓ Rust installed: $RUST_VERSION"
fi

# ─────────────────────────────────────────────────
# [2/5] Check/Install WASM target
# ─────────────────────────────────────────────────
echo ""
echo "[2/5] Checking $TARGET target..."
if ! rustup target list --installed | grep -q "$TARGET"; then
    echo "Installing $TARGET target..."
    rustup target add $TARGET
    echo "✓ $TARGET installed"
else
    echo "✓ $TARGET already installed"
fi

# ─────────────────────────────────────────────────
# [3/5] Parse arguments
# ─────────────────────────────────────────────────
RELEASE_FLAG=""
BUILD_CONFIG="debug"
CLEAN=false
TEST_ONLY=false

for arg in "$@"; do
    case $arg in
        --release)
            RELEASE_FLAG="--release"
            BUILD_CONFIG="release"
            ;;
        --clean)
            CLEAN=true
            ;;
        --test)
            TEST_ONLY=true
            ;;
        --help)
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --release    Build in release mode (optimized)"
            echo "  --clean      Clean build artifacts before building"
            echo "  --test       Run tests only (no build)"
            echo "  --help       Show this help message"
            echo ""
            exit 0
            ;;
    esac
done

# ─────────────────────────────────────────────────
# [3/5] Clean if requested
# ─────────────────────────────────────────────────
if [ "$CLEAN" = true ]; then
    echo ""
    echo "[3/5] Cleaning build artifacts..."
    pushd "$WASM_DIR" > /dev/null
    cargo clean
    popd > /dev/null
    echo "✓ Clean complete"
else
    echo ""
    echo "[3/5] Skipping clean (use --clean to clean)"
fi

# ─────────────────────────────────────────────────
# [4/5] Run tests if requested
# ─────────────────────────────────────────────────
if [ "$TEST_ONLY" = true ]; then
    echo ""
    echo "[4/5] Running tests..."
    pushd "$WASM_DIR" > /dev/null
    cargo test
    TEST_RESULT=$?
    popd > /dev/null
    if [ $TEST_RESULT -ne 0 ]; then
        echo "❌ Tests failed"
        exit 1
    fi
    echo "✓ Tests passed"
    exit 0
fi

# ─────────────────────────────────────────────────
# [4/5] Build WASM module
# ─────────────────────────────────────────────────
echo ""
echo "[4/5] Building WASM module..."
echo "  Mode: $BUILD_CONFIG"
echo "  Target: $TARGET"
echo ""

pushd "$WASM_DIR" > /dev/null
cargo build --target $TARGET $RELEASE_FLAG
BUILD_RESULT=$?
popd > /dev/null

if [ $BUILD_RESULT -ne 0 ]; then
    echo ""
    echo "❌ Build failed"
    exit 1
fi

# ─────────────────────────────────────────────────
# [5/5] Show output summary
# ─────────────────────────────────────────────────
echo ""
echo "[5/5] Build Summary"
echo "─────────────────────────────────────────────────"

# Check for binary output
WASM_BINARY="$WASM_DIR/target/$TARGET/$BUILD_CONFIG/wasm-ml-benchmark.wasm"
WASM_LIB="$WASM_DIR/target/$TARGET/$BUILD_CONFIG/wasm_ml.wasm"

if [ -f "$WASM_BINARY" ]; then
    WASM_SIZE=$(ls -lh "$WASM_BINARY" | awk '{print $5}')
    echo "✓ Binary: $WASM_BINARY"
    echo "  Size: $WASM_SIZE"
fi

if [ -f "$WASM_LIB" ]; then
    LIB_SIZE=$(ls -lh "$WASM_LIB" | awk '{print $5}')
    echo "✓ Library: $WASM_LIB"
    echo "  Size: $LIB_SIZE"
fi

# Check for examples
EXAMPLE_DIR="$WASM_DIR/target/$TARGET/$BUILD_CONFIG/examples"
if [ -d "$EXAMPLE_DIR" ]; then
    EXAMPLES=$(ls "$EXAMPLE_DIR"/*.wasm 2>/dev/null || true)
    if [ -n "$EXAMPLES" ]; then
        echo ""
        echo "Examples:"
        for ex in $EXAMPLES; do
            EX_SIZE=$(ls -lh "$ex" | awk '{print $5}')
            echo "  • $(basename $ex) ($EX_SIZE)"
        done
    fi
fi

echo ""
echo "╔════════════════════════════════════════════════╗"
echo "║  ✓ wasm-ml build complete!                    ║"
echo "╚════════════════════════════════════════════════╝"
echo ""
echo "To run with attestation-enabled runtime:"
echo "  ./wasmtime-webgpu-host/target/release/wasmtime-webgpu-host \\"
echo "      $WASM_BINARY \\"
echo "      --dir ./data"
