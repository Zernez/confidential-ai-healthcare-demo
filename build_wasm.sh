#!/bin/bash
set -e

# ==================================================
echo -e "WASM ML Build Script"
echo -e "=================================================="

projectRoot="$(cd "$(dirname "$0")" && pwd)"
wasmDir="$projectRoot/wasm-ml"

# [1/6] Check Rust installation
echo -e "[1/6] Checking Rust installation..."
if ! command -v rustc &> /dev/null; then
    echo -e "Rust not found. Installing..."
    curl --proto '=https' --tlsv1.2 https://sh.rustup.rs -sSf | sh
else
    rustVersion=$(rustc --version)
    echo -e "Rust installed: $rustVersion"
fi

# [2/6] Check WASM target
echo -e "[2/6] Checking wasm32-wasi target..."
if ! rustup target list --installed | grep -q "wasm32-wasip1"; then
    echo -e "Installing wasm32-wasip1 target..."
    rustup target add wasm32-wasip1
    echo -e "wasm32-wasip1 installed"
else
    echo -e "wasm32-wasip1 already installed"
fi

# [3/6] Clean if requested
if [[ "$1" == "--clean" ]]; then
    echo -e "[3/6] Cleaning build artifacts..."
    pushd "$wasmDir" > /dev/null
    cargo clean
    popd > /dev/null
    echo -e "Clean complete"
fi

# [4/6] Run tests if requested
if [[ "$1" == "--test" ]]; then
    echo -e "[4/6] Running tests..."
    pushd "$wasmDir" > /dev/null
    cargo test
    if [ $? -ne 0 ]; then
        echo -e "Tests failed"
        popd > /dev/null
        exit 1
    fi
    popd > /dev/null
    echo -e "Tests complete"
    exit 0
fi

releaseFlag=""
buildConfig="debug"
if [[ "$1" == "--release" ]]; then
    releaseFlag="--release"
    buildConfig="release"
fi

echo -e "[5/6] Building WASM module..."
pushd "$wasmDir" > /dev/null
echo -e "Build mode: $buildConfig"
cargo build --target wasm32-wasip1 $releaseFlag
if [ $? -ne 0 ]; then
    echo -e "Build failed"
    popd > /dev/null
    exit 1
fi
popd > /dev/null
echo -e "Build complete"

# [6/6] Show output location
wasmOutput="$wasmDir/target/wasm32-wasip1/$buildConfig/wasm_ml.wasm"
if [ -f "$wasmOutput" ]; then
    wasmSize=$(du -k "$wasmOutput" | cut -f1)
    echo -e "[6/6] Build Summary"
    echo -e "  Location: $wasmOutput"
    echo -e "  Size: ${wasmSize} KB"
else
    echo -e "[6/6] Build Summary"
    echo -e "  Location: $wasmOutput (not found)"
fi