#!/bin/bash
# Build script for Python baseline with TEE attestation bindings

set -e

echo "╔════════════════════════════════════════════════════════════╗"
echo "║   Building Python Baseline (PyO3 TEE Attestation)         ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

cd "$(dirname "$0")"

# Check for maturin
if ! command -v maturin &> /dev/null; then
    echo "[ERROR] maturin not found. Install with: pip install maturin"
    exit 1
fi

# Check for Rust
if ! command -v cargo &> /dev/null; then
    echo "[ERROR] Rust not found. Install from: https://rustup.rs"
    exit 1
fi

echo "[1/3] Building tee_attestation module..."
cd tee_attestation

# Build with maturin (release mode)
maturin develop --release --features attestation-amd

if [ $? -ne 0 ]; then
    echo "[ERROR] Build failed"
    exit 1
fi

cd ..

echo ""
echo "[2/3] Verifying installation..."
python3 -c "import tee_attestation; print(f'  tee_attestation module loaded: {tee_attestation}')"

echo ""
echo "[3/3] Build complete!"
echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║    Ready to run!                                          ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "Run benchmark with:"
echo "  python3 src/main.py"
echo ""
echo "Or use the run script:"
echo "  ./run.sh"
