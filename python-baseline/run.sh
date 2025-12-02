#!/bin/bash
# Run Python baseline benchmark

set -e

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║   Running Python Baseline Benchmark                        ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

cd "$(dirname "$0")"

# Check if tee_attestation is installed
if ! python3 -c "import tee_attestation" 2>/dev/null; then
    echo "[WARNING] tee_attestation module not found"
    echo "[WARNING] Run ./build.sh first to build the attestation bindings"
    echo ""
fi

# Run benchmark
python3 src/main.py
