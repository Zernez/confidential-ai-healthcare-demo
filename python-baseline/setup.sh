#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# Setup script for Python Baseline
# Checks CUDA compatibility and builds attestation module
# ═══════════════════════════════════════════════════════════════════════════

set -e

cd "$(dirname "$0")"

echo "╔════════════════════════════════════════════════════════════╗"
echo "║   Python Baseline Setup                                    ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# ═══════════════════════════════════════════════════════════════════════════
# Check CUDA compatibility
# ═══════════════════════════════════════════════════════════════════════════

echo "[1/4] Checking CUDA environment..."

# Get driver CUDA version
if command -v nvidia-smi &> /dev/null; then
    DRIVER_CUDA=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
    DRIVER_CUDA_VER=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
    echo "  NVIDIA Driver: $DRIVER_CUDA"
    echo "  Driver CUDA Version: $DRIVER_CUDA_VER"
else
    echo "  [WARNING] nvidia-smi not found"
    DRIVER_CUDA_VER="unknown"
fi

# Get toolkit CUDA version
if command -v nvcc &> /dev/null; then
    TOOLKIT_CUDA=$(nvcc --version | grep "release" | awk '{print $6}' | cut -d',' -f1)
    echo "  CUDA Toolkit Version: $TOOLKIT_CUDA"
    
    # Compare versions
    DRIVER_MAJOR=$(echo $DRIVER_CUDA_VER | cut -d'.' -f1)
    TOOLKIT_MAJOR=$(echo $TOOLKIT_CUDA | cut -d'.' -f1)
    
    if [ "$TOOLKIT_MAJOR" -gt "$DRIVER_MAJOR" ]; then
        echo ""
        echo "  [WARNING] CUDA toolkit ($TOOLKIT_CUDA) is newer than driver supports ($DRIVER_CUDA_VER)"
        echo "  [WARNING] cuML/RAPIDS may not work. Consider:"
        echo "           1. Update NVIDIA driver, or"
        echo "           2. Install older CUDA toolkit: conda install cuda-version=12.8"
        echo ""
        echo "  Continuing with CPU fallback mode..."
    fi
else
    echo "  [WARNING] nvcc not found"
fi

# ═══════════════════════════════════════════════════════════════════════════
# Check Rust
# ═══════════════════════════════════════════════════════════════════════════

echo ""
echo "[2/4] Checking Rust installation..."

if ! command -v cargo &> /dev/null; then
    echo "  [ERROR] Rust not found!"
    echo "  Install from: https://rustup.rs"
    echo ""
    echo "  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
    exit 1
fi

RUST_VERSION=$(rustc --version)
echo "  $RUST_VERSION"

# ═══════════════════════════════════════════════════════════════════════════
# Check maturin
# ═══════════════════════════════════════════════════════════════════════════

echo ""
echo "[3/4] Checking maturin..."

if ! command -v maturin &> /dev/null; then
    echo "  maturin not found, installing..."
    pip install maturin
fi

MATURIN_VERSION=$(maturin --version)
echo "  $MATURIN_VERSION"

# ═══════════════════════════════════════════════════════════════════════════
# Build attestation module
# ═══════════════════════════════════════════════════════════════════════════

echo ""
echo "[4/4] Building tee_attestation module..."

cd tee_attestation
maturin develop --release

if [ $? -ne 0 ]; then
    echo ""
    echo "  [ERROR] Build failed!"
    echo "  Check that you have the required system libraries:"
    echo "    - libtss2-dev (for vTPM)"
    echo "    - OpenSSL development headers"
    exit 1
fi

cd ..

# ═══════════════════════════════════════════════════════════════════════════
# Verify
# ═══════════════════════════════════════════════════════════════════════════

echo ""
echo "Verifying installation..."

python3 -c "
import tee_attestation
print('  tee_attestation module: OK')
try:
    info = tee_attestation.detect_tee()
    print(f'  TEE Detection: {info.tee_type}')
except Exception as e:
    print(f'  TEE Detection: {e}')
"

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║   Setup Complete!                                         ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "Run benchmark with:"
echo "  ./run.sh"
echo ""
echo "Or run all benchmarks:"
echo "  cd .. && ./run_all_benchmarks.sh"
