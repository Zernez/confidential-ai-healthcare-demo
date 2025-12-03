#!/bin/bash

# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  Setup NVIDIA CUDA Environment for wasmtime-gpu-host                       ║
# ║                                                                            ║
# ║  This script configures the environment to use CUDA 12.6 system libraries ║
# ║  which are compatible with cudarc/cuBLAS bindings.                        ║
# ║                                                                            ║
# ║  Usage:                                                                    ║
# ║    source ./setup_nvidia_cuda.sh                                          ║
# ║                                                                            ║
# ║  Note: Use 'source' to apply environment changes to current shell!        ║
# ╚════════════════════════════════════════════════════════════════════════════╝

set -e

echo ""
echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║  Setting up NVIDIA CUDA Environment                                       ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# Check for NVIDIA GPU
# ─────────────────────────────────────────────────────────────────────────────
echo "[1/5] Checking NVIDIA GPU..."

if ! command -v nvidia-smi &> /dev/null; then
    echo "  ✗ nvidia-smi not found. Is the NVIDIA driver installed?"
    exit 1
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)

if [ -z "$GPU_NAME" ]; then
    echo "  ✗ No NVIDIA GPU detected"
    exit 1
fi

echo "  ✓ GPU: $GPU_NAME"
echo "  ✓ Driver: $DRIVER_VERSION"

# ─────────────────────────────────────────────────────────────────────────────
# Find CUDA 12.x installation
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "[2/5] Looking for CUDA 12.x installation..."

CUDA_PATH=""

# Check common CUDA 12.x locations (prefer newer versions)
for version in 12.6 12.5 12.4 12.3 12.2 12.1 12.0; do
    if [ -d "/usr/local/cuda-$version" ]; then
        CUDA_PATH="/usr/local/cuda-$version"
        break
    fi
done

# Fallback to generic cuda path if it's version 12.x
if [ -z "$CUDA_PATH" ] && [ -d "/usr/local/cuda" ]; then
    CUDA_VERSION=$(/usr/local/cuda/bin/nvcc --version 2>/dev/null | grep "release" | sed 's/.*release \([0-9]*\.[0-9]*\).*/\1/')
    if [[ "$CUDA_VERSION" == 12.* ]]; then
        CUDA_PATH="/usr/local/cuda"
    fi
fi

if [ -z "$CUDA_PATH" ]; then
    echo "  ✗ CUDA 12.x not found!"
    echo ""
    echo "  Please install CUDA 12.x toolkit:"
    echo "    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb"
    echo "    sudo dpkg -i cuda-keyring_1.1-1_all.deb"
    echo "    sudo apt-get update"
    echo "    sudo apt-get install -y libcublas-12-6 libcurand-12-6 cuda-cudart-12-6"
    exit 1
fi

echo "  ✓ Found CUDA at: $CUDA_PATH"

# ─────────────────────────────────────────────────────────────────────────────
# Verify required libraries
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "[3/5] Verifying CUDA libraries..."

CUDA_LIB_PATH="$CUDA_PATH/lib64"

if [ ! -d "$CUDA_LIB_PATH" ]; then
    echo "  ✗ CUDA lib64 directory not found: $CUDA_LIB_PATH"
    exit 1
fi

# Check for cuBLAS
CUBLAS_LIB=$(find "$CUDA_LIB_PATH" -name "libcublas.so*" -type f 2>/dev/null | head -1)
if [ -z "$CUBLAS_LIB" ]; then
    echo "  ✗ libcublas.so not found in $CUDA_LIB_PATH"
    echo "    Install with: sudo apt-get install -y libcublas-12-6"
    exit 1
fi
echo "  ✓ cuBLAS: $(basename $CUBLAS_LIB)"

# Check for cuRAND (optional but useful)
CURAND_LIB=$(find "$CUDA_LIB_PATH" -name "libcurand.so*" -type f 2>/dev/null | head -1)
if [ -n "$CURAND_LIB" ]; then
    echo "  ✓ cuRAND: $(basename $CURAND_LIB)"
else
    echo "  ⚠ cuRAND not found (optional)"
fi

# Check for CUDA runtime
CUDART_LIB=$(find "$CUDA_LIB_PATH" -name "libcudart.so*" -type f 2>/dev/null | head -1)
if [ -n "$CUDART_LIB" ]; then
    echo "  ✓ CUDA Runtime: $(basename $CUDART_LIB)"
fi

# ─────────────────────────────────────────────────────────────────────────────
# Configure environment
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "[4/5] Configuring environment..."

# CRITICAL: Clear LD_LIBRARY_PATH to avoid conflicts with conda CUDA 13.x
# Then set only the CUDA 12.x system libraries
unset LD_LIBRARY_PATH
export LD_LIBRARY_PATH="$CUDA_LIB_PATH"

# Also add the NVIDIA driver library path if needed
if [ -d "/usr/lib/x86_64-linux-gnu" ]; then
    # Check if libcuda.so is there
    if [ -f "/usr/lib/x86_64-linux-gnu/libcuda.so" ] || [ -L "/usr/lib/x86_64-linux-gnu/libcuda.so" ]; then
        export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu"
    fi
fi

# Set CUDA_HOME for tools that need it
export CUDA_HOME="$CUDA_PATH"
export CUDA_PATH="$CUDA_PATH"

echo "  ✓ LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo "  ✓ CUDA_HOME=$CUDA_HOME"

# ─────────────────────────────────────────────────────────────────────────────
# Verify setup
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "[5/5] Verifying setup..."

# Quick test - try to load the library
if python3 -c "import ctypes; ctypes.CDLL('libcublas.so')" 2>/dev/null; then
    echo "  ✓ cuBLAS library loads successfully"
else
    # Try with full path
    if python3 -c "import ctypes; ctypes.CDLL('$CUDA_LIB_PATH/libcublas.so.12')" 2>/dev/null; then
        echo "  ✓ cuBLAS library loads successfully"
    else
        echo "  ⚠ Could not verify cuBLAS loading (may still work)"
    fi
fi

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║  ✓ CUDA Environment Ready                                                 ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "  GPU:            $GPU_NAME"
echo "  CUDA Path:      $CUDA_PATH"
echo "  cuBLAS:         $(basename $CUBLAS_LIB)"
echo ""
echo "  You can now run:"
echo "    ./run_with_attestation.sh --rust"
echo "    ./run_with_attestation.sh --cpp"
echo ""
echo "  ⚠️  Note: This environment is set for the current shell only."
echo "      Run 'source ./setup_nvidia_cuda.sh' again in new terminals."
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# Export for subprocesses
# ─────────────────────────────────────────────────────────────────────────────
export CUDA_SETUP_COMPLETE=1
