#!/bin/bash
# ╔════════════════════════════════════════════════╗
# ║  Setup NVIDIA Vulkan for Headless Compute     ║
# ║  Azure VM with NVIDIA H100                    ║
# ╚════════════════════════════════════════════════╝

set -e

echo "╔════════════════════════════════════════════════╗"
echo "║  NVIDIA Vulkan Setup for Headless Compute     ║"
echo "╚════════════════════════════════════════════════╝"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# ─────────────────────────────────────────────────
# Check NVIDIA driver
# ─────────────────────────────────────────────────
echo "[1/5] Checking NVIDIA driver..."
if command -v nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null || true)
    if [ -n "$GPU_INFO" ]; then
        echo -e "${GREEN}✓ NVIDIA driver installed${NC}"
        echo "  GPU: $GPU_INFO"
    else
        echo -e "${RED}✗ NVIDIA driver not working${NC}"
        exit 1
    fi
else
    echo -e "${RED}✗ nvidia-smi not found${NC}"
    echo "  Install NVIDIA driver first"
    exit 1
fi

# ─────────────────────────────────────────────────
# Check/Install Vulkan packages
# ─────────────────────────────────────────────────
echo ""
echo "[2/5] Checking Vulkan packages..."

VULKAN_PACKAGES=(
    "vulkan-tools"
    "libvulkan1"
    "libvulkan-dev"
)

MISSING_PACKAGES=()
for pkg in "${VULKAN_PACKAGES[@]}"; do
    if ! dpkg -l | grep -q "^ii  $pkg"; then
        MISSING_PACKAGES+=("$pkg")
    fi
done

if [ ${#MISSING_PACKAGES[@]} -gt 0 ]; then
    echo -e "${YELLOW}Installing missing Vulkan packages: ${MISSING_PACKAGES[*]}${NC}"
    sudo apt-get update
    sudo apt-get install -y "${MISSING_PACKAGES[@]}"
else
    echo -e "${GREEN}✓ Vulkan packages installed${NC}"
fi

# ─────────────────────────────────────────────────
# Find NVIDIA Vulkan ICD
# ─────────────────────────────────────────────────
echo ""
echo "[3/5] Locating NVIDIA Vulkan ICD..."

ICD_PATHS=(
    "/usr/share/vulkan/icd.d/nvidia_icd.json"
    "/etc/vulkan/icd.d/nvidia_icd.json"
    "/usr/share/vulkan/icd.d/nvidia_icd.x86_64.json"
    "/usr/lib/x86_64-linux-gnu/nvidia/vulkan/icd.d/nvidia_icd.json"
)

NVIDIA_ICD=""
for path in "${ICD_PATHS[@]}"; do
    if [ -f "$path" ]; then
        NVIDIA_ICD="$path"
        echo -e "${GREEN}✓ Found NVIDIA ICD: $path${NC}"
        break
    fi
done

if [ -z "$NVIDIA_ICD" ]; then
    echo -e "${YELLOW}⚠ NVIDIA ICD not found in standard locations${NC}"
    
    # Try to find it
    echo "  Searching for nvidia_icd.json..."
    FOUND_ICD=$(find /usr -name "nvidia_icd*.json" 2>/dev/null | head -1)
    
    if [ -n "$FOUND_ICD" ]; then
        NVIDIA_ICD="$FOUND_ICD"
        echo -e "${GREEN}✓ Found: $FOUND_ICD${NC}"
    else
        echo -e "${RED}✗ Could not find NVIDIA Vulkan ICD${NC}"
        echo ""
        echo "  Try installing: sudo apt install nvidia-vulkan-icd"
        echo "  Or check if NVIDIA driver includes Vulkan support"
        exit 1
    fi
fi

# ─────────────────────────────────────────────────
# Create/Update environment file
# ─────────────────────────────────────────────────
echo ""
echo "[4/5] Setting up environment..."

ENV_FILE="$HOME/.nvidia-vulkan-env"

cat > "$ENV_FILE" << EOF
# NVIDIA Vulkan environment for headless compute
# Source this file: source $ENV_FILE

# Vulkan ICD (Installable Client Driver) path
export VK_ICD_FILENAMES="$NVIDIA_ICD"

# Alternative variable for newer Vulkan loaders
export VK_DRIVER_FILES="$NVIDIA_ICD"

# Disable display requirement for Vulkan
export VK_LOADER_DISABLE_ALL_LAYERS=1

# Optional: Debug Vulkan loader
# export VK_LOADER_DEBUG=all
EOF

echo -e "${GREEN}✓ Created: $ENV_FILE${NC}"

# Also add to .bashrc if not already there
if ! grep -q "nvidia-vulkan-env" "$HOME/.bashrc" 2>/dev/null; then
    echo "" >> "$HOME/.bashrc"
    echo "# NVIDIA Vulkan for headless compute" >> "$HOME/.bashrc"
    echo "source $ENV_FILE" >> "$HOME/.bashrc"
    echo -e "${GREEN}✓ Added to .bashrc${NC}"
fi

# Source it now
source "$ENV_FILE"

# ─────────────────────────────────────────────────
# Test Vulkan
# ─────────────────────────────────────────────────
echo ""
echo "[5/5] Testing Vulkan setup..."

# Test with vulkaninfo
if command -v vulkaninfo &> /dev/null; then
    echo "Running vulkaninfo..."
    
    # Capture vulkaninfo output
    VULKAN_OUTPUT=$(vulkaninfo --summary 2>&1 || true)
    
    if echo "$VULKAN_OUTPUT" | grep -qi "nvidia"; then
        echo -e "${GREEN}✓ NVIDIA GPU detected in Vulkan!${NC}"
        echo ""
        echo "Vulkan devices:"
        echo "$VULKAN_OUTPUT" | grep -E "(deviceName|driverVersion|apiVersion)" | head -10
    elif echo "$VULKAN_OUTPUT" | grep -qi "llvmpipe"; then
        echo -e "${YELLOW}⚠ Only software renderer (llvmpipe) found${NC}"
        echo ""
        echo "This means Vulkan is not seeing the NVIDIA GPU."
        echo "Debug output:"
        echo "$VULKAN_OUTPUT" | head -20
    else
        echo -e "${YELLOW}⚠ Unexpected vulkaninfo output${NC}"
        echo "$VULKAN_OUTPUT" | head -20
    fi
else
    echo -e "${YELLOW}⚠ vulkaninfo not found, skipping test${NC}"
    echo "  Install with: sudo apt install vulkan-tools"
fi

# ─────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Setup complete!"
echo ""
echo "Environment variables set:"
echo "  VK_ICD_FILENAMES=$VK_ICD_FILENAMES"
echo "  VK_DRIVER_FILES=$VK_DRIVER_FILES"
echo ""
echo "To apply in current shell:"
echo "  source $ENV_FILE"
echo ""
echo "To test GPU backend:"
echo "  ./wasmtime-webgpu-host/target/release/wasmtime-webgpu-host --check-gpu dummy.wasm"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
