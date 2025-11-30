#!/bin/bash
set -e

# ╔════════════════════════════════════════════════╗
# ║  WASI C++ Environment Setup Script             ║
# ║  Installa wasi-sdk e dipendenze per C++ WASM  ║
# ╚════════════════════════════════════════════════╝

echo "╔════════════════════════════════════════════════╗"
echo "║  WASI C++ Environment Setup                    ║"
echo "╚════════════════════════════════════════════════╝"
echo ""

# Detect OS
OS="$(uname -s)"
case "${OS}" in
    Linux*)     MACHINE=Linux;;
    Darwin*)    MACHINE=Mac;;
    CYGWIN*)    MACHINE=Cygwin;;
    MINGW*)     MACHINE=MinGw;;
    *)          MACHINE="UNKNOWN:${OS}"
esac

echo "[INFO] Detected OS: $MACHINE"
echo ""

# Set installation directory
INSTALL_DIR="$HOME/wasi-tools"
WASI_SDK_DIR="$INSTALL_DIR/wasi-sdk"
DEPS_DIR="$PWD/wasmwebgpu-ml/external"

mkdir -p "$INSTALL_DIR"
mkdir -p "$DEPS_DIR"

# ═══════════════════════════════════════════════
# 1. Install WASI SDK
# ═══════════════════════════════════════════════

WASI_SDK_VERSION="24"
WASI_SDK_VERSION_FULL="24.0"

if [ -d "$WASI_SDK_DIR" ]; then
    echo "[1/5] ✓ WASI SDK already installed at: $WASI_SDK_DIR"
else
    echo "[1/5] Installing WASI SDK $WASI_SDK_VERSION_FULL..."
    
    if [ "$MACHINE" == "Linux" ]; then
        WASI_SDK_URL="https://github.com/WebAssembly/wasi-sdk/releases/download/wasi-sdk-${WASI_SDK_VERSION}/wasi-sdk-${WASI_SDK_VERSION_FULL}-x86_64-linux.tar.gz"
        WASI_SDK_ARCHIVE="wasi-sdk-${WASI_SDK_VERSION_FULL}-x86_64-linux.tar.gz"
    elif [ "$MACHINE" == "Mac" ]; then
        WASI_SDK_URL="https://github.com/WebAssembly/wasi-sdk/releases/download/wasi-sdk-${WASI_SDK_VERSION}/wasi-sdk-${WASI_SDK_VERSION_FULL}-arm64-macos.tar.gz"
        WASI_SDK_ARCHIVE="wasi-sdk-${WASI_SDK_VERSION_FULL}-arm64-macos.tar.gz"
    else
        echo "ERROR: Unsupported OS for automatic WASI SDK installation"
        echo "Please install WASI SDK manually from: https://github.com/WebAssembly/wasi-sdk/releases"
        exit 1
    fi
    
    echo "  Downloading from: $WASI_SDK_URL"
    wget -q --show-progress "$WASI_SDK_URL" -O "/tmp/$WASI_SDK_ARCHIVE"
    
    echo "  Extracting..."
    tar -xzf "/tmp/$WASI_SDK_ARCHIVE" -C "$INSTALL_DIR"
    mv "$INSTALL_DIR/wasi-sdk-${WASI_SDK_VERSION_FULL}"* "$WASI_SDK_DIR"
    
    rm "/tmp/$WASI_SDK_ARCHIVE"
    echo "  ✓ WASI SDK installed"
fi

# Add to PATH
export WASI_SDK_PATH="$WASI_SDK_DIR"
export PATH="$WASI_SDK_DIR/bin:$PATH"

echo ""

# ═══════════════════════════════════════════════
# 2. Install/Check CMake
# ═══════════════════════════════════════════════

if command -v cmake &> /dev/null; then
    CMAKE_VERSION=$(cmake --version | head -n1 | cut -d' ' -f3)
    echo "[2/5] CMake already installed: $CMAKE_VERSION"
else
    echo "[2/5] Installing CMake..."
    
    if [ "$MACHINE" == "Linux" ]; then
        sudo apt-get update -qq
        sudo apt-get install -y cmake
    elif [ "$MACHINE" == "Mac" ]; then
        brew install cmake
    else
        echo "ERROR: Please install CMake manually"
        exit 1
    fi
    
    echo "CMake installed"
fi

echo ""

# ═══════════════════════════════════════════════
# 3. Download C++ Header-Only Libraries
# ═══════════════════════════════════════════════

echo "[3/5] Downloading C++ dependencies..."

# nlohmann/json (JSON serialization)
if [ ! -f "$DEPS_DIR/json.hpp" ]; then
    echo "  Downloading nlohmann/json..."
    wget -q --show-progress \
        "https://github.com/nlohmann/json/releases/download/v3.11.3/json.hpp" \
        -O "$DEPS_DIR/json.hpp"
    echo "nlohmann/json installed"
else
    echo "nlohmann/json already present"
fi

# fast-cpp-csv-parser (CSV parsing)
if [ ! -f "$DEPS_DIR/csv.h" ]; then
    echo "  Downloading fast-cpp-csv-parser..."
    wget -q --show-progress \
        "https://raw.githubusercontent.com/ben-strasser/fast-cpp-csv-parser/master/csv.h" \
        -O "$DEPS_DIR/csv.h"
    echo "fast-cpp-csv-parser installed"
else
    echo "fast-cpp-csv-parser already present"
fi

echo ""

# ═══════════════════════════════════════════════
# 4. Setup WASI WebGPU Bindings
# ═══════════════════════════════════════════════

echo "[4/5] Setting up WASI WebGPU bindings..."

# Create placeholder for wasi:webgpu headers
# These will be generated from WIT files or provided by the runtime
WASI_WEBGPU_INCLUDE="$DEPS_DIR/wasi"
mkdir -p "$WASI_WEBGPU_INCLUDE"

# Note: In a real setup, these would come from:
# - wasi-webgpu WIT files + wit-bindgen
# - Or provided by wasmtime/wasmer runtime
# For now, we'll create a minimal header structure

cat > "$WASI_WEBGPU_INCLUDE/webgpu.h" << 'EOF'
/* WASI WebGPU C Bindings - Placeholder
 * 
 * In production, these bindings would be generated from WIT files:
 * https://github.com/WebAssembly/wasi-gfx
 * 
 * For now, this is a minimal interface compatible with WebGPU spec
 */

#ifndef WASI_WEBGPU_H
#define WASI_WEBGPU_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations
typedef struct wgpu_instance_impl* wgpu_instance;
typedef struct wgpu_adapter_impl* wgpu_adapter;
typedef struct wgpu_device_impl* wgpu_device;
typedef struct wgpu_queue_impl* wgpu_queue;
typedef struct wgpu_buffer_impl* wgpu_buffer;
typedef struct wgpu_shader_module_impl* wgpu_shader_module;
typedef struct wgpu_compute_pipeline_impl* wgpu_compute_pipeline;
typedef struct wgpu_bind_group_impl* wgpu_bind_group;
typedef struct wgpu_bind_group_layout_impl* wgpu_bind_group_layout;
typedef struct wgpu_pipeline_layout_impl* wgpu_pipeline_layout;
typedef struct wgpu_command_encoder_impl* wgpu_command_encoder;

// Enums
typedef enum {
    WGPU_POWER_PREFERENCE_LOW_POWER = 0,
    WGPU_POWER_PREFERENCE_HIGH_PERFORMANCE = 1,
} wgpu_power_preference;

typedef enum {
    WGPU_BUFFER_USAGE_MAP_READ = 0x0001,
    WGPU_BUFFER_USAGE_MAP_WRITE = 0x0002,
    WGPU_BUFFER_USAGE_COPY_SRC = 0x0004,
    WGPU_BUFFER_USAGE_COPY_DST = 0x0008,
    WGPU_BUFFER_USAGE_STORAGE = 0x0080,
    WGPU_BUFFER_USAGE_UNIFORM = 0x0040,
} wgpu_buffer_usage;

typedef enum {
    WGPU_MAP_MODE_READ = 0x0001,
    WGPU_MAP_MODE_WRITE = 0x0002,
} wgpu_map_mode;

// Function declarations (to be implemented by runtime)
__attribute__((import_module("wasi:webgpu"), import_name("create_instance")))
wgpu_instance wgpu_create_instance(void);

__attribute__((import_module("wasi:webgpu"), import_name("request_adapter")))
wgpu_adapter wgpu_request_adapter(wgpu_instance instance, wgpu_power_preference pref);

__attribute__((import_module("wasi:webgpu"), import_name("request_device")))
wgpu_device wgpu_request_device(wgpu_adapter adapter);

__attribute__((import_module("wasi:webgpu"), import_name("get_queue")))
wgpu_queue wgpu_device_get_queue(wgpu_device device);

__attribute__((import_module("wasi:webgpu"), import_name("create_buffer")))
wgpu_buffer wgpu_device_create_buffer(wgpu_device device, size_t size, uint32_t usage);

__attribute__((import_module("wasi:webgpu"), import_name("create_shader_module")))
wgpu_shader_module wgpu_device_create_shader_module(wgpu_device device, const char* wgsl_code);

// More functions would be declared here...
// This is a minimal set for demonstration

#ifdef __cplusplus
}
#endif

#endif // WASI_WEBGPU_H
EOF

echo "WASI WebGPU headers created (minimal interface)"
echo "Note: Full bindings would come from wasi-gfx WIT files"

echo ""

# ═══════════════════════════════════════════════
# 5. Create Environment Setup File
# ═══════════════════════════════════════════════

echo "[5/5] Creating environment activation script..."

ENV_FILE="$PWD/wasmwebgpu-ml/env.sh"

cat > "$ENV_FILE" << EOF
#!/bin/bash
# WASI C++ Environment Variables
# Source this file before building: source wasmwebgpu-ml/env.sh

export WASI_SDK_PATH="$WASI_SDK_DIR"
export CC="\$WASI_SDK_PATH/bin/clang"
export CXX="\$WASI_SDK_PATH/bin/clang++"
export AR="\$WASI_SDK_PATH/bin/llvm-ar"
export RANLIB="\$WASI_SDK_PATH/bin/llvm-ranlib"
export PATH="\$WASI_SDK_PATH/bin:\$PATH"

# CMake toolchain file
export CMAKE_TOOLCHAIN_FILE="\$WASI_SDK_PATH/share/cmake/wasi-sdk.cmake"

echo "WASI C++ environment activated!"
echo "  WASI SDK: \$WASI_SDK_PATH"
echo "  Compiler: \$CXX"
echo ""
echo "Ready to build with: cd wasmwebgpu-ml && mkdir -p build && cd build && cmake .."
EOF

chmod +x "$ENV_FILE"

echo "Environment file created: $ENV_FILE"

echo ""

# ═══════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════

echo "╔════════════════════════════════════════════════╗"
echo "║  ✓ WASI C++ Environment Setup Complete!       ║"
echo "╚════════════════════════════════════════════════╝"
echo ""
echo "Installed components:"
echo "  • WASI SDK $WASI_SDK_VERSION_FULL: $WASI_SDK_DIR"
echo "  • CMake: $(cmake --version | head -n1)"
echo "  • nlohmann/json: $DEPS_DIR/json.hpp"
echo "  • fast-cpp-csv-parser: $DEPS_DIR/csv.h"
echo "  • WASI WebGPU bindings: $WASI_WEBGPU_INCLUDE/webgpu.h"
echo ""
echo "Next steps:"
echo "  1. Activate environment: source wasmwebgpu-ml/env.sh"
echo "  2. Build project: cd wasmwebgpu-ml && ./build.sh"
echo "  3. Run benchmark: ../run_wasmwebgpu_benchmark.sh"
echo ""
echo "Environment ready!"
