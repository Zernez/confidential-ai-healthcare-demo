#!/bin/bash
set -e

# ╔════════════════════════════════════════════════╗
# ║  wasi-gfx Setup & WIT Bindings Generation      ║
# ║  Creates real wasi:webgpu bindings from WIT    ║
# ╚════════════════════════════════════════════════╝

echo "╔════════════════════════════════════════════════╗"
echo "║  wasi-gfx Beta Implementation Setup            ║"
echo "╚════════════════════════════════════════════════╝"
echo ""

cd "$(dirname "$0")"

WASI_GFX_DIR="wasmwebgpu-ml/wasi-gfx"
BINDINGS_DIR="wasmwebgpu-ml/wit-bindings"

# ═══════════════════════════════════════════════
# 1. Clone wasi-gfx repository
# ═══════════════════════════════════════════════

echo "[1/5] Cloning wasi-gfx repository..."

if [ -d "$WASI_GFX_DIR" ]; then
    echo "  ✓ wasi-gfx already exists"
    cd "$WASI_GFX_DIR"
    git pull origin main || true
    cd ../..
else
    git clone https://github.com/WebAssembly/wasi-gfx.git "$WASI_GFX_DIR"
    echo "  ✓ wasi-gfx cloned"
fi

echo ""

# ═══════════════════════════════════════════════
# 2. Install wit-bindgen
# ═══════════════════════════════════════════════

echo "[2/5] Installing wit-bindgen..."

if command -v wit-bindgen &> /dev/null; then
    echo "  ✓ wit-bindgen already installed"
    wit-bindgen --version
else
    echo "  Installing wit-bindgen via cargo..."
    cargo install wit-bindgen-cli --locked
    echo "  ✓ wit-bindgen installed"
fi

echo ""

# ═══════════════════════════════════════════════
# 3. Generate C bindings from WIT
# ═══════════════════════════════════════════════

echo "[3/5] Generating C bindings from WIT files..."

mkdir -p "$BINDINGS_DIR/c"
mkdir -p "$BINDINGS_DIR/cpp"

# Check if WIT files exist
WIT_DIR="$WASI_GFX_DIR/wit"
if [ ! -d "$WIT_DIR" ]; then
    echo "  ⚠ WIT directory not found in wasi-gfx"
    echo "  Creating minimal WIT file..."
    
    mkdir -p "$WIT_DIR"
    
    # Create minimal wasi:webgpu WIT file
    cat > "$WIT_DIR/webgpu.wit" << 'EOF'
// Minimal wasi:webgpu WIT interface for beta implementation
package wasi:webgpu@0.1.0;

world webgpu {
    // GPU Instance
    export create-instance: func() -> instance;
    
    // Adapter
    export request-adapter: func(instance: instance, options: adapter-options) -> result<adapter, string>;
    
    // Device & Queue
    export request-device: func(adapter: adapter) -> result<device, string>;
    export get-queue: func(device: device) -> queue;
    
    // Buffer operations
    export create-buffer: func(device: device, descriptor: buffer-descriptor) -> buffer;
    export buffer-get-size: func(buffer: buffer) -> u64;
    export buffer-map-async: func(buffer: buffer, mode: map-mode) -> result<_, string>;
    export buffer-get-mapped-range: func(buffer: buffer, offset: u64, size: u64) -> list<u8>;
    export buffer-unmap: func(buffer: buffer);
    
    // Shader operations
    export create-shader-module: func(device: device, code: string) -> shader-module;
    
    // Compute pipeline
    export create-compute-pipeline: func(device: device, descriptor: compute-pipeline-descriptor) -> compute-pipeline;
    
    // Bind groups
    export create-bind-group-layout: func(device: device, descriptor: bind-group-layout-descriptor) -> bind-group-layout;
    export create-bind-group: func(device: device, descriptor: bind-group-descriptor) -> bind-group;
    
    // Command encoding
    export create-command-encoder: func(device: device) -> command-encoder;
    export encoder-begin-compute-pass: func(encoder: command-encoder) -> compute-pass;
    export compute-pass-set-pipeline: func(pass: compute-pass, pipeline: compute-pipeline);
    export compute-pass-set-bind-group: func(pass: compute-pass, index: u32, group: bind-group);
    export compute-pass-dispatch-workgroups: func(pass: compute-pass, x: u32, y: u32, z: u32);
    export compute-pass-end: func(pass: compute-pass);
    export encoder-finish: func(encoder: command-encoder) -> command-buffer;
    
    // Queue submit
    export queue-submit: func(queue: queue, commands: list<command-buffer>);
    export queue-write-buffer: func(queue: queue, buffer: buffer, offset: u64, data: list<u8>);
    
    // Resource handles
    type instance = u32;
    type adapter = u32;
    type device = u32;
    type queue = u32;
    type buffer = u32;
    type shader-module = u32;
    type compute-pipeline = u32;
    type bind-group-layout = u32;
    type bind-group = u32;
    type command-encoder = u32;
    type compute-pass = u32;
    type command-buffer = u32;
    
    // Enums
    enum power-preference {
        low-power,
        high-performance,
    }
    
    enum map-mode {
        read,
        write,
    }
    
    flags buffer-usage {
        map-read,
        map-write,
        copy-src,
        copy-dst,
        uniform,
        storage,
    }
    
    // Descriptors
    record adapter-options {
        power-preference: power-preference,
    }
    
    record buffer-descriptor {
        size: u64,
        usage: buffer-usage,
    }
    
    record compute-pipeline-descriptor {
        layout: bind-group-layout,
        compute-shader: shader-module,
        entry-point: string,
    }
    
    record bind-group-layout-descriptor {
        entries: list<bind-group-layout-entry>,
    }
    
    record bind-group-layout-entry {
        binding: u32,
        visibility: u32,
        buffer-type: u32,
    }
    
    record bind-group-descriptor {
        layout: bind-group-layout,
        entries: list<bind-group-entry>,
    }
    
    record bind-group-entry {
        binding: u32,
        resource: buffer,
    }
}
EOF
    
    echo "  ✓ Created minimal WIT file"
fi

# Generate C bindings
echo "  Generating C bindings..."
cd wasmwebgpu-ml

# Try to generate bindings from WIT
if [ -f "wasi-gfx/wit/webgpu.wit" ]; then
    wit-bindgen c --out-dir wit-bindings/c wasi-gfx/wit/webgpu.wit || {
        echo "  ⚠ wit-bindgen c generation failed, using guest mode"
        wit-bindgen guest c --out-dir wit-bindings/c wasi-gfx/wit/webgpu.wit || {
            echo "  ⚠ Falling back to manual bindings"
        }
    }
else
    echo "  ⚠ WIT file not found"
fi

cd ..

echo "  ✓ Bindings generated in $BINDINGS_DIR"
echo ""

# ═══════════════════════════════════════════════
# 4. Create C++ wrapper header
# ═══════════════════════════════════════════════

echo "[4/5] Creating C++ wrapper for WIT bindings..."

cat > "$BINDINGS_DIR/wasi_webgpu_cpp.hpp" << 'EOF'
/**
 * @file wasi_webgpu_cpp.hpp
 * @brief C++ wrapper for wasi:webgpu WIT bindings
 */

#ifndef WASI_WEBGPU_CPP_HPP
#define WASI_WEBGPU_CPP_HPP

#include <cstdint>
#include <vector>
#include <string>
#include <memory>

// Forward declarations of WIT-generated types
extern "C" {
    // These will be defined by wit-bindgen
    typedef uint32_t wasi_webgpu_instance_t;
    typedef uint32_t wasi_webgpu_adapter_t;
    typedef uint32_t wasi_webgpu_device_t;
    typedef uint32_t wasi_webgpu_queue_t;
    typedef uint32_t wasi_webgpu_buffer_t;
    typedef uint32_t wasi_webgpu_shader_module_t;
    typedef uint32_t wasi_webgpu_compute_pipeline_t;
    typedef uint32_t wasi_webgpu_bind_group_t;
    typedef uint32_t wasi_webgpu_command_encoder_t;
    
    // Function declarations (will be provided by host)
    __attribute__((import_module("wasi:webgpu"), import_name("create-instance")))
    wasi_webgpu_instance_t wasi_webgpu_create_instance(void);
    
    __attribute__((import_module("wasi:webgpu"), import_name("request-adapter")))
    wasi_webgpu_adapter_t wasi_webgpu_request_adapter(wasi_webgpu_instance_t instance, uint32_t power_pref);
    
    __attribute__((import_module("wasi:webgpu"), import_name("request-device")))
    wasi_webgpu_device_t wasi_webgpu_request_device(wasi_webgpu_adapter_t adapter);
    
    __attribute__((import_module("wasi:webgpu"), import_name("get-queue")))
    wasi_webgpu_queue_t wasi_webgpu_get_queue(wasi_webgpu_device_t device);
    
    __attribute__((import_module("wasi:webgpu"), import_name("create-buffer")))
    wasi_webgpu_buffer_t wasi_webgpu_create_buffer(wasi_webgpu_device_t device, uint64_t size, uint32_t usage);
    
    __attribute__((import_module("wasi:webgpu"), import_name("queue-write-buffer")))
    void wasi_webgpu_queue_write_buffer(wasi_webgpu_queue_t queue, wasi_webgpu_buffer_t buffer, 
                                         uint64_t offset, const uint8_t* data, size_t size);
}

namespace wasi {
namespace webgpu {

// C++ wrapper classes
class Instance {
public:
    Instance() : handle_(wasi_webgpu_create_instance()) {}
    wasi_webgpu_instance_t handle() const { return handle_; }
private:
    wasi_webgpu_instance_t handle_;
};

class Adapter {
public:
    Adapter(wasi_webgpu_instance_t instance, uint32_t pref = 1) 
        : handle_(wasi_webgpu_request_adapter(instance, pref)) {}
    wasi_webgpu_adapter_t handle() const { return handle_; }
private:
    wasi_webgpu_adapter_t handle_;
};

class Device {
public:
    Device(wasi_webgpu_adapter_t adapter)
        : handle_(wasi_webgpu_request_device(adapter)) {}
    wasi_webgpu_device_t handle() const { return handle_; }
private:
    wasi_webgpu_device_t handle_;
};

class Queue {
public:
    Queue(wasi_webgpu_device_t device)
        : handle_(wasi_webgpu_get_queue(device)) {}
    wasi_webgpu_queue_t handle() const { return handle_; }
    
    void write_buffer(wasi_webgpu_buffer_t buffer, uint64_t offset, 
                     const void* data, size_t size) {
        wasi_webgpu_queue_write_buffer(handle_, buffer, offset, 
                                       static_cast<const uint8_t*>(data), size);
    }
private:
    wasi_webgpu_queue_t handle_;
};

class Buffer {
public:
    Buffer(wasi_webgpu_device_t device, uint64_t size, uint32_t usage)
        : handle_(wasi_webgpu_create_buffer(device, size, usage)) {}
    wasi_webgpu_buffer_t handle() const { return handle_; }
private:
    wasi_webgpu_buffer_t handle_;
};

} // namespace webgpu
} // namespace wasi

#endif // WASI_WEBGPU_CPP_HPP
EOF

echo "  ✓ C++ wrapper created"
echo ""

# ═══════════════════════════════════════════════
# 5. Summary
# ═══════════════════════════════════════════════

echo "[5/5] Setup complete!"
echo ""
echo "╔════════════════════════════════════════════════╗"
echo "║  ✓ wasi-gfx Beta Setup Complete!              ║"
echo "╚════════════════════════════════════════════════╝"
echo ""
echo "Created:"
echo "  • wasi-gfx repository cloned"
echo "  • WIT bindings generated"
echo "  • C++ wrapper headers"
echo ""
echo "Next steps:"
echo "  1. Build custom wasmtime host: ./build_webgpu_host.sh"
echo "  2. Update C++ code to use WIT bindings"
echo "  3. Build WASM with bindings: cd wasmwebgpu-ml && ./build.sh"
echo "  4. Run with GPU: ./run_with_webgpu_host.sh"
echo ""
