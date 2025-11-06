/**
 * @file wasi_webgpu_wrapper.hpp
 * @brief C++ wrapper for WASI WebGPU C API
 * 
 * Provides RAII-style wrappers and modern C++ interface for wasi:webgpu
 * 
 * NOTE: This is a placeholder/design for the C++ interface.
 * Actual implementation depends on final wasi:webgpu C bindings.
 */

#ifndef WASI_WEBGPU_WRAPPER_HPP
#define WASI_WEBGPU_WRAPPER_HPP

#include <memory>
#include <vector>
#include <string>
#include <functional>
#include <cstdint>

// Forward declare C types (from wasi:webgpu)
// TODO: Include actual header when available
// #include <wasi/webgpu.h>

namespace wasi {
namespace webgpu {

// ═══════════════════════════════════════════════════════════════════════
// Enums (matching WebGPU spec)
// ═══════════════════════════════════════════════════════════════════════

enum class PowerPreference {
    LowPower,
    HighPerformance
};

enum class BufferUsage : uint32_t {
    MapRead = 0x0001,
    MapWrite = 0x0002,
    CopySrc = 0x0004,
    CopyDst = 0x0008,
    Uniform = 0x0040,
    Storage = 0x0080,
};

enum class MapMode : uint32_t {
    Read = 0x0001,
    Write = 0x0002,
};

// ═══════════════════════════════════════════════════════════════════════
// Forward Declarations
// ═══════════════════════════════════════════════════════════════════════

class Instance;
class Adapter;
class Device;
class Queue;
class Buffer;
class ShaderModule;
class ComputePipeline;
class BindGroup;
class CommandEncoder;

// ═══════════════════════════════════════════════════════════════════════
// RAII Wrappers
// ═══════════════════════════════════════════════════════════════════════

/**
 * @class Instance
 * @brief WebGPU instance (entry point)
 */
class Instance {
public:
    Instance();
    ~Instance();
    
    // Non-copyable
    Instance(const Instance&) = delete;
    Instance& operator=(const Instance&) = delete;
    
    // Movable
    Instance(Instance&&) noexcept;
    Instance& operator=(Instance&&) noexcept;
    
    /**
     * @brief Request GPU adapter
     */
    Adapter request_adapter(PowerPreference preference = PowerPreference::HighPerformance);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/**
 * @class Adapter
 * @brief GPU adapter (represents a physical GPU)
 */
class Adapter {
public:
    Adapter() = default;
    ~Adapter();
    
    Adapter(Adapter&&) noexcept;
    Adapter& operator=(Adapter&&) noexcept;
    
    /**
     * @brief Request logical device
     */
    Device request_device();
    
    /**
     * @brief Check if adapter is valid
     */
    bool is_valid() const;

private:
    friend class Instance;
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/**
 * @class Device
 * @brief Logical GPU device
 */
class Device {
public:
    Device() = default;
    ~Device();
    
    Device(Device&&) noexcept;
    Device& operator=(Device&&) noexcept;
    
    /**
     * @brief Get command queue
     */
    Queue get_queue();
    
    /**
     * @brief Create buffer
     */
    Buffer create_buffer(size_t size, BufferUsage usage);
    
    /**
     * @brief Create shader module from WGSL
     */
    ShaderModule create_shader_module(const std::string& wgsl_code);
    
    /**
     * @brief Check if device is valid
     */
    bool is_valid() const;

private:
    friend class Adapter;
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/**
 * @class Queue
 * @brief Command queue for submitting work
 */
class Queue {
public:
    Queue() = default;
    ~Queue();
    
    Queue(Queue&&) noexcept;
    Queue& operator=(Queue&&) noexcept;
    
    /**
     * @brief Submit commands
     */
    void submit(std::vector<CommandEncoder>&& commands);
    
    /**
     * @brief Write data to buffer
     */
    void write_buffer(Buffer& buffer, size_t offset, const void* data, size_t size);

private:
    friend class Device;
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/**
 * @class Buffer
 * @brief GPU buffer for data storage
 */
class Buffer {
public:
    Buffer() = default;
    ~Buffer();
    
    Buffer(Buffer&&) noexcept;
    Buffer& operator=(Buffer&&) noexcept;
    
    /**
     * @brief Map buffer for reading/writing
     */
    void map_async(MapMode mode, std::function<void()> callback);
    
    /**
     * @brief Get mapped range
     */
    void* get_mapped_range(size_t offset = 0, size_t size = 0);
    
    /**
     * @brief Unmap buffer
     */
    void unmap();
    
    /**
     * @brief Get buffer size
     */
    size_t size() const;

private:
    friend class Device;
    struct Impl;
    std::unique_ptr<Impl> impl_;
    size_t size_ = 0;
};

/**
 * @class ShaderModule
 * @brief Compiled shader module
 */
class ShaderModule {
public:
    ShaderModule() = default;
    ~ShaderModule();
    
    ShaderModule(ShaderModule&&) noexcept;
    ShaderModule& operator=(ShaderModule&&) noexcept;

private:
    friend class Device;
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/**
 * @class ComputePipeline
 * @brief Compute pipeline for GPU compute operations
 */
class ComputePipeline {
public:
    ComputePipeline() = default;
    ~ComputePipeline();
    
    ComputePipeline(ComputePipeline&&) noexcept;
    ComputePipeline& operator=(ComputePipeline&&) noexcept;

private:
    friend class Device;
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/**
 * @class BindGroup
 * @brief Resource bindings for shaders
 */
class BindGroup {
public:
    BindGroup() = default;
    ~BindGroup();
    
    BindGroup(BindGroup&&) noexcept;
    BindGroup& operator=(BindGroup&&) noexcept;

private:
    friend class Device;
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/**
 * @class CommandEncoder
 * @brief Records GPU commands
 */
class CommandEncoder {
public:
    CommandEncoder() = default;
    ~CommandEncoder();
    
    CommandEncoder(CommandEncoder&&) noexcept;
    CommandEncoder& operator=(CommandEncoder&&) noexcept;
    
    /**
     * @brief Begin compute pass
     */
    void begin_compute_pass();
    
    /**
     * @brief Set compute pipeline
     */
    void set_pipeline(const ComputePipeline& pipeline);
    
    /**
     * @brief Set bind group
     */
    void set_bind_group(uint32_t index, const BindGroup& group);
    
    /**
     * @brief Dispatch workgroups
     */
    void dispatch(uint32_t x, uint32_t y = 1, uint32_t z = 1);
    
    /**
     * @brief End compute pass
     */
    void end_compute_pass();

private:
    friend class Device;
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// ═══════════════════════════════════════════════════════════════════════
// Utility Functions
// ═══════════════════════════════════════════════════════════════════════

/**
 * @brief Check if WebGPU is available
 */
bool is_available();

/**
 * @brief Get WebGPU version string
 */
std::string get_version();

} // namespace webgpu
} // namespace wasi

// ═══════════════════════════════════════════════════════════════════════
// Implementation Notes
// ═══════════════════════════════════════════════════════════════════════

/*
 * This header provides a modern C++ interface to wasi:webgpu.
 * 
 * The actual implementation will depend on the final C bindings from:
 * https://github.com/WebAssembly/wasi-gfx
 * 
 * Key design principles:
 * 1. RAII: Resources automatically cleaned up
 * 2. Move-only semantics: No accidental copies
 * 3. Type safety: Enums instead of raw integers
 * 4. Modern C++: std::string, std::vector, std::function
 * 
 * Usage example:
 * 
 *   using namespace wasi::webgpu;
 *   
 *   Instance instance;
 *   Adapter adapter = instance.request_adapter();
 *   Device device = adapter.request_device();
 *   Queue queue = device.get_queue();
 *   
 *   auto buffer = device.create_buffer(1024, BufferUsage::Storage);
 *   auto shader = device.create_shader_module(wgsl_code);
 *   
 *   // ... setup pipeline, bind groups, etc.
 *   
 *   CommandEncoder encoder = device.create_command_encoder();
 *   encoder.begin_compute_pass();
 *   encoder.set_pipeline(pipeline);
 *   encoder.dispatch(32, 1, 1);
 *   encoder.end_compute_pass();
 *   
 *   queue.submit({std::move(encoder)});
 *   
 *   // Resources automatically cleaned up when out of scope
 */

#endif // WASI_WEBGPU_WRAPPER_HPP
