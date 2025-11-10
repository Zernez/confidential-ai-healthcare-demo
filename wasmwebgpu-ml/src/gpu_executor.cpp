/**
 * @file gpu_executor.cpp
 * @brief Implementation of GPU executor with wasi:webgpu direct calls
 */

#include "gpu_executor.hpp"
#include "random_forest.hpp"
#include <iostream>
#include <stdexcept>
#include <fstream>
#include <cmath>
#include <cstring>

// ═══════════════════════════════════════════════════════════════════════
// Direct wasi:webgpu Function Declarations
// ═══════════════════════════════════════════════════════════════════════

// Map C++ function names to linker names
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreturn-type-c-linkage"

extern "C" {
    __attribute__((import_module("wasi:webgpu")))
    __attribute__((import_name("create-instance")))
    uint32_t __wasi_webgpu_create_instance();
    
    __attribute__((import_module("wasi:webgpu")))
    __attribute__((import_name("request-adapter")))
    uint32_t __wasi_webgpu_request_adapter(uint32_t instance, uint32_t power_preference);
    
    __attribute__((import_module("wasi:webgpu")))
    __attribute__((import_name("request-device")))
    uint32_t __wasi_webgpu_request_device(uint32_t adapter);
    
    __attribute__((import_module("wasi:webgpu")))
    __attribute__((import_name("get-queue")))
    uint32_t __wasi_webgpu_get_queue(uint32_t device);
    
    __attribute__((import_module("wasi:webgpu")))
    __attribute__((import_name("create-buffer")))
    uint32_t __wasi_webgpu_create_buffer(uint32_t device, uint64_t size, uint32_t usage);
    
    __attribute__((import_module("wasi:webgpu")))
    __attribute__((import_name("queue-write-buffer")))
    void __wasi_webgpu_queue_write_buffer(uint32_t queue, uint32_t buffer, 
                                          uint64_t offset, const void* data, uint32_t data_len);
    
    __attribute__((import_module("wasi:webgpu")))
    __attribute__((import_name("create-shader-module")))
    uint32_t __wasi_webgpu_create_shader_module(uint32_t device, 
                                                const char* code, uint32_t code_len);
}

#pragma clang diagnostic pop

namespace ml {

/**
 * @struct GpuExecutor::Impl
 * @brief Private implementation with real wasi:webgpu calls
 */
struct GpuExecutor::Impl {
    bool gpu_available;
    
    // WebGPU resource handles (IDs returned by wasi:webgpu)
    uint32_t instance_id;
    uint32_t adapter_id;
    uint32_t device_id;
    uint32_t queue_id;
    
    // Shader cache
    std::string bootstrap_shader;
    std::string split_shader;
    std::string average_shader;
    
    Impl() : gpu_available(false), instance_id(0), adapter_id(0), 
             device_id(0), queue_id(0) {
        std::cerr << "[GPU] Initializing WebGPU via wasi:webgpu..." << std::endl;
        
        // Load shaders first
        load_shaders();
        
        // Try to initialize GPU
        if (try_init_gpu()) {
            gpu_available = true;
            std::cerr << "[GPU] wasi:webgpu initialized successfully" << std::endl;
        } else {
            std::cerr << "[GPU] wasi:webgpu not available, will use CPU fallback" << std::endl;
        }
    }
    
        bool try_init_gpu() {
            std::cerr << "[GPU] Calling wasi:webgpu create-instance..." << std::endl;
            instance_id = __wasi_webgpu_create_instance();
            std::cerr << "[GPU]   Instance ID: " << instance_id << std::endl;
            if (instance_id == 0) {
                std::cerr << "[GPU] Failed to create instance" << std::endl;
                return false;
            }

            std::cerr << "[GPU] Calling wasi:webgpu request-adapter..." << std::endl;
            adapter_id = __wasi_webgpu_request_adapter(instance_id, 1); // 1 = high performance
            std::cerr << "[GPU]   Adapter ID: " << adapter_id << std::endl;
            if (adapter_id == 0) {
                std::cerr << "[GPU] Failed to request adapter" << std::endl;
                return false;
            }

            std::cerr << "[GPU] Calling wasi:webgpu request-device..." << std::endl;
            device_id = __wasi_webgpu_request_device(adapter_id);
            std::cerr << "[GPU]   Device ID: " << device_id << std::endl;
            if (device_id == 0) {
                std::cerr << "[GPU] Failed to request device" << std::endl;
                return false;
            }

            std::cerr << "[GPU] Calling wasi:webgpu get-queue..." << std::endl;
            queue_id = __wasi_webgpu_get_queue(device_id);
            std::cerr << "[GPU] Queue ID: " << queue_id << std::endl;
            if (queue_id == 0) {
                std::cerr << "[GPU] Failed to get queue" << std::endl;
                return false;
            }

            return true;
        }
    
    void load_shaders() {
        // Load WGSL shaders (mounted directly without shaders/ prefix)
        bootstrap_shader = load_shader_file("shaders/bootstrap_sample.wgsl");
        split_shader = load_shader_file("find_split.wgsl");
        average_shader = load_shader_file("average.wgsl");
        
        if (!bootstrap_shader.empty() && !split_shader.empty() && !average_shader.empty()) {
            std::cerr << "[GPU] All shaders loaded successfully" << std::endl;
        } else {
            std::cerr << "[GPU] Some shaders failed to load" << std::endl;
        }
    }
    
    std::string load_shader_file(const std::string& path) {
        std::ifstream file(path);
        if (!file) {
            std::cerr << "[GPU] Could not load shader: " << path << std::endl;
            return "";
        }
        
        std::string content((std::istreambuf_iterator<char>(file)),
                           std::istreambuf_iterator<char>());
        
        std::cerr << "[GPU] Loaded shader: " << path << " (" << content.size() << " bytes)" << std::endl;
        return content;
    }
    
    ~Impl() {
        if (gpu_available) {
            std::cerr << "[GPU] Cleaning up wasi:webgpu resources..." << std::endl;
            // Resource cleanup would happen here
        }
    }
};

// ═══════════════════════════════════════════════════════════════════════
// Public Interface Implementation
// ═══════════════════════════════════════════════════════════════════════

GpuExecutor::GpuExecutor()
    : impl_(std::make_unique<Impl>())
    , available_(impl_->gpu_available)
{
    std::cout << "[GPU] GPU Executor created" << std::endl;
    if (available_) {
        std::cout << "[GPU] GPU acceleration available via wasi:webgpu" << std::endl;
    } else {
        std::cout << "[GPU] Using CPU fallback" << std::endl;
    }
}

GpuExecutor::~GpuExecutor() = default;

bool GpuExecutor::is_available() const {
    return available_;
}

std::vector<uint32_t> GpuExecutor::bootstrap_sample(size_t n_samples, uint32_t seed) {
    std::cout << "[GPU] bootstrap_sample (n_samples=" << n_samples 
              << ", seed=" << seed << ")" << std::endl;
    
    if (!available_) {
        std::cerr << "[GPU] GPU not available, using CPU bootstrap" << std::endl;
        
        // CPU fallback
        std::vector<uint32_t> indices;
        indices.reserve(n_samples);
        
        // Simple XORshift PRNG (matching WGSL shader)
        auto xorshift = [](uint32_t x) -> uint32_t {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            return x;
        };
        
        for (size_t i = 0; i < n_samples; ++i) {
            uint32_t rng_state = seed + i * 747796405u + 2891336453u;
            rng_state = xorshift(rng_state);
            rng_state = xorshift(rng_state);
            uint32_t idx = rng_state % n_samples;
            indices.push_back(idx);
        }
        
        return indices;
    }
    
    // GPU implementation using wasi:webgpu
    std::cout << "[GPU] Executing bootstrap_sample via wasi:webgpu..." << std::endl;
    
        // Create output buffer
        uint64_t buffer_size = n_samples * sizeof(uint32_t);
        uint32_t buffer_id = __wasi_webgpu_create_buffer(
            impl_->device_id,
            buffer_size,
            0x0084 // STORAGE | COPY_SRC
        );

        std::cout << "[GPU] Created buffer ID: " << buffer_id << " (size=" << buffer_size << ")" << std::endl;

        // TODO: Create shader module, pipeline, dispatch compute
        // For now, fall back to CPU
        std::cerr << "[GPU] Full GPU pipeline not yet implemented, using CPU" << std::endl;

        // CPU fallback
        std::vector<uint32_t> indices;
        indices.reserve(n_samples);

        auto xorshift = [](uint32_t x) -> uint32_t {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            return x;
        };

        for (size_t i = 0; i < n_samples; ++i) {
            uint32_t rng_state = seed + i * 747796405u + 2891336453u;
            rng_state = xorshift(rng_state);
            rng_state = xorshift(rng_state);
            uint32_t idx = rng_state % n_samples;
            indices.push_back(idx);
        }

        return indices;
}

std::pair<float, float> GpuExecutor::find_best_split(
    const std::vector<float>& data,
    const std::vector<float>& labels,
    const std::vector<uint32_t>& indices,
    size_t n_features,
    size_t feature_idx
) {
    // CPU fallback - simplified version
    std::vector<float> values;
    values.reserve(indices.size());
    for (uint32_t idx : indices) {
        values.push_back(data[idx * n_features + feature_idx]);
    }
    
    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end()), values.end());
    
    if (values.size() < 2) {
        return {0.0f, std::numeric_limits<float>::infinity()};
    }
    
    float best_threshold = 0.0f;
    float best_score = std::numeric_limits<float>::infinity();
    
    for (size_t i = 0; i + 1 < values.size(); ++i) {
        float threshold = (values[i] + values[i + 1]) / 2.0f;
        
        float left_sum = 0.0f, right_sum = 0.0f;
        size_t left_count = 0, right_count = 0;
        
        for (uint32_t idx : indices) {
            float val = data[idx * n_features + feature_idx];
            float label = labels[idx];
            
            if (val <= threshold) {
                left_sum += label;
                left_count++;
            } else {
                right_sum += label;
                right_count++;
            }
        }
        
        if (left_count == 0 || right_count == 0) continue;
        
        float left_mean = left_sum / left_count;
        float right_mean = right_sum / right_count;
        
        float mse = 0.0f;
        for (uint32_t idx : indices) {
            float val = data[idx * n_features + feature_idx];
            float label = labels[idx];
            float mean = (val <= threshold) ? left_mean : right_mean;
            float diff = label - mean;
            mse += diff * diff;
        }
        
        if (mse < best_score) {
            best_score = mse;
            best_threshold = threshold;
        }
    }
    
    return {best_threshold, best_score};
}

std::vector<float> GpuExecutor::average_predictions(
    const std::vector<float>& tree_predictions,
    size_t n_samples,
    size_t n_trees
) {
    std::cout << "[GPU] average_predictions (n_samples=" << n_samples 
              << ", n_trees=" << n_trees << ")" << std::endl;
    
    // CPU fallback for now
    std::vector<float> result;
    result.reserve(n_samples);
    
    for (size_t i = 0; i < n_samples; ++i) {
        float sum = 0.0f;
        for (size_t j = 0; j < n_trees; ++j) {
            sum += tree_predictions[i * n_trees + j];
        }
        result.push_back(sum / static_cast<float>(n_trees));
    }
    
    return result;
}

std::vector<float> GpuExecutor::predict(
    const RandomForest& forest,
    const std::vector<float>& input_data,
    size_t n_features
) {
    size_t n_samples = input_data.size() / n_features;
    
    std::cout << "[GPU] predict (n_samples=" << n_samples << ")" << std::endl;
    
    // Get predictions from all trees
    std::vector<float> tree_predictions;
    tree_predictions.reserve(n_samples * forest.n_trees());
    
    for (size_t i = 0; i < n_samples; ++i) {
        const float* sample = &input_data[i * n_features];
        auto preds = forest.get_tree_predictions(sample);
        tree_predictions.insert(tree_predictions.end(), preds.begin(), preds.end());
    }
    
    // Average predictions
    return average_predictions(tree_predictions, n_samples, forest.n_trees());
}

bool GpuExecutor::compile_shader(const std::string& shader_code, 
                                 const std::string& entry_point) {
    std::cout << "[GPU] Compiling shader: " << entry_point << std::endl;
    
    if (!available_) {
        return false;
    }
    
    uint32_t shader_id = __wasi_webgpu_create_shader_module(
        impl_->device_id,
        shader_code.c_str(),
        static_cast<uint32_t>(shader_code.size())
    );

    if (shader_id == 0) {
        std::cerr << "[GPU] Failed to compile shader" << std::endl;
        return false;
    }

    std::cout << "[GPU] Shader module ID: " << shader_id << std::endl;
    return true;
}

bool GpuExecutor::execute_compute(size_t workgroup_count_x, 
                                  size_t workgroup_count_y,
                                  size_t workgroup_count_z) {
    std::cout << "[GPU] Executing compute: " 
              << workgroup_count_x << "x" 
              << workgroup_count_y << "x" 
              << workgroup_count_z << std::endl;
    
    if (!available_) {
        return false;
    }
    
    // TODO: Full compute pipeline implementation
    return true;
}

} // namespace ml
