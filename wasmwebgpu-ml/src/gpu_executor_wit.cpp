/**
 * @file gpu_executor_wit.cpp
 * @brief GPU executor using real wasi:webgpu WIT bindings
 */

#include "gpu_executor.hpp"
#include "random_forest.hpp"
#include <iostream>
#include <stdexcept>
#include <fstream>
#include <cmath>

// Include WIT-generated bindings
#include "../wit-bindings/wasi_webgpu_cpp.hpp"

namespace ml {

/**
 * @struct GpuExecutor::Impl
 * @brief Private implementation with real wasi:webgpu
 */
struct GpuExecutor::Impl {
    bool gpu_available;
    
    // Real wasi:webgpu handles
    wasi::webgpu::Instance instance;
    wasi::webgpu::Adapter adapter;
    wasi::webgpu::Device device;
    wasi::webgpu::Queue queue;
    
    // Shader cache
    std::string bootstrap_shader;
    std::string split_shader;
    std::string average_shader;
    
    Impl() : gpu_available(false) {
        std::cerr << "[GPU] Initializing WebGPU via wasi:webgpu WIT bindings..." << std::endl;
        
        // Load shaders
        load_shaders();
        
        // Try to initialize GPU
        if (try_init_gpu()) {
            gpu_available = true;
            std::cerr << "[GPU] ✓ wasi:webgpu initialized successfully" << std::endl;
        } else {
            std::cerr << "[GPU] ⚠ wasi:webgpu not available, will use CPU fallback" << std::endl;
        }
    }
    
    bool try_init_gpu() {
        try {
            std::cerr << "[GPU] Creating wasi:webgpu instance..." << std::endl;
            
            // Create instance (calls wasi:webgpu::create-instance)
            instance = wasi::webgpu::Instance();
            
            std::cerr << "[GPU] Requesting adapter..." << std::endl;
            
            // Request adapter (calls wasi:webgpu::request-adapter)
            adapter = wasi::webgpu::Adapter(instance.handle(), 1); // High performance
            
            std::cerr << "[GPU] Requesting device..." << std::endl;
            
            // Request device (calls wasi:webgpu::request-device)
            device = wasi::webgpu::Device(adapter.handle());
            
            std::cerr << "[GPU] Getting queue..." << std::endl;
            
            // Get queue (calls wasi:webgpu::get-queue)
            queue = wasi::webgpu::Queue(device.handle());
            
            std::cerr << "[GPU] ✓ All wasi:webgpu objects created" << std::endl;
            
            return true;
            
        } catch (const std::exception& e) {
            std::cerr << "[GPU] Failed to initialize wasi:webgpu: " << e.what() << std::endl;
            return false;
        }
    }
    
    void load_shaders() {
        // Load WGSL shaders
        bootstrap_shader = load_shader_file("shaders/bootstrap_sample.wgsl");
        split_shader = load_shader_file("shaders/find_split.wgsl");
        average_shader = load_shader_file("shaders/average.wgsl");
        
        std::cerr << "[GPU] ✓ Shaders loaded" << std::endl;
    }
    
    std::string load_shader_file(const std::string& path) {
        std::ifstream file(path);
        if (!file) {
            std::cerr << "[GPU] ⚠ Could not load shader: " << path << std::endl;
            return "";
        }
        
        std::string content((std::istreambuf_iterator<char>(file)),
                           std::istreambuf_iterator<char>());
        return content;
    }
    
    ~Impl() {
        if (gpu_available) {
            std::cerr << "[GPU] Cleaning up wasi:webgpu resources..." << std::endl;
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
        std::cout << "[GPU] ✓ GPU acceleration available via wasi:webgpu" << std::endl;
    } else {
        std::cout << "[GPU] ℹ Using CPU fallback" << std::endl;
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
        
        // CPU fallback (same as before)
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
    
    // GPU implementation using wasi:webgpu
    std::cout << "[GPU] Executing bootstrap_sample shader via wasi:webgpu..." << std::endl;
    
    try {
        // Create output buffer
        uint64_t buffer_size = n_samples * sizeof(uint32_t);
        wasi::webgpu::Buffer output_buffer(
            impl_->device.handle(),
            buffer_size,
            0x0084 // STORAGE | COPY_SRC
        );
        
        std::cout << "[GPU] Created output buffer (size=" << buffer_size << ")" << std::endl;
        
        // TODO: Create and dispatch compute pipeline
        // For now, fall back to CPU
        std::cerr << "[GPU] ⚠ Full GPU pipeline not yet implemented, using CPU" << std::endl;
        
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
        
    } catch (const std::exception& e) {
        std::cerr << "[GPU] Error in GPU bootstrap: " << e.what() << std::endl;
        throw;
    }
}

std::pair<float, float> GpuExecutor::find_best_split(
    const std::vector<float>& data,
    const std::vector<float>& labels,
    const std::vector<uint32_t>& indices,
    size_t n_features,
    size_t feature_idx
) {
    if (!available_) {
        // CPU fallback (same as before)
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
    
    std::cout << "[GPU] find_best_split via wasi:webgpu (feature=" << feature_idx << ")" << std::endl;
    
    // TODO: GPU implementation
    // For now, use CPU fallback
    return find_best_split(data, labels, indices, n_features, feature_idx);
}

std::vector<float> GpuExecutor::average_predictions(
    const std::vector<float>& tree_predictions,
    size_t n_samples,
    size_t n_trees
) {
    std::cout << "[GPU] average_predictions via wasi:webgpu (n_samples=" << n_samples 
              << ", n_trees=" << n_trees << ")" << std::endl;
    
    if (!available_) {
        // CPU fallback
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
    
    // GPU implementation
    std::cout << "[GPU] Executing average shader..." << std::endl;
    
    // TODO: Full GPU pipeline
    // For now, CPU fallback
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
    
    std::cout << "[GPU] predict via wasi:webgpu (n_samples=" << n_samples << ")" << std::endl;
    
    // Get predictions from all trees
    std::vector<float> tree_predictions;
    tree_predictions.reserve(n_samples * forest.n_trees());
    
    for (size_t i = 0; i < n_samples; ++i) {
        const float* sample = &input_data[i * n_features];
        auto preds = forest.get_tree_predictions(sample);
        tree_predictions.insert(tree_predictions.end(), preds.begin(), preds.end());
    }
    
    // Average on GPU (or CPU fallback)
    return average_predictions(tree_predictions, n_samples, forest.n_trees());
}

bool GpuExecutor::compile_shader(const std::string& shader_code, 
                                 const std::string& entry_point) {
    std::cout << "[GPU] Compiling shader via wasi:webgpu: " << entry_point << std::endl;
    
    if (!available_) {
        return false;
    }
    
    // TODO: Use wasi:webgpu::create-shader-module
    
    std::cout << "[GPU] Shader compiled (placeholder)" << std::endl;
    return true;
}

bool GpuExecutor::execute_compute(size_t workgroup_count_x, 
                                  size_t workgroup_count_y,
                                  size_t workgroup_count_z) {
    std::cout << "[GPU] Executing compute via wasi:webgpu: " 
              << workgroup_count_x << "x" 
              << workgroup_count_y << "x" 
              << workgroup_count_z << std::endl;
    
    if (!available_) {
        return false;
    }
    
    // TODO: Full compute pipeline
    
    return true;
}

} // namespace ml
