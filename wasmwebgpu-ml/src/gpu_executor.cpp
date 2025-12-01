/**
 * @file gpu_executor.cpp  
 * @brief Implementation of GPU executor using wasi:gpu host functions
 * 
 * This implementation uses the wasi:gpu interface which is backend-agnostic.
 * The host runtime can implement it using WebGPU/Vulkan or CUDA.
 */

#include "gpu_executor.hpp"
#include "wasi_gpu.h"
#include <iostream>
#include <cstring>
#include <algorithm>
#include <cmath>

namespace ml {

// ═══════════════════════════════════════════════════════════════════════════
// GpuExecutor Implementation
// ═══════════════════════════════════════════════════════════════════════════

GpuExecutor::GpuExecutor() : available_(false) {
    // Query device info from host using the wrapper function
    wasi_gpu_device_info_t info = {0};
    wasi_gpu_get_device_info(&info);
    
    device_info_.name = info.name;
    device_info_.backend = info.backend;
    device_info_.total_memory = info.total_memory;
    device_info_.is_hardware = (info.is_hardware != 0);
    device_info_.compute_capability = info.compute_capability;
    
    // Check if we got valid device info
    if (!device_info_.name.empty() && device_info_.total_memory > 0) {
        available_ = true;
    }
}

GpuExecutor::~GpuExecutor() {
}

uint32_t GpuExecutor::create_buffer(uint64_t size, uint32_t usage) {
    if (!available_) return 0;
    
    wasi_gpu_buffer_id buffer_id = 0;
    wasi_gpu_error err = wasi_gpu_buffer_create(size, usage, &buffer_id);
    
    if (err != WASI_GPU_SUCCESS) {
        return 0;
    }
    
    return buffer_id;
}

bool GpuExecutor::write_buffer(uint32_t buffer_id, uint64_t offset, const void* data, size_t size) {
    if (!available_ || buffer_id == 0) return false;
    
    wasi_gpu_error err = wasi_gpu_buffer_write(
        buffer_id, 
        offset, 
        static_cast<const uint8_t*>(data), 
        static_cast<uint32_t>(size)
    );
    
    return err == WASI_GPU_SUCCESS;
}

bool GpuExecutor::read_buffer(uint32_t buffer_id, uint64_t offset, void* data, size_t size) {
    if (!available_ || buffer_id == 0) return false;
    
    uint32_t out_len = 0;
    wasi_gpu_error err = wasi_gpu_buffer_read(
        buffer_id,
        offset,
        static_cast<uint32_t>(size),
        static_cast<uint8_t*>(data),
        &out_len
    );
    
    return err == WASI_GPU_SUCCESS && out_len == size;
}

void GpuExecutor::destroy_buffer(uint32_t buffer_id) {
    if (available_ && buffer_id != 0) {
        wasi_gpu_buffer_destroy(buffer_id);
    }
}

void GpuExecutor::sync() {
    if (available_) {
        wasi_gpu_sync();
    }
}

std::vector<uint32_t> GpuExecutor::bootstrap_sample(size_t n_samples, uint32_t seed, uint32_t max_index) {
    if (!available_) {
        return bootstrap_sample_cpu(n_samples, seed, max_index);
    }
    
    // Create output buffer
    uint64_t output_size = n_samples * sizeof(uint32_t);
    uint32_t output_buffer = create_buffer(
        output_size, 
        WASI_GPU_BUFFER_USAGE_STORAGE | WASI_GPU_BUFFER_USAGE_COPY_SRC
    );
    
    if (output_buffer == 0) {
        return bootstrap_sample_cpu(n_samples, seed, max_index);
    }
    
    // Set up parameters
    wasi_gpu_bootstrap_params_t params;
    params.n_samples = static_cast<uint32_t>(n_samples);
    params.seed = seed;
    params.max_index = max_index;
    
    // Call kernel
    wasi_gpu_error err = wasi_gpu_kernel_bootstrap_sample(&params, output_buffer);
    
    if (err != WASI_GPU_SUCCESS) {
        destroy_buffer(output_buffer);
        return bootstrap_sample_cpu(n_samples, seed, max_index);
    }
    
    // Read results
    std::vector<uint32_t> indices(n_samples);
    if (!read_buffer(output_buffer, 0, indices.data(), output_size)) {
        destroy_buffer(output_buffer);
        return bootstrap_sample_cpu(n_samples, seed, max_index);
    }
    
    // Cleanup
    destroy_buffer(output_buffer);
    
    return indices;
}

std::pair<float, float> GpuExecutor::find_best_split(
    const std::vector<uint32_t>& indices,
    size_t feature_idx,
    const std::vector<float>& thresholds,
    uint32_t data_buffer,
    uint32_t labels_buffer,
    size_t n_features
) {
    if (!available_ || thresholds.empty()) {
        return {0.0f, std::numeric_limits<float>::infinity()};
    }
    
    size_t n_thresholds = thresholds.size();
    
    // Create indices buffer
    uint64_t indices_size = indices.size() * sizeof(uint32_t);
    uint32_t indices_buffer = create_buffer(
        indices_size,
        WASI_GPU_BUFFER_USAGE_STORAGE | WASI_GPU_BUFFER_USAGE_COPY_DST
    );
    if (indices_buffer == 0) return {0.0f, std::numeric_limits<float>::infinity()};
    write_buffer(indices_buffer, 0, indices.data(), indices_size);
    
    // Create thresholds buffer
    uint64_t thresholds_size = n_thresholds * sizeof(float);
    uint32_t thresholds_buffer = create_buffer(
        thresholds_size,
        WASI_GPU_BUFFER_USAGE_STORAGE | WASI_GPU_BUFFER_USAGE_COPY_DST
    );
    if (thresholds_buffer == 0) {
        destroy_buffer(indices_buffer);
        return {0.0f, std::numeric_limits<float>::infinity()};
    }
    write_buffer(thresholds_buffer, 0, thresholds.data(), thresholds_size);
    
    // Create output scores buffer
    uint64_t scores_size = n_thresholds * sizeof(float);
    uint32_t scores_buffer = create_buffer(
        scores_size,
        WASI_GPU_BUFFER_USAGE_STORAGE | WASI_GPU_BUFFER_USAGE_COPY_SRC
    );
    if (scores_buffer == 0) {
        destroy_buffer(indices_buffer);
        destroy_buffer(thresholds_buffer);
        return {0.0f, std::numeric_limits<float>::infinity()};
    }
    
    // Set up parameters
    wasi_gpu_find_split_params_t params;
    params.n_samples = static_cast<uint32_t>(indices.size());
    params.n_features = static_cast<uint32_t>(n_features);
    params.feature_idx = static_cast<uint32_t>(feature_idx);
    params.n_thresholds = static_cast<uint32_t>(n_thresholds);
    
    // Call kernel
    wasi_gpu_error err = wasi_gpu_kernel_find_split(
        &params,
        data_buffer,
        labels_buffer,
        indices_buffer,
        thresholds_buffer,
        scores_buffer
    );
    
    float best_threshold = 0.0f;
    float best_score = std::numeric_limits<float>::infinity();
    
    if (err == WASI_GPU_SUCCESS) {
        // Read scores
        std::vector<float> scores(n_thresholds);
        if (read_buffer(scores_buffer, 0, scores.data(), scores_size)) {
            // Find best
            for (size_t i = 0; i < n_thresholds; ++i) {
                if (scores[i] < best_score) {
                    best_score = scores[i];
                    best_threshold = thresholds[i];
                }
            }
        }
    }
    
    // Cleanup
    destroy_buffer(indices_buffer);
    destroy_buffer(thresholds_buffer);
    destroy_buffer(scores_buffer);
    
    return {best_threshold, best_score};
}

std::vector<float> GpuExecutor::average_predictions(
    const std::vector<float>& tree_predictions,
    size_t n_samples,
    size_t n_trees
) {
    if (!available_) {
        return average_predictions_cpu(tree_predictions, n_samples, n_trees);
    }
    
    // Create input buffer
    uint64_t input_size = tree_predictions.size() * sizeof(float);
    uint32_t input_buffer = create_buffer(
        input_size,
        WASI_GPU_BUFFER_USAGE_STORAGE | WASI_GPU_BUFFER_USAGE_COPY_DST
    );
    if (input_buffer == 0) {
        return average_predictions_cpu(tree_predictions, n_samples, n_trees);
    }
    write_buffer(input_buffer, 0, tree_predictions.data(), input_size);
    
    // Create output buffer
    uint64_t output_size = n_samples * sizeof(float);
    uint32_t output_buffer = create_buffer(
        output_size,
        WASI_GPU_BUFFER_USAGE_STORAGE | WASI_GPU_BUFFER_USAGE_COPY_SRC
    );
    if (output_buffer == 0) {
        destroy_buffer(input_buffer);
        return average_predictions_cpu(tree_predictions, n_samples, n_trees);
    }
    
    // Set up parameters
    wasi_gpu_average_params_t params;
    params.n_trees = static_cast<uint32_t>(n_trees);
    params.n_samples = static_cast<uint32_t>(n_samples);
    
    // Call kernel
    wasi_gpu_error err = wasi_gpu_kernel_average(&params, input_buffer, output_buffer);
    
    std::vector<float> result;
    
    if (err == WASI_GPU_SUCCESS) {
        result.resize(n_samples);
        if (!read_buffer(output_buffer, 0, result.data(), output_size)) {
            result = average_predictions_cpu(tree_predictions, n_samples, n_trees);
        }
    } else {
        result = average_predictions_cpu(tree_predictions, n_samples, n_trees);
    }
    
    // Cleanup
    destroy_buffer(input_buffer);
    destroy_buffer(output_buffer);
    
    return result;
}

std::vector<float> GpuExecutor::matmul(
    const std::vector<float>& a,
    const std::vector<float>& b,
    size_t m, size_t k, size_t n
) {
    std::vector<float> c(m * n, 0.0f);
    
    if (!available_) {
        // CPU fallback
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n; ++j) {
                float sum = 0.0f;
                for (size_t l = 0; l < k; ++l) {
                    sum += a[i * k + l] * b[l * n + j];
                }
                c[i * n + j] = sum;
            }
        }
        return c;
    }
    
    // Create buffers
    uint32_t a_buffer = create_buffer(a.size() * sizeof(float), 
        WASI_GPU_BUFFER_USAGE_STORAGE | WASI_GPU_BUFFER_USAGE_COPY_DST);
    uint32_t b_buffer = create_buffer(b.size() * sizeof(float),
        WASI_GPU_BUFFER_USAGE_STORAGE | WASI_GPU_BUFFER_USAGE_COPY_DST);
    uint32_t c_buffer = create_buffer(c.size() * sizeof(float),
        WASI_GPU_BUFFER_USAGE_STORAGE | WASI_GPU_BUFFER_USAGE_COPY_SRC | WASI_GPU_BUFFER_USAGE_COPY_DST);
    
    if (a_buffer == 0 || b_buffer == 0 || c_buffer == 0) {
        destroy_buffer(a_buffer);
        destroy_buffer(b_buffer);
        destroy_buffer(c_buffer);
        // CPU fallback
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n; ++j) {
                float sum = 0.0f;
                for (size_t l = 0; l < k; ++l) {
                    sum += a[i * k + l] * b[l * n + j];
                }
                c[i * n + j] = sum;
            }
        }
        return c;
    }
    
    write_buffer(a_buffer, 0, a.data(), a.size() * sizeof(float));
    write_buffer(b_buffer, 0, b.data(), b.size() * sizeof(float));
    write_buffer(c_buffer, 0, c.data(), c.size() * sizeof(float)); // Zero init
    
    // Set up parameters
    wasi_gpu_matmul_params_t params;
    params.m = static_cast<uint32_t>(m);
    params.k = static_cast<uint32_t>(k);
    params.n = static_cast<uint32_t>(n);
    params.trans_a = 0;
    params.trans_b = 0;
    params.alpha = 1.0f;
    params.beta = 0.0f;
    
    // Call kernel
    wasi_gpu_error err = wasi_gpu_kernel_matmul(&params, a_buffer, b_buffer, c_buffer);
    
    if (err == WASI_GPU_SUCCESS) {
        read_buffer(c_buffer, 0, c.data(), c.size() * sizeof(float));
    } else {
        // CPU fallback
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n; ++j) {
                float sum = 0.0f;
                for (size_t l = 0; l < k; ++l) {
                    sum += a[i * k + l] * b[l * n + j];
                }
                c[i * n + j] = sum;
            }
        }
    }
    
    // Cleanup
    destroy_buffer(a_buffer);
    destroy_buffer(b_buffer);
    destroy_buffer(c_buffer);
    
    return c;
}

// CPU Fallback implementations
std::vector<uint32_t> GpuExecutor::bootstrap_sample_cpu(size_t n_samples, uint32_t seed, uint32_t max_index) {
    std::vector<uint32_t> indices;
    indices.reserve(n_samples);
    
    auto xorshift = [](uint32_t x) -> uint32_t {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        return x;
    };
    
    for (size_t i = 0; i < n_samples; ++i) {
        uint32_t rng_state = seed + static_cast<uint32_t>(i) * 747796405u + 2891336453u;
        rng_state = xorshift(rng_state);
        rng_state = xorshift(rng_state);
        uint32_t idx = rng_state % max_index;
        indices.push_back(idx);
    }
    
    return indices;
}

std::vector<float> GpuExecutor::average_predictions_cpu(
    const std::vector<float>& tree_predictions,
    size_t n_samples,
    size_t n_trees
) {
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

// ═══════════════════════════════════════════════════════════════════════════
// GpuTrainer Implementation
// ═══════════════════════════════════════════════════════════════════════════

GpuTrainer::GpuTrainer() 
    : data_buffer_(0), labels_buffer_(0), n_samples_(0), n_features_(0) {
}

GpuTrainer::~GpuTrainer() {
    cleanup();
}

bool GpuTrainer::upload_training_data(
    const std::vector<float>& data,
    const std::vector<float>& labels,
    size_t n_samples,
    size_t n_features
) {
    if (!executor_.is_available()) {
        std::cerr << "[GpuTrainer] GPU not available" << std::endl;
        return false;
    }
    
    n_samples_ = n_samples;
    n_features_ = n_features;
    
    // Create and upload data buffer
    uint64_t data_size = data.size() * sizeof(float);
    data_buffer_ = executor_.create_buffer(
        data_size,
        WASI_GPU_BUFFER_USAGE_STORAGE | WASI_GPU_BUFFER_USAGE_COPY_DST
    );
    if (data_buffer_ == 0) return false;
    executor_.write_buffer(data_buffer_, 0, data.data(), data_size);
    
    // Create and upload labels buffer
    uint64_t labels_size = labels.size() * sizeof(float);
    labels_buffer_ = executor_.create_buffer(
        labels_size,
        WASI_GPU_BUFFER_USAGE_STORAGE | WASI_GPU_BUFFER_USAGE_COPY_DST
    );
    if (labels_buffer_ == 0) {
        executor_.destroy_buffer(data_buffer_);
        data_buffer_ = 0;
        return false;
    }
    executor_.write_buffer(labels_buffer_, 0, labels.data(), labels_size);
    
    std::cerr << "[GpuTrainer] Uploaded " << n_samples << " samples x " 
              << n_features << " features to GPU" << std::endl;
    
    return true;
}

std::vector<uint32_t> GpuTrainer::bootstrap_sample(size_t n_samples, uint32_t seed) {
    return executor_.bootstrap_sample(n_samples, seed, static_cast<uint32_t>(n_samples_));
}

std::pair<float, float> GpuTrainer::find_best_split(
    const std::vector<uint32_t>& indices,
    size_t feature_idx,
    const std::vector<float>& thresholds
) {
    return executor_.find_best_split(
        indices, feature_idx, thresholds,
        data_buffer_, labels_buffer_, n_features_
    );
}

void GpuTrainer::cleanup() {
    if (data_buffer_ != 0) {
        executor_.destroy_buffer(data_buffer_);
        data_buffer_ = 0;
    }
    if (labels_buffer_ != 0) {
        executor_.destroy_buffer(labels_buffer_);
        labels_buffer_ = 0;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// GpuPredictor Implementation
// ═══════════════════════════════════════════════════════════════════════════

GpuPredictor::GpuPredictor() {}

std::vector<float> GpuPredictor::average_predictions(
    const std::vector<float>& tree_predictions,
    size_t n_samples,
    size_t n_trees
) {
    return executor_.average_predictions(tree_predictions, n_samples, n_trees);
}

} // namespace ml
