/**
 * @file gpu_executor.hpp
 * @brief GPU compute acceleration using wasi:gpu interface
 * 
 * This module provides GPU acceleration through the wasi:gpu host functions.
 * The actual GPU backend (WebGPU/Vulkan or CUDA) is determined by the host runtime.
 */

#ifndef GPU_EXECUTOR_HPP
#define GPU_EXECUTOR_HPP

#include <vector>
#include <string>
#include <memory>
#include <cstdint>

namespace ml {

// Forward declarations
class RandomForest;

/**
 * @struct DeviceInfo
 * @brief GPU device information from host
 */
struct DeviceInfo {
    std::string name;
    std::string backend;
    uint64_t total_memory;
    bool is_hardware;
    std::string compute_capability;
};

/**
 * @class GpuExecutor
 * @brief GPU compute executor using wasi:gpu host functions
 * 
 * This class wraps the wasi:gpu interface to provide high-level
 * GPU operations for ML workloads. The actual GPU implementation
 * (CUDA, WebGPU, etc.) is provided by the host runtime.
 */
class GpuExecutor {
public:
    /**
     * @brief Initialize GPU executor and query device info
     */
    GpuExecutor();
    
    /**
     * @brief Destructor - cleanup GPU resources
     */
    ~GpuExecutor();
    
    /**
     * @brief Check if GPU is available
     */
    bool is_available() const { return available_; }
    
    /**
     * @brief Check if using hardware GPU (not software renderer)
     */
    bool is_hardware_gpu() const { return device_info_.is_hardware; }
    
    /**
     * @brief Get device information
     */
    const DeviceInfo& device_info() const { return device_info_; }
    
    /**
     * @brief Get device name
     */
    const std::string& device_name() const { return device_info_.name; }
    
    /**
     * @brief Get backend type ("cuda", "vulkan", "webgpu")
     */
    const std::string& backend() const { return device_info_.backend; }
    
    // ═══════════════════════════════════════════════════════════════════
    // Buffer Management
    // ═══════════════════════════════════════════════════════════════════
    
    /**
     * @brief Create a GPU buffer
     * @param size Buffer size in bytes
     * @param usage Usage flags (STORAGE, UNIFORM, etc.)
     * @return Buffer ID or 0 on failure
     */
    uint32_t create_buffer(uint64_t size, uint32_t usage);
    
    /**
     * @brief Write data to GPU buffer
     */
    bool write_buffer(uint32_t buffer_id, uint64_t offset, const void* data, size_t size);
    
    /**
     * @brief Read data from GPU buffer
     */
    bool read_buffer(uint32_t buffer_id, uint64_t offset, void* data, size_t size);
    
    /**
     * @brief Destroy GPU buffer
     */
    void destroy_buffer(uint32_t buffer_id);
    
    /**
     * @brief Synchronize GPU operations
     */
    void sync();
    
    // ═══════════════════════════════════════════════════════════════════
    // ML Kernel Operations
    // ═══════════════════════════════════════════════════════════════════
    
    /**
     * @brief Bootstrap sampling on GPU
     * @param n_samples Number of samples to generate
     * @param seed Random seed
     * @param max_index Maximum index value (dataset size)
     * @return Sampled indices (with replacement)
     */
    std::vector<uint32_t> bootstrap_sample(size_t n_samples, uint32_t seed, uint32_t max_index);
    
    /**
     * @brief Find best split for a feature on GPU
     * @param indices Sample indices to consider (buffer on GPU)
     * @param feature_idx Feature to evaluate
     * @param thresholds Candidate threshold values
     * @param data_buffer Pre-uploaded data buffer ID
     * @param labels_buffer Pre-uploaded labels buffer ID
     * @param n_features Number of features
     * @return Pair of (best_threshold, best_score)
     */
    std::pair<float, float> find_best_split(
        const std::vector<uint32_t>& indices,
        size_t feature_idx,
        const std::vector<float>& thresholds,
        uint32_t data_buffer,
        uint32_t labels_buffer,
        size_t n_features
    );
    
    /**
     * @brief Average tree predictions on GPU
     * @param tree_predictions Predictions from all trees (flat: n_samples * n_trees)
     * @param n_samples Number of samples
     * @param n_trees Number of trees
     * @return Averaged predictions for each sample
     */
    std::vector<float> average_predictions(
        const std::vector<float>& tree_predictions,
        size_t n_samples,
        size_t n_trees
    );
    
    /**
     * @brief Matrix multiplication on GPU
     * @param a Matrix A (m x k)
     * @param b Matrix B (k x n)
     * @param m Rows in A
     * @param k Columns in A / Rows in B
     * @param n Columns in B
     * @return Result matrix C (m x n)
     */
    std::vector<float> matmul(
        const std::vector<float>& a,
        const std::vector<float>& b,
        size_t m, size_t k, size_t n
    );

private:
    bool available_;
    DeviceInfo device_info_;
    
    // CPU fallback implementations
    std::vector<uint32_t> bootstrap_sample_cpu(size_t n_samples, uint32_t seed, uint32_t max_index);
    std::vector<float> average_predictions_cpu(
        const std::vector<float>& tree_predictions,
        size_t n_samples,
        size_t n_trees
    );
};

/**
 * @class GpuTrainer
 * @brief GPU-accelerated training for RandomForest
 */
class GpuTrainer {
public:
    GpuTrainer();
    ~GpuTrainer();
    
    /**
     * @brief Check if GPU training is available
     */
    bool is_available() const { return executor_.is_available(); }
    
    /**
     * @brief Upload training data to GPU (call once)
     */
    bool upload_training_data(
        const std::vector<float>& data,
        const std::vector<float>& labels,
        size_t n_samples,
        size_t n_features
    );
    
    /**
     * @brief Generate bootstrap sample indices
     */
    std::vector<uint32_t> bootstrap_sample(size_t n_samples, uint32_t seed);
    
    /**
     * @brief Find best split for a feature
     */
    std::pair<float, float> find_best_split(
        const std::vector<uint32_t>& indices,
        size_t feature_idx,
        const std::vector<float>& thresholds
    );
    
    /**
     * @brief Cleanup GPU resources
     */
    void cleanup();

private:
    GpuExecutor executor_;
    uint32_t data_buffer_;
    uint32_t labels_buffer_;
    size_t n_samples_;
    size_t n_features_;
};

/**
 * @class GpuPredictor
 * @brief GPU-accelerated prediction
 */
class GpuPredictor {
public:
    GpuPredictor();
    
    /**
     * @brief Check if GPU prediction is available
     */
    bool is_available() const { return executor_.is_available(); }
    
    /**
     * @brief Average tree predictions
     */
    std::vector<float> average_predictions(
        const std::vector<float>& tree_predictions,
        size_t n_samples,
        size_t n_trees
    );

private:
    GpuExecutor executor_;
};

} // namespace ml

#endif // GPU_EXECUTOR_HPP
