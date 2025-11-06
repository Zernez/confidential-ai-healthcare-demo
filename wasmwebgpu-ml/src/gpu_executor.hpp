/**
 * @file gpu_executor.hpp
 * @brief GPU compute acceleration using WASI WebGPU
 * 
 * Handles parallel prediction and training operations using WebGPU compute shaders
 */

#ifndef GPU_EXECUTOR_HPP
#define GPU_EXECUTOR_HPP

#include <vector>
#include <string>
#include <memory>

namespace ml {

// Forward declarations
class RandomForest;

/**
 * @class GpuExecutor
 * @brief GPU compute executor for ML operations
 */
class GpuExecutor {
public:
    /**
     * @brief Initialize GPU device and queue
     */
    GpuExecutor();
    
    /**
     * @brief Destructor - cleanup GPU resources
     */
    ~GpuExecutor();
    
    /**
     * @brief Check if GPU is available and initialized
     */
    bool is_available() const;
    
    /**
     * @brief Bootstrap sampling on GPU
     * @param n_samples Number of samples to generate
     * @param seed Random seed
     * @return Sampled indices (with replacement)
     */
    std::vector<uint32_t> bootstrap_sample(size_t n_samples, uint32_t seed);
    
    /**
     * @brief Find best split for a feature on GPU
     * @param data Flat array of features
     * @param labels Target values
     * @param indices Sample indices to consider
     * @param n_features Number of features
     * @param feature_idx Feature to evaluate
     * @return Pair of (best_threshold, best_score)
     */
    std::pair<float, float> find_best_split(
        const std::vector<float>& data,
        const std::vector<float>& labels,
        const std::vector<uint32_t>& indices,
        size_t n_features,
        size_t feature_idx
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
     * @brief Run predictions on GPU
     * @param forest Trained RandomForest
     * @param input_data Flat array of features
     * @param n_features Number of features per sample
     * @return Predictions for each sample
     */
    std::vector<float> predict(
        const RandomForest& forest,
        const std::vector<float>& input_data,
        size_t n_features
    );

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    
    bool available_;
    
    /**
     * @brief Load and compile WGSL shader
     */
    bool compile_shader(const std::string& shader_code, const std::string& entry_point);
    
    /**
     * @brief Execute compute shader
     */
    bool execute_compute(size_t workgroup_count_x, 
                        size_t workgroup_count_y,
                        size_t workgroup_count_z);
};

} // namespace ml

#endif // GPU_EXECUTOR_HPP
