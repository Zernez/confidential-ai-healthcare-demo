/**
 * @file random_forest.hpp
 * @brief RandomForest implementation with GPU acceleration via wasi:gpu
 * 
 * Implements RandomForest regressor with decision trees using bagging.
 * Supports both CPU and GPU training/inference via wasi:gpu host functions.
 */

#ifndef RANDOM_FOREST_HPP
#define RANDOM_FOREST_HPP

#include <vector>
#include <memory>
#include <random>
#include <string>
#include "dataset.hpp"
#include "gpu_executor.hpp"

namespace ml {

// Forward declarations
class GpuExecutor;
class GpuTrainer;
class GpuPredictor;

/**
 * @enum NodeType
 * @brief Type of tree node
 */
enum class NodeType {
    LEAF,      ///< Leaf node (prediction value)
    INTERNAL   ///< Internal node (split decision)
};

/**
 * @struct TreeNode
 * @brief Node in a decision tree
 */
struct TreeNode {
    NodeType type;
    
    // For leaf nodes
    float value;
    
    // For internal nodes
    size_t feature_idx;
    float threshold;
    std::unique_ptr<TreeNode> left;
    std::unique_ptr<TreeNode> right;
    
    /**
     * @brief Create a leaf node
     */
    static std::unique_ptr<TreeNode> make_leaf(float value);
    
    /**
     * @brief Create an internal node
     */
    static std::unique_ptr<TreeNode> make_internal(
        size_t feature_idx, 
        float threshold,
        std::unique_ptr<TreeNode> left,
        std::unique_ptr<TreeNode> right
    );
    
    /**
     * @brief Predict for a single sample
     */
    float predict(const float* sample) const;
};

/**
 * @class DecisionTree
 * @brief Single decision tree for regression
 */
class DecisionTree {
public:
    /**
     * @brief Construct tree with max depth
     */
    explicit DecisionTree(size_t max_depth);
    
    /**
     * @brief Train tree on CPU
     */
    void train_cpu(const std::vector<float>& data,
                   const std::vector<float>& labels,
                   size_t n_samples,
                   size_t n_features,
                   std::mt19937& rng);
    
    /**
     * @brief Train tree with GPU acceleration via wasi:gpu
     */
    void train_with_gpu(const std::vector<float>& data,
                        const std::vector<float>& labels,
                        const std::vector<uint32_t>& bootstrap_indices,
                        size_t n_features,
                        GpuTrainer& gpu_trainer,
                        std::mt19937& rng);
    
    /**
     * @brief Predict for a single sample
     */
    float predict(const float* sample) const;
    
    /**
     * @brief Serialize tree to JSON
     */
    std::string to_json() const;
    
    /**
     * @brief Deserialize tree from JSON
     */
    static DecisionTree from_json(const std::string& json_str);

private:
    size_t max_depth_;
    std::unique_ptr<TreeNode> root_;
    
    /**
     * @brief Recursively build tree (CPU version)
     */
    std::unique_ptr<TreeNode> build_tree_cpu(
        const std::vector<float>& data,
        const std::vector<float>& labels,
        const std::vector<size_t>& indices,
        size_t n_features,
        size_t depth,
        std::mt19937& rng
    );
    
    /**
     * @brief Recursively build tree with GPU split finding
     */
    std::unique_ptr<TreeNode> build_tree_gpu(
        const std::vector<float>& data,
        const std::vector<float>& labels,
        const std::vector<size_t>& indices,
        size_t n_features,
        size_t depth,
        GpuTrainer& gpu_trainer,
        std::mt19937& rng
    );
    
    /**
     * @brief Find best split for a node (CPU)
     */
    struct SplitInfo {
        size_t feature_idx;
        float threshold;
        float score;
    };
    
    SplitInfo find_best_split_cpu(
        const std::vector<float>& data,
        const std::vector<float>& labels,
        const std::vector<size_t>& indices,
        size_t n_features,
        std::mt19937& rng
    );
    
    /**
     * @brief Find best split using GPU
     */
    SplitInfo find_best_split_gpu(
        const std::vector<float>& data,
        const std::vector<float>& labels,
        const std::vector<size_t>& indices,
        size_t n_features,
        GpuTrainer& gpu_trainer,
        std::mt19937& rng
    );
    
    /**
     * @brief Split indices based on feature and threshold
     */
    std::pair<std::vector<size_t>, std::vector<size_t>> split_data(
        const std::vector<float>& data,
        const std::vector<size_t>& indices,
        size_t n_features,
        size_t feature_idx,
        float threshold
    );
    
    /**
     * @brief Calculate MSE for a set of samples
     */
    float calculate_mse(
        const std::vector<float>& labels,
        const std::vector<size_t>& indices
    );
};

/**
 * @class RandomForest
 * @brief Ensemble of decision trees
 */
class RandomForest {
public:
    /**
     * @brief Construct RandomForest
     * @param n_estimators Number of trees
     * @param max_depth Maximum tree depth
     */
    RandomForest(size_t n_estimators, size_t max_depth);
    
    /**
     * @brief Train forest on CPU
     * @param dataset Training dataset
     */
    void train_cpu(const Dataset& dataset);
    
    /**
     * @brief Train forest with GPU acceleration via wasi:gpu
     * @param dataset Training dataset
     * @param gpu_trainer GPU trainer with pre-uploaded data
     */
    void train_with_gpu(const Dataset& dataset, GpuTrainer& gpu_trainer);
    
    /**
     * @brief Train forest with GPU and progress callback
     * @param dataset Training dataset
     * @param gpu_trainer GPU trainer with pre-uploaded data
     * @param progress_callback Called with (trees_trained, total_trees)
     */
    template<typename Callback>
    void train_with_gpu(const Dataset& dataset, GpuTrainer& gpu_trainer, Callback progress_callback);
    
    /**
     * @brief Predict on CPU
     */
    std::vector<float> predict_cpu(const std::vector<float>& data,
                                    size_t n_samples,
                                    size_t n_features) const;
    
    /**
     * @brief Predict with GPU acceleration via wasi:gpu
     */
    std::vector<float> predict_with_gpu(const std::vector<float>& data,
                                         size_t n_samples,
                                         size_t n_features,
                                         GpuPredictor& predictor);
    
    /**
     * @brief Get tree predictions for GPU processing
     */
    std::vector<float> get_tree_predictions(const float* sample) const;
    
    /**
     * @brief Get number of trees
     */
    size_t n_trees() const { return trees_.size(); }
    
    /**
     * @brief Serialize to JSON
     */
    std::string to_json() const;
    
    /**
     * @brief Deserialize from JSON
     */
    static RandomForest from_json(const std::string& json_str);

private:
    size_t n_estimators_;
    size_t max_depth_;
    std::vector<DecisionTree> trees_;
};

// Template implementation for train_with_gpu with callback
template<typename Callback>
void RandomForest::train_with_gpu(const Dataset& dataset, GpuTrainer& gpu_trainer, Callback progress_callback) {
    if (!gpu_trainer.is_available()) {
        train_cpu(dataset);
        return;
    }
    
    std::random_device rd;
    std::mt19937 rng(rd());
    
    for (size_t i = 0; i < n_estimators_; ++i) {
        // Bootstrap sample on GPU
        uint32_t seed = rng();
        std::vector<uint32_t> bootstrap_indices = gpu_trainer.bootstrap_sample(
            dataset.size(), seed
        );
        
        // Extract bootstrapped data
        std::vector<float> sampled_data;
        std::vector<float> sampled_labels;
        
        sampled_data.reserve(dataset.size() * dataset.n_features());
        sampled_labels.reserve(dataset.size());
        
        for (uint32_t idx : bootstrap_indices) {
            const float* sample = dataset.get_sample(idx);
            sampled_data.insert(sampled_data.end(), sample, sample + dataset.n_features());
            sampled_labels.push_back(dataset.get_label(idx));
        }
        
        // Train tree with GPU-accelerated split finding
        DecisionTree tree(max_depth_);
        tree.train_with_gpu(sampled_data, sampled_labels, bootstrap_indices,
                            dataset.n_features(), gpu_trainer, rng);
        
        trees_.push_back(std::move(tree));
        
        // Call progress callback
        progress_callback(i + 1, n_estimators_);
    }
}

} // namespace ml

#endif // RANDOM_FOREST_HPP
