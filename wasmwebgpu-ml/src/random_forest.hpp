/**
 * @file random_forest.hpp
 * @brief RandomForest implementation with GPU acceleration
 * 
 * Implements RandomForest regressor with decision trees using bagging.
 * Supports both CPU and GPU training/inference via WebGPU.
 */

#ifndef RANDOM_FOREST_HPP
#define RANDOM_FOREST_HPP

#include <vector>
#include <memory>
#include <random>
#include <string>
#include "dataset.hpp"

namespace ml {

// Forward declaration
class GpuExecutor;

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
     * @param data Flat array of features (row-major)
     * @param labels Target values
     * @param n_samples Number of samples
     * @param n_features Number of features
     * @param rng Random number generator
     */
    void train_cpu(const std::vector<float>& data,
                   const std::vector<float>& labels,
                   size_t n_samples,
                   size_t n_features,
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
    std::unique_ptr<TreeNode> build_tree(
        const std::vector<float>& data,
        const std::vector<float>& labels,
        const std::vector<size_t>& indices,
        size_t n_features,
        size_t depth,
        std::mt19937& rng
    );
    
    /**
     * @brief Find best split for a node
     */
    struct SplitInfo {
        size_t feature_idx;
        float threshold;
        float score;
    };
    
    SplitInfo find_best_split(
        const std::vector<float>& data,
        const std::vector<float>& labels,
        const std::vector<size_t>& indices,
        size_t n_features,
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
     * @brief Train forest with GPU acceleration
     * @param dataset Training dataset
     * @param gpu GPU executor
     */
    void train_gpu(const Dataset& dataset, GpuExecutor& gpu);
    
    /**
     * @brief Predict on CPU
     * @param data Flat array of features
     * @param n_samples Number of samples
     * @param n_features Number of features
     * @return Predictions for each sample
     */
    std::vector<float> predict_cpu(const std::vector<float>& data,
                                    size_t n_samples,
                                    size_t n_features) const;
    
    /**
     * @brief Predict with GPU acceleration
     * @param data Flat array of features
     * @param n_samples Number of samples
     * @param n_features Number of features
     * @param gpu GPU executor
     * @return Predictions for each sample
     */
    std::vector<float> predict_gpu(const std::vector<float>& data,
                                    size_t n_samples,
                                    size_t n_features,
                                    GpuExecutor& gpu);
    
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

} // namespace ml

#endif // RANDOM_FOREST_HPP
