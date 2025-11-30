/**
 * @file random_forest.cpp
 * @brief Implementation of RandomForest algorithm with wasi:gpu support
 */

#include "random_forest.hpp"
#include "gpu_executor.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iostream>
#include <limits>
#include <cstring>

namespace ml {

// ═══════════════════════════════════════════════════════════════════════
// TreeNode Implementation
// ═══════════════════════════════════════════════════════════════════════

std::unique_ptr<TreeNode> TreeNode::make_leaf(float value) {
    auto node = std::make_unique<TreeNode>();
    node->type = NodeType::LEAF;
    node->value = value;
    return node;
}

std::unique_ptr<TreeNode> TreeNode::make_internal(
    size_t feature_idx, 
    float threshold,
    std::unique_ptr<TreeNode> left,
    std::unique_ptr<TreeNode> right
) {
    auto node = std::make_unique<TreeNode>();
    node->type = NodeType::INTERNAL;
    node->feature_idx = feature_idx;
    node->threshold = threshold;
    node->left = std::move(left);
    node->right = std::move(right);
    return node;
}

float TreeNode::predict(const float* sample) const {
    if (type == NodeType::LEAF) {
        return value;
    } else {
        if (sample[feature_idx] <= threshold) {
            return left->predict(sample);
        } else {
            return right->predict(sample);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// DecisionTree Implementation
// ═══════════════════════════════════════════════════════════════════════

DecisionTree::DecisionTree(size_t max_depth)
    : max_depth_(max_depth)
    , root_(nullptr) {}

void DecisionTree::train_cpu(const std::vector<float>& data,
                              const std::vector<float>& labels,
                              size_t n_samples,
                              size_t n_features,
                              std::mt19937& rng) {
    std::vector<size_t> indices(n_samples);
    std::iota(indices.begin(), indices.end(), 0);
    root_ = build_tree_cpu(data, labels, indices, n_features, 0, rng);
}

void DecisionTree::train_with_gpu(const std::vector<float>& data,
                                   const std::vector<float>& labels,
                                   const std::vector<uint32_t>& bootstrap_indices,
                                   size_t n_features,
                                   GpuTrainer& gpu_trainer,
                                   std::mt19937& rng) {
    std::vector<size_t> indices(bootstrap_indices.begin(), bootstrap_indices.end());
    root_ = build_tree_gpu(data, labels, indices, n_features, 0, gpu_trainer, rng);
}

std::unique_ptr<TreeNode> DecisionTree::build_tree_cpu(
    const std::vector<float>& data,
    const std::vector<float>& labels,
    const std::vector<size_t>& indices,
    size_t n_features,
    size_t depth,
    std::mt19937& rng
) {
    // Base cases
    if (depth >= max_depth_ || indices.size() < 2) {
        float sum = 0.0f;
        for (size_t idx : indices) {
            sum += labels[idx];
        }
        float mean = sum / indices.size();
        return TreeNode::make_leaf(mean);
    }
    
    // Find best split
    auto split = find_best_split_cpu(data, labels, indices, n_features, rng);
    
    // If no good split found, create leaf
    if (std::isinf(split.score)) {
        float sum = 0.0f;
        for (size_t idx : indices) {
            sum += labels[idx];
        }
        float mean = sum / indices.size();
        return TreeNode::make_leaf(mean);
    }
    
    // Split data
    auto [left_indices, right_indices] = split_data(
        data, indices, n_features, split.feature_idx, split.threshold
    );
    
    if (left_indices.empty() || right_indices.empty()) {
        float sum = 0.0f;
        for (size_t idx : indices) {
            sum += labels[idx];
        }
        float mean = sum / indices.size();
        return TreeNode::make_leaf(mean);
    }
    
    // Recursively build subtrees
    auto left = build_tree_cpu(data, labels, left_indices, n_features, depth + 1, rng);
    auto right = build_tree_cpu(data, labels, right_indices, n_features, depth + 1, rng);
    
    return TreeNode::make_internal(
        split.feature_idx,
        split.threshold,
        std::move(left),
        std::move(right)
    );
}

std::unique_ptr<TreeNode> DecisionTree::build_tree_gpu(
    const std::vector<float>& data,
    const std::vector<float>& labels,
    const std::vector<size_t>& indices,
    size_t n_features,
    size_t depth,
    GpuTrainer& gpu_trainer,
    std::mt19937& rng
) {
    // Base cases
    if (depth >= max_depth_ || indices.size() < 2) {
        float sum = 0.0f;
        for (size_t idx : indices) {
            sum += labels[idx];
        }
        float mean = sum / indices.size();
        return TreeNode::make_leaf(mean);
    }
    
    // Find best split using GPU
    auto split = find_best_split_gpu(data, labels, indices, n_features, gpu_trainer, rng);
    
    // If no good split found, create leaf
    if (std::isinf(split.score)) {
        float sum = 0.0f;
        for (size_t idx : indices) {
            sum += labels[idx];
        }
        float mean = sum / indices.size();
        return TreeNode::make_leaf(mean);
    }
    
    // Split data
    auto [left_indices, right_indices] = split_data(
        data, indices, n_features, split.feature_idx, split.threshold
    );
    
    if (left_indices.empty() || right_indices.empty()) {
        float sum = 0.0f;
        for (size_t idx : indices) {
            sum += labels[idx];
        }
        float mean = sum / indices.size();
        return TreeNode::make_leaf(mean);
    }
    
    // Recursively build subtrees
    auto left = build_tree_gpu(data, labels, left_indices, n_features, depth + 1, gpu_trainer, rng);
    auto right = build_tree_gpu(data, labels, right_indices, n_features, depth + 1, gpu_trainer, rng);
    
    return TreeNode::make_internal(
        split.feature_idx,
        split.threshold,
        std::move(left),
        std::move(right)
    );
}

DecisionTree::SplitInfo DecisionTree::find_best_split_cpu(
    const std::vector<float>& data,
    const std::vector<float>& labels,
    const std::vector<size_t>& indices,
    size_t n_features,
    std::mt19937& rng
) {
    SplitInfo best;
    best.score = std::numeric_limits<float>::infinity();
    best.feature_idx = 0;
    best.threshold = 0.0f;
    
    // Random feature selection (sqrt of total features)
    size_t n_features_to_try = static_cast<size_t>(std::ceil(std::sqrt(n_features)));
    
    std::vector<size_t> features(n_features);
    std::iota(features.begin(), features.end(), 0);
    std::shuffle(features.begin(), features.end(), rng);
    
    for (size_t i = 0; i < n_features_to_try; ++i) {
        size_t feature_idx = features[i];
        
        // Collect unique values for this feature
        std::vector<float> values;
        values.reserve(indices.size());
        for (size_t idx : indices) {
            values.push_back(data[idx * n_features + feature_idx]);
        }
        std::sort(values.begin(), values.end());
        values.erase(std::unique(values.begin(), values.end()), values.end());
        
        // Try different thresholds
        for (size_t j = 0; j + 1 < values.size(); ++j) {
            float threshold = (values[j] + values[j + 1]) / 2.0f;
            
            auto [left_indices, right_indices] = split_data(
                data, indices, n_features, feature_idx, threshold
            );
            
            if (left_indices.empty() || right_indices.empty()) {
                continue;
            }
            
            float score = calculate_mse(labels, left_indices) +
                         calculate_mse(labels, right_indices);
            
            if (score < best.score) {
                best.score = score;
                best.feature_idx = feature_idx;
                best.threshold = threshold;
            }
        }
    }
    
    return best;
}

DecisionTree::SplitInfo DecisionTree::find_best_split_gpu(
    const std::vector<float>& data,
    const std::vector<float>& labels,
    const std::vector<size_t>& indices,
    size_t n_features,
    GpuTrainer& gpu_trainer,
    std::mt19937& rng
) {
    SplitInfo best;
    best.score = std::numeric_limits<float>::infinity();
    best.feature_idx = 0;
    best.threshold = 0.0f;
    
    // Random feature selection
    size_t n_features_to_try = static_cast<size_t>(std::ceil(std::sqrt(n_features)));
    
    std::vector<size_t> features(n_features);
    std::iota(features.begin(), features.end(), 0);
    std::shuffle(features.begin(), features.end(), rng);
    
    // Convert indices to uint32_t
    std::vector<uint32_t> indices_u32(indices.begin(), indices.end());
    
    for (size_t i = 0; i < n_features_to_try; ++i) {
        size_t feature_idx = features[i];
        
        // Collect unique values and compute thresholds
        std::vector<float> values;
        values.reserve(indices.size());
        for (size_t idx : indices) {
            values.push_back(data[idx * n_features + feature_idx]);
        }
        std::sort(values.begin(), values.end());
        values.erase(std::unique(values.begin(), values.end()), values.end());
        
        if (values.size() < 2) continue;
        
        // Compute thresholds
        std::vector<float> thresholds;
        for (size_t j = 0; j + 1 < values.size(); ++j) {
            thresholds.push_back((values[j] + values[j + 1]) / 2.0f);
        }
        
        if (thresholds.empty()) continue;
        
        // Use GPU to find best split for this feature
        auto [threshold, score] = gpu_trainer.find_best_split(
            indices_u32,
            feature_idx,
            thresholds
        );
        
        if (score < best.score) {
            best.score = score;
            best.feature_idx = feature_idx;
            best.threshold = threshold;
        }
    }
    
    return best;
}

std::pair<std::vector<size_t>, std::vector<size_t>> 
DecisionTree::split_data(
    const std::vector<float>& data,
    const std::vector<size_t>& indices,
    size_t n_features,
    size_t feature_idx,
    float threshold
) {
    std::vector<size_t> left, right;
    
    for (size_t idx : indices) {
        float value = data[idx * n_features + feature_idx];
        if (value <= threshold) {
            left.push_back(idx);
        } else {
            right.push_back(idx);
        }
    }
    
    return {left, right};
}

float DecisionTree::calculate_mse(
    const std::vector<float>& labels,
    const std::vector<size_t>& indices
) {
    if (indices.empty()) {
        return 0.0f;
    }
    
    float sum = 0.0f;
    for (size_t idx : indices) {
        sum += labels[idx];
    }
    float mean = sum / indices.size();
    
    float mse = 0.0f;
    for (size_t idx : indices) {
        float diff = labels[idx] - mean;
        mse += diff * diff;
    }
    
    return mse * indices.size();
}

float DecisionTree::predict(const float* sample) const {
    if (!root_) {
        std::cerr << "ERROR: Tree not trained" << std::endl;
        return 0.0f;
    }
    return root_->predict(sample);
}

// ═══════════════════════════════════════════════════════════════════════
// RandomForest Implementation
// ═══════════════════════════════════════════════════════════════════════

RandomForest::RandomForest(size_t n_estimators, size_t max_depth)
    : n_estimators_(n_estimators)
    , max_depth_(max_depth) {
    trees_.reserve(n_estimators);
}

void RandomForest::train_cpu(const Dataset& dataset) {
    std::random_device rd;
    std::mt19937 rng(rd());
    
    std::cout << "[TRAINING] RandomForest: " << n_estimators_ 
              << " trees, max_depth " << max_depth_ << std::endl;
    std::cout << "[TRAINING] Training on CPU..." << std::endl;
    
    for (size_t i = 0; i < n_estimators_; ++i) {
        auto [sampled_data, sampled_labels] = dataset.bootstrap_sample(rng);
        
        DecisionTree tree(max_depth_);
        tree.train_cpu(sampled_data, sampled_labels, 
                      dataset.size(), dataset.n_features(), rng);
        
        trees_.push_back(std::move(tree));
        
        if ((i + 1) % 10 == 0) {
            std::cerr << "Trained " << (i + 1) << "/" << n_estimators_ 
                      << " trees (CPU)" << std::endl;
        }
    }
    
    std::cout << "[TRAINING] Training completed!" << std::endl;
}

void RandomForest::train_with_gpu(const Dataset& dataset, GpuTrainer& gpu_trainer) {
    if (!gpu_trainer.is_available()) {
        std::cout << "[TRAINING] GPU not available, using CPU..." << std::endl;
        train_cpu(dataset);
        return;
    }
    
    std::random_device rd;
    std::mt19937 rng(rd());
    
    std::cout << "[TRAINING] RandomForest: " << n_estimators_ 
              << " trees, max_depth " << max_depth_ << std::endl;
    std::cout << "[TRAINING] Training with GPU acceleration (wasi:gpu)..." << std::endl;
    
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
        
        if ((i + 1) % 10 == 0) {
            std::cerr << "Trained " << (i + 1) << "/" << n_estimators_ 
                      << " trees (GPU)" << std::endl;
        }
    }
    
    std::cout << "[TRAINING] Training completed!" << std::endl;
}

std::vector<float> RandomForest::predict_cpu(
    const std::vector<float>& data,
    size_t n_samples,
    size_t n_features
) const {
    std::vector<float> predictions;
    predictions.reserve(n_samples);
    
    for (size_t i = 0; i < n_samples; ++i) {
        const float* sample = &data[i * n_features];
        
        float sum = 0.0f;
        for (const auto& tree : trees_) {
            sum += tree.predict(sample);
        }
        float avg = sum / trees_.size();
        
        predictions.push_back(avg);
    }
    
    return predictions;
}

std::vector<float> RandomForest::predict_with_gpu(
    const std::vector<float>& data,
    size_t n_samples,
    size_t n_features,
    GpuPredictor& predictor
) {
    if (!predictor.is_available()) {
        std::cout << "[INFERENCE] GPU not available, using CPU..." << std::endl;
        return predict_cpu(data, n_samples, n_features);
    }
    
    // Get tree predictions for all samples
    std::vector<float> tree_predictions;
    tree_predictions.reserve(n_samples * trees_.size());
    
    for (size_t i = 0; i < n_samples; ++i) {
        const float* sample = &data[i * n_features];
        auto preds = get_tree_predictions(sample);
        tree_predictions.insert(tree_predictions.end(), preds.begin(), preds.end());
    }
    
    // Average on GPU
    return predictor.average_predictions(tree_predictions, n_samples, trees_.size());
}

std::vector<float> RandomForest::get_tree_predictions(const float* sample) const {
    std::vector<float> predictions;
    predictions.reserve(trees_.size());
    
    for (const auto& tree : trees_) {
        predictions.push_back(tree.predict(sample));
    }
    
    return predictions;
}

// ═══════════════════════════════════════════════════════════════════════
// Serialization (JSON) - Simplified
// ═══════════════════════════════════════════════════════════════════════

std::string DecisionTree::to_json() const {
    return "{}";
}

DecisionTree DecisionTree::from_json(const std::string& json_str) {
    return DecisionTree(0);
}

std::string RandomForest::to_json() const {
    char buf[256];
    snprintf(buf, sizeof(buf), 
             "{\"n_estimators\":%zu,\"max_depth\":%zu,\"n_trees\":%zu}",
             n_estimators_, max_depth_, trees_.size());
    return std::string(buf);
}

RandomForest RandomForest::from_json(const std::string& json_str) {
    // Simple parse - not production quality
    return RandomForest(100, 10);
}

} // namespace ml
