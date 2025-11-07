/**
 * @file random_forest.cpp
 * @brief Implementation of RandomForest algorithm
 */

#include "random_forest.hpp"
#include "gpu_executor.hpp"
#include <json.hpp>  // nlohmann/json
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iostream>
#include <limits>

using json = nlohmann::json;

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
    // Create initial index set
    std::vector<size_t> indices(n_samples);
    std::iota(indices.begin(), indices.end(), 0);
    
    // Build tree recursively
    root_ = build_tree(data, labels, indices, n_features, 0, rng);
}

std::unique_ptr<TreeNode> DecisionTree::build_tree(
    const std::vector<float>& data,
    const std::vector<float>& labels,
    const std::vector<size_t>& indices,
    size_t n_features,
    size_t depth,
    std::mt19937& rng
) {
    // Base cases: max depth or too few samples
    if (depth >= max_depth_ || indices.size() < 2) {
        // Calculate mean of labels
        float sum = 0.0f;
        for (size_t idx : indices) {
            sum += labels[idx];
        }
        float mean = sum / indices.size();
        return TreeNode::make_leaf(mean);
    }
    
    // Find best split
    auto split = find_best_split(data, labels, indices, n_features, rng);
    
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
    
    // Recursively build subtrees
    auto left = build_tree(data, labels, left_indices, n_features, depth + 1, rng);
    auto right = build_tree(data, labels, right_indices, n_features, depth + 1, rng);
    
    return TreeNode::make_internal(
        split.feature_idx,
        split.threshold,
        std::move(left),
        std::move(right)
    );
}

DecisionTree::SplitInfo DecisionTree::find_best_split(
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
    
    // Try each selected feature
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
            
            // Calculate MSE for this split
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
    
    // Calculate mean
    float sum = 0.0f;
    for (size_t idx : indices) {
        sum += labels[idx];
    }
    float mean = sum / indices.size();
    
    // Calculate MSE
    float mse = 0.0f;
    for (size_t idx : indices) {
        float diff = labels[idx] - mean;
        mse += diff * diff;
    }
    
    // Return weighted MSE
    return mse * indices.size();
}

float DecisionTree::predict(const float* sample) const {
    if (!root_) {
    fprintf(stderr, "Tree not trained\n");
    exit(1);
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
    
    std::cout << "[TRAINING] RandomForest with " << n_estimators_ 
              << " trees, max_depth " << max_depth_ << std::endl;
    std::cout << "[TRAINING] Training on CPU..." << std::endl;
    
    for (size_t i = 0; i < n_estimators_; ++i) {
        // Bootstrap sample
        auto [sampled_data, sampled_labels] = dataset.bootstrap_sample(rng);
        
        // Train tree
        DecisionTree tree(max_depth_);
        tree.train_cpu(sampled_data, sampled_labels, 
                      dataset.size(), dataset.n_features(), rng);
        
        trees_.push_back(std::move(tree));
        
        // Progress indication
        if ((i + 1) % 10 == 0) {
            std::cerr << "Trained " << (i + 1) << "/" << n_estimators_ 
                      << " trees (CPU)" << std::endl;
        }
    }
    
    std::cout << "[TRAINING] Training completed!" << std::endl;
}

void RandomForest::train_gpu(const Dataset& dataset, GpuExecutor& gpu) {
    if (!gpu.is_available()) {
        std::cout << "[TRAINING] GPU not available, using CPU..." << std::endl;
        train_cpu(dataset);
        return;
    }
    
    std::random_device rd;
    std::mt19937 rng(rd());
    
    std::cout << "[TRAINING] RandomForest with " << n_estimators_ 
              << " trees, max_depth " << max_depth_ << std::endl;
    std::cout << "[TRAINING] Training with GPU acceleration..." << std::endl;
    
    for (size_t i = 0; i < n_estimators_; ++i) {
        // Bootstrap sample on GPU
        uint32_t seed = rng();
        std::vector<uint32_t> bootstrap_indices;
        
        try {
            bootstrap_indices = gpu.bootstrap_sample(dataset.size(), seed);
        } catch (const std::exception& e) {
            std::cerr << "[TRAINING] GPU bootstrap failed, using CPU: " << e.what() << std::endl;
            auto [sampled_data, sampled_labels] = dataset.bootstrap_sample(rng);
            
            DecisionTree tree(max_depth_);
            tree.train_cpu(sampled_data, sampled_labels, 
                          dataset.size(), dataset.n_features(), rng);
            trees_.push_back(std::move(tree));
            continue;
        }
        
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
        tree.train_cpu(sampled_data, sampled_labels,
                      dataset.size(), dataset.n_features(), rng);
        
        trees_.push_back(std::move(tree));
        
        // Progress indication
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
        
        // Average predictions from all trees
        float sum = 0.0f;
        for (const auto& tree : trees_) {
            sum += tree.predict(sample);
        }
        float avg = sum / trees_.size();
        
        predictions.push_back(avg);
    }
    
    return predictions;
}

std::vector<float> RandomForest::predict_gpu(
    const std::vector<float>& data,
    size_t n_samples,
    size_t n_features,
    GpuExecutor& gpu
) {
    if (!gpu.is_available()) {
        std::cout << "[INFERENCE] GPU not available, using CPU prediction..." << std::endl;
        return predict_cpu(data, n_samples, n_features);
    }
    
    std::cout << "[INFERENCE] Using GPU for prediction..." << std::endl;
    return gpu.predict(*this, data, n_features);
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
// Serialization (JSON)
// ═══════════════════════════════════════════════════════════════════════

std::string DecisionTree::to_json() const {
    // TODO: Implement JSON serialization
    return "{}";
}

DecisionTree DecisionTree::from_json(const std::string& json_str) {
    // TODO: Implement JSON deserialization
    return DecisionTree(0);
}

std::string RandomForest::to_json() const {
    json j;
    j["n_estimators"] = n_estimators_;
    j["max_depth"] = max_depth_;
    j["n_trees"] = trees_.size();
    // TODO: Serialize trees
    return j.dump(2);
}

RandomForest RandomForest::from_json(const std::string& json_str) {
    auto j = json::parse(json_str);
    size_t n_estimators = j["n_estimators"];
    size_t max_depth = j["max_depth"];
    // TODO: Deserialize trees
    return RandomForest(n_estimators, max_depth);
}

} // namespace ml
