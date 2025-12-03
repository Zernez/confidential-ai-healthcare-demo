/**
 * @file dataset.cpp
 * @brief Implementation of Dataset class
 */

#include "dataset.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>

namespace ml {

Dataset::Dataset() 
    : n_samples_(0), n_features_(0) {}

Dataset::Dataset(const std::vector<float>& data,
                 const std::vector<float>& labels,
                 size_t n_samples,
                 size_t n_features)
    : data_(data)
    , labels_(labels)
    , n_samples_(n_samples)
    , n_features_(n_features) 
{
    if (data.size() != n_samples * n_features) {
    fprintf(stderr, "Data size mismatch\n");
    exit(1);
    }
    if (labels.size() != n_samples) {
    fprintf(stderr, "Labels size mismatch\n");
    exit(1);
    }
}

Dataset Dataset::from_csv(const std::string& filepath, size_t n_features) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        fprintf(stderr, "Cannot open file: %s\n", filepath.c_str());
        exit(1);
    }
    std::vector<float> data;
    std::vector<float> labels;
    std::string line;
    size_t n_samples = 0;
    // Skip header
    std::getline(file, line);
    while (std::getline(file, line)) {
        std::istringstream ss(line);
        std::string token;
        // Read features
        for (size_t i = 0; i < n_features; ++i) {
            if (!std::getline(ss, token, ',')) {
                fprintf(stderr, "Invalid CSV format at line %zu\n", n_samples + 2);
                exit(1);
            }
            float value = 0.0f;
            // WASI: no try/catch, use explicit error check
            char* endptr = nullptr;
            value = strtof(token.c_str(), &endptr);
            if (endptr == token.c_str() || *endptr != '\0') {
                fprintf(stderr, "Invalid float value: %s\n", token.c_str());
                exit(1);
            }
            data.push_back(value);
        }
        // Read label (last column)
        if (!std::getline(ss, token, ',')) {
            fprintf(stderr, "Missing label at line %zu\n", n_samples + 2);
            exit(1);
        }
        float label = 0.0f;
        char* endptr = nullptr;
        label = strtof(token.c_str(), &endptr);
        if (endptr == token.c_str() || *endptr != '\0') {
            fprintf(stderr, "Invalid label value: %s\n", token.c_str());
            exit(1);
        }
        labels.push_back(label);
        n_samples++;
    }
    file.close();
    return ml::Dataset(data, labels, n_samples, n_features);
}

const float* Dataset::get_sample(size_t idx) const {
    if (idx >= n_samples_) {
    fprintf(stderr, "Sample index out of range\n");
    exit(1);
    }
    return &data_[idx * n_features_];
}

float Dataset::get_label(size_t idx) const {
    if (idx >= n_samples_) {
    fprintf(stderr, "Label index out of range\n");
    exit(1);
    }
    return labels_[idx];
}

std::pair<std::vector<float>, std::vector<float>> 
Dataset::bootstrap_sample(std::mt19937& rng) const {
    std::uniform_int_distribution<size_t> dist(0, n_samples_ - 1);
    
    std::vector<float> sampled_data;
    std::vector<float> sampled_labels;
    
    sampled_data.reserve(n_samples_ * n_features_);
    sampled_labels.reserve(n_samples_);
    
    // Sample with replacement
    for (size_t i = 0; i < n_samples_; ++i) {
        size_t idx = dist(rng);
        
        // Copy sample
        const float* sample = get_sample(idx);
        sampled_data.insert(sampled_data.end(), sample, sample + n_features_);
        
        // Copy label
        sampled_labels.push_back(get_label(idx));
    }
    
    return {sampled_data, sampled_labels};
}

} // namespace ml
