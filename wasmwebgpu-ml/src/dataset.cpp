/**
 * @file dataset.cpp
 * @brief Implementation of Dataset class
 */

#include "dataset.hpp"
#include <csv.h>  // fast-cpp-csv-parser
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
        throw std::invalid_argument("Data size mismatch");
    }
    if (labels.size() != n_samples) {
        throw std::invalid_argument("Labels size mismatch");
    }
}

Dataset Dataset::from_csv(const std::string& filepath, size_t n_features) {
    std::cout << "[LOADING] Reading CSV: " << filepath << std::endl;
    
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filepath);
    }
    
    std::vector<float> data;
    std::vector<float> labels;
    std::string line;
    size_t n_samples = 0;
    
    // Skip header
    std::getline(file, line);
    
    // Read data
    while (std::getline(file, line)) {
        std::istringstream ss(line);
        std::string token;
        
        // Read features
        for (size_t i = 0; i < n_features; ++i) {
            if (!std::getline(ss, token, ',')) {
                throw std::runtime_error("Invalid CSV format at line " + 
                                       std::to_string(n_samples + 2));
            }
            try {
                float value = std::stof(token);
                data.push_back(value);
            } catch (const std::exception& e) {
                throw std::runtime_error("Invalid float value: " + token);
            }
        }
        
        // Read label (last column)
        if (!std::getline(ss, token, ',')) {
            throw std::runtime_error("Missing label at line " + 
                                   std::to_string(n_samples + 2));
        }
        try {
            float label = std::stof(token);
            labels.push_back(label);
        } catch (const std::exception& e) {
            throw std::runtime_error("Invalid label value: " + token);
        }
        
        n_samples++;
    }
    
    file.close();
    
    std::cout << "[LOADING] Loaded " << n_samples << " samples with " 
              << n_features << " features" << std::endl;
    
    return Dataset(data, labels, n_samples, n_features);
}

const float* Dataset::get_sample(size_t idx) const {
    if (idx >= n_samples_) {
        throw std::out_of_range("Sample index out of range");
    }
    return &data_[idx * n_features_];
}

float Dataset::get_label(size_t idx) const {
    if (idx >= n_samples_) {
        throw std::out_of_range("Label index out of range");
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
