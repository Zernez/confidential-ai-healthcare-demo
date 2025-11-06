/**
 * @file dataset.hpp
 * @brief Dataset management for ML training and inference
 * 
 * Handles CSV loading, data storage, and bootstrap sampling for RandomForest
 */

#ifndef DATASET_HPP
#define DATASET_HPP

#include <vector>
#include <string>
#include <random>
#include <stdexcept>

namespace ml {

/**
 * @class Dataset
 * @brief Container for ML dataset with features and labels
 */
class Dataset {
public:
    /**
     * @brief Construct empty dataset
     */
    Dataset();
    
    /**
     * @brief Construct dataset from vectors
     * @param data Flat array of features (row-major: n_samples * n_features)
     * @param labels Target values (n_samples)
     * @param n_samples Number of samples
     * @param n_features Number of features per sample
     */
    Dataset(const std::vector<float>& data,
            const std::vector<float>& labels,
            size_t n_samples,
            size_t n_features);
    
    /**
     * @brief Load dataset from CSV file
     * @param filepath Path to CSV file
     * @param n_features Number of feature columns (target is last column)
     * @return Dataset object
     */
    static Dataset from_csv(const std::string& filepath, size_t n_features);
    
    /**
     * @brief Get a sample by index
     * @param idx Sample index
     * @return Pointer to feature array (n_features elements)
     */
    const float* get_sample(size_t idx) const;
    
    /**
     * @brief Get a label by index
     * @param idx Sample index
     * @return Label value
     */
    float get_label(size_t idx) const;
    
    /**
     * @brief Bootstrap sample (sampling with replacement)
     * @param rng Random number generator
     * @return Pair of (sampled_data, sampled_labels)
     */
    std::pair<std::vector<float>, std::vector<float>> 
    bootstrap_sample(std::mt19937& rng) const;
    
    /**
     * @brief Get number of samples
     */
    size_t size() const { return n_samples_; }
    
    /**
     * @brief Get number of features
     */
    size_t n_features() const { return n_features_; }
    
    /**
     * @brief Get raw data pointer
     */
    const std::vector<float>& data() const { return data_; }
    
    /**
     * @brief Get raw labels pointer
     */
    const std::vector<float>& labels() const { return labels_; }

private:
    std::vector<float> data_;       ///< Flat array of features (row-major)
    std::vector<float> labels_;     ///< Target values
    size_t n_samples_;              ///< Number of samples
    size_t n_features_;             ///< Number of features per sample
};

} // namespace ml

#endif // DATASET_HPP
