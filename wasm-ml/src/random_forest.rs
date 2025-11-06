//! RandomForest implementation for regression
//! 
//! This implementation uses bagging (bootstrap aggregating) to train
//! multiple decision trees and averages their predictions.
//! 
//! Now includes GPU-accelerated training via WebGPU compute shaders.

use serde::{Deserialize, Serialize};
use rand::Rng;
use rand_xoshiro::Xoshiro256PlusPlus;
use rand::SeedableRng;

use crate::data::Dataset;
use crate::gpu_training::GpuTrainer;

/// A single decision tree node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TreeNode {
    Leaf {
        value: f32,
    },
    Internal {
        feature_idx: usize,
        threshold: f32,
        left: Box<TreeNode>,
        right: Box<TreeNode>,
    },
}

impl TreeNode {
    /// Predict for a single sample
    pub fn predict(&self, sample: &[f32]) -> f32 {
        match self {
            TreeNode::Leaf { value } => *value,
            TreeNode::Internal { feature_idx, threshold, left, right } => {
                if sample[*feature_idx] <= *threshold {
                    left.predict(sample)
                } else {
                    right.predict(sample)
                }
            }
        }
    }
}

/// A single decision tree
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionTree {
    root: TreeNode,
    max_depth: usize,
}

impl DecisionTree {
    pub fn new(max_depth: usize) -> Self {
        Self {
            root: TreeNode::Leaf { value: 0.0 },
            max_depth,
        }
    }
    
    /// Train the tree on a dataset
    pub fn train(&mut self, data: &[f32], labels: &[f32], n_features: usize, rng: &mut impl Rng) {
        let n_samples = labels.len();
        let indices: Vec<usize> = (0..n_samples).collect();
        self.root = self.build_tree(data, labels, &indices, n_features, 0, rng);
    }
    
    /// Train tree with GPU-accelerated split finding
    pub async fn train_gpu(
        &mut self,
        data: &[f32],
        labels: &[f32],
        indices: &[u32],
        n_features: usize,
        gpu_trainer: &GpuTrainer,
        rng: &mut impl Rng,
    ) -> Result<(), String> {
        let indices_usize: Vec<usize> = indices.iter().map(|&x| x as usize).collect();
        self.root = self.build_tree_gpu(data, labels, &indices_usize, n_features, 0, gpu_trainer, rng)
            .await?;
        Ok(())
    }
    
    fn build_tree(
        &self,
        data: &[f32],
        labels: &[f32],
        indices: &[usize],
        n_features: usize,
        depth: usize,
        rng: &mut impl Rng,
    ) -> TreeNode {
        // Base cases: max depth or too few samples
        if depth >= self.max_depth || indices.len() < 2 {
            let mean = indices.iter().map(|&i| labels[i]).sum::<f32>() / indices.len() as f32;
            return TreeNode::Leaf { value: mean };
        }
        
        // Find best split
        let (best_feature, best_threshold, best_score) = self.find_best_split(
            data,
            labels,
            indices,
            n_features,
            rng,
        );
        
        // If no good split found, create leaf
        if best_score.is_infinite() {
            let mean = indices.iter().map(|&i| labels[i]).sum::<f32>() / indices.len() as f32;
            return TreeNode::Leaf { value: mean };
        }
        
        // Split data
        let (left_indices, right_indices) = self.split_data(
            data,
            indices,
            n_features,
            best_feature,
            best_threshold,
        );
        
        // Recursively build subtrees
        let left = Box::new(self.build_tree(data, labels, &left_indices, n_features, depth + 1, rng));
        let right = Box::new(self.build_tree(data, labels, &right_indices, n_features, depth + 1, rng));
        
        TreeNode::Internal {
            feature_idx: best_feature,
            threshold: best_threshold,
            left,
            right,
        }
    }
    
    async fn build_tree_gpu(
        &self,
        data: &[f32],
        labels: &[f32],
        indices: &[usize],
        n_features: usize,
        depth: usize,
        gpu_trainer: &GpuTrainer,
        rng: &mut impl Rng,
    ) -> Result<TreeNode, String> {
        // Base cases
        if depth >= self.max_depth || indices.len() < 2 {
            let mean = indices.iter().map(|&i| labels[i]).sum::<f32>() / indices.len() as f32;
            return Ok(TreeNode::Leaf { value: mean });
        }
        
        // Find best split using GPU
        let (best_feature, best_threshold, best_score) = self.find_best_split_gpu(
            data,
            labels,
            indices,
            n_features,
            gpu_trainer,
            rng,
        ).await?;
        
        if best_score.is_infinite() {
            let mean = indices.iter().map(|&i| labels[i]).sum::<f32>() / indices.len() as f32;
            return Ok(TreeNode::Leaf { value: mean });
        }
        
        // Split data
        let (left_indices, right_indices) = self.split_data(
            data,
            indices,
            n_features,
            best_feature,
            best_threshold,
        );
        
        // Recursively build subtrees
        let left = Box::new(self.build_tree_gpu(data, labels, &left_indices, n_features, depth + 1, gpu_trainer, rng).await?);
        let right = Box::new(self.build_tree_gpu(data, labels, &right_indices, n_features, depth + 1, gpu_trainer, rng).await?);
        
        Ok(TreeNode::Internal {
            feature_idx: best_feature,
            threshold: best_threshold,
            left,
            right,
        })
    }
    
    async fn find_best_split_gpu(
        &self,
        data: &[f32],
        labels: &[f32],
        indices: &[usize],
        n_features: usize,
        gpu_trainer: &GpuTrainer,
        rng: &mut impl Rng,
    ) -> Result<(usize, f32, f32), String> {
        let mut best_feature = 0;
        let mut best_threshold = 0.0;
        let mut best_score = f32::INFINITY;
        
        // Random feature selection
        let n_features_to_try = (n_features as f32).sqrt().ceil() as usize;
        let mut features: Vec<usize> = (0..n_features).collect();
        
        use rand::seq::SliceRandom;
        features.shuffle(rng);
        
        // Convert indices to u32 for GPU
        let indices_u32: Vec<u32> = indices.iter().map(|&x| x as u32).collect();
        
        // Try each feature
        for &feature_idx in features.iter().take(n_features_to_try) {
            // Use GPU to find best split for this feature
            let (threshold, score) = gpu_trainer.find_best_split(
                data,
                labels,
                &indices_u32,
                n_features,
                feature_idx,
            ).await?;
            
            if score < best_score {
                best_score = score;
                best_feature = feature_idx;
                best_threshold = threshold;
            }
        }
        
        Ok((best_feature, best_threshold, best_score))
    }
    
    fn find_best_split(
        &self,
        data: &[f32],
        labels: &[f32],
        indices: &[usize],
        n_features: usize,
        rng: &mut impl Rng,
    ) -> (usize, f32, f32) {
        let mut best_feature = 0;
        let mut best_threshold = 0.0;
        let mut best_score = f32::INFINITY;
        
        // Random feature selection (sqrt of total features)
        let n_features_to_try = (n_features as f32).sqrt().ceil() as usize;
        let mut features: Vec<usize> = (0..n_features).collect();
        
        // Shuffle and select random features
        use rand::seq::SliceRandom;
        features.shuffle(rng);
        
        for &feature_idx in features.iter().take(n_features_to_try) {
            // Get unique values for this feature
            let mut values: Vec<f32> = indices
                .iter()
                .map(|&i| data[i * n_features + feature_idx])
                .collect();
            values.sort_by(|a, b| a.partial_cmp(b).unwrap());
            values.dedup();
            
            // Try different thresholds
            for i in 0..values.len().saturating_sub(1) {
                let threshold = (values[i] + values[i + 1]) / 2.0;
                
                // Calculate MSE for this split
                let (left_indices, right_indices) = self.split_data(
                    data,
                    indices,
                    n_features,
                    feature_idx,
                    threshold,
                );
                
                if left_indices.is_empty() || right_indices.is_empty() {
                    continue;
                }
                
                let score = self.calculate_mse(&left_indices, labels)
                    + self.calculate_mse(&right_indices, labels);
                
                if score < best_score {
                    best_score = score;
                    best_feature = feature_idx;
                    best_threshold = threshold;
                }
            }
        }
        
        (best_feature, best_threshold, best_score)
    }
    
    fn split_data(
        &self,
        data: &[f32],
        indices: &[usize],
        n_features: usize,
        feature_idx: usize,
        threshold: f32,
    ) -> (Vec<usize>, Vec<usize>) {
        let mut left = Vec::new();
        let mut right = Vec::new();
        
        for &idx in indices {
            let value = data[idx * n_features + feature_idx];
            if value <= threshold {
                left.push(idx);
            } else {
                right.push(idx);
            }
        }
        
        (left, right)
    }
    
    fn calculate_mse(&self, indices: &[usize], labels: &[f32]) -> f32 {
        if indices.is_empty() {
            return 0.0;
        }
        
        let mean = indices.iter().map(|&i| labels[i]).sum::<f32>() / indices.len() as f32;
        let mse = indices
            .iter()
            .map(|&i| {
                let diff = labels[i] - mean;
                diff * diff
            })
            .sum::<f32>()
            / indices.len() as f32;
        
        mse * indices.len() as f32
    }
    
    pub fn predict(&self, sample: &[f32]) -> f32 {
        self.root.predict(sample)
    }
}

/// RandomForest ensemble
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RandomForest {
    trees: Vec<DecisionTree>,
    n_estimators: usize,
    max_depth: usize,
}

impl RandomForest {
    pub fn new(n_estimators: usize, max_depth: usize) -> Self {
        Self {
            trees: Vec::with_capacity(n_estimators),
            n_estimators,
            max_depth,
        }
    }
    
    /// Train the forest on CPU
    pub fn train(&mut self, dataset: &Dataset) -> Result<(), String> {
        let mut rng = Xoshiro256PlusPlus::from_entropy();
        
        for i in 0..self.n_estimators {
            // Bootstrap sample
            let (sampled_data, sampled_labels) = dataset.bootstrap_sample(&mut rng);
            
            // Train tree
            let mut tree = DecisionTree::new(self.max_depth);
            tree.train(&sampled_data, &sampled_labels, dataset.n_features, &mut rng);
            
            self.trees.push(tree);
            
            // Progress indication
            if (i + 1) % 10 == 0 {
                eprintln!("Trained {}/{} trees (CPU)", i + 1, self.n_estimators);
            }
        }
        
        Ok(())
    }
    
    /// Train the forest with GPU acceleration
    pub async fn train_gpu(&mut self, dataset: &Dataset, gpu_trainer: &GpuTrainer) -> Result<(), String> {
        let mut rng = Xoshiro256PlusPlus::from_entropy();
        
        for i in 0..self.n_estimators {
            // Bootstrap sample on GPU
            let seed = rng.gen();
            let bootstrap_indices = gpu_trainer.bootstrap_sample(dataset.n_samples, seed)
                .await
                .map_err(|e| format!("GPU bootstrap failed: {}", e))?;
            
            // Extract bootstrapped data
            let mut sampled_data = Vec::with_capacity(dataset.n_samples * dataset.n_features);
            let mut sampled_labels = Vec::with_capacity(dataset.n_samples);
            
            for &idx in &bootstrap_indices {
                sampled_data.extend_from_slice(dataset.get_sample(idx as usize));
                sampled_labels.push(dataset.get_label(idx as usize));
            }
            
            // Train tree with GPU-accelerated split finding
            let mut tree = DecisionTree::new(self.max_depth);
            tree.train_gpu(
                &sampled_data,
                &sampled_labels,
                &bootstrap_indices,
                dataset.n_features,
                gpu_trainer,
                &mut rng,
            ).await?;
            
            self.trees.push(tree);
            
            // Progress indication
            if (i + 1) % 10 == 0 {
                eprintln!("Trained {}/{} trees (GPU)", i + 1, self.n_estimators);
            }
        }
        
        Ok(())
    }
    
    /// Predict on CPU (fallback)
    pub fn predict_cpu(
        &self,
        data: &[f32],
        n_samples: usize,
        n_features: usize,
    ) -> Result<Vec<f32>, String> {
        let mut predictions = Vec::with_capacity(n_samples);
        
        for sample_idx in 0..n_samples {
            let start = sample_idx * n_features;
            let end = start + n_features;
            let sample = &data[start..end];
            
            // Average predictions from all trees
            let sum: f32 = self.trees.iter().map(|tree| tree.predict(sample)).sum();
            let avg = sum / self.trees.len() as f32;
            
            predictions.push(avg);
        }
        
        Ok(predictions)
    }
    
    /// Get tree predictions for GPU processing
    pub fn get_tree_predictions(&self, sample: &[f32]) -> Vec<f32> {
        self.trees.iter().map(|tree| tree.predict(sample)).collect()
    }
    
    pub fn n_trees(&self) -> usize {
        self.trees.len()
    }
}
