//! Dataset handling and preprocessing

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dataset {
    pub data: Vec<f32>,
    pub labels: Vec<f32>,
    pub n_samples: usize,
    pub n_features: usize,
}

impl Dataset {
    pub fn new(
        data: Vec<f32>,
        labels: Vec<f32>,
        n_samples: usize,
        n_features: usize,
    ) -> Result<Self, String> {
        if data.len() != n_samples * n_features {
            return Err(format!(
                "Data size mismatch: expected {}, got {}",
                n_samples * n_features,
                data.len()
            ));
        }
        
        if labels.len() != n_samples {
            return Err(format!(
                "Labels size mismatch: expected {}, got {}",
                n_samples,
                labels.len()
            ));
        }
        
        Ok(Self {
            data,
            labels,
            n_samples,
            n_features,
        })
    }
    
    /// Get a sample by index
    pub fn get_sample(&self, idx: usize) -> &[f32] {
        let start = idx * self.n_features;
        let end = start + self.n_features;
        &self.data[start..end]
    }
    
    /// Get a label by index
    pub fn get_label(&self, idx: usize) -> f32 {
        self.labels[idx]
    }
    
    /// Bootstrap sampling (with replacement) for bagging
    pub fn bootstrap_sample(&self, rng: &mut impl rand::Rng) -> (Vec<f32>, Vec<f32>) {
        use rand::seq::SliceRandom;
        
        let indices: Vec<usize> = (0..self.n_samples)
            .map(|_| rng.gen_range(0..self.n_samples))
            .collect();
        
        let mut sampled_data = Vec::with_capacity(self.n_samples * self.n_features);
        let mut sampled_labels = Vec::with_capacity(self.n_samples);
        
        for &idx in &indices {
            sampled_data.extend_from_slice(self.get_sample(idx));
            sampled_labels.push(self.get_label(idx));
        }
        
        (sampled_data, sampled_labels)
    }
}
