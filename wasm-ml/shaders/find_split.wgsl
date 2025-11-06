// Best split finding shader
// Computes MSE for each candidate threshold in parallel

struct SplitParams {
    n_samples: u32,
    n_features: u32,
    feature_idx: u32,
    n_thresholds: u32,
}

@group(0) @binding(0) var<storage, read> data: array<f32>;
@group(0) @binding(1) var<storage, read> labels: array<f32>;
@group(0) @binding(2) var<storage, read> indices: array<u32>;
@group(0) @binding(3) var<storage, read> thresholds: array<f32>;
@group(0) @binding(4) var<storage, read_write> scores: array<f32>;
@group(0) @binding(5) var<uniform> params: SplitParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let threshold_idx = global_id.x;
    
    // Boundary check
    if (threshold_idx >= params.n_thresholds) {
        return;
    }
    
    let threshold = thresholds[threshold_idx];
    
    // First pass: compute sums and counts for left and right
    var left_sum: f32 = 0.0;
    var left_count: u32 = 0u;
    var right_sum: f32 = 0.0;
    var right_count: u32 = 0u;
    
    for (var i: u32 = 0u; i < params.n_samples; i++) {
        let sample_idx = indices[i];
        let feature_value = data[sample_idx * params.n_features + params.feature_idx];
        let label = labels[sample_idx];
        
        if (feature_value <= threshold) {
            left_sum += label;
            left_count++;
        } else {
            right_sum += label;
            right_count++;
        }
    }
    
    // Avoid division by zero
    if (left_count == 0u || right_count == 0u) {
        scores[threshold_idx] = 1e10; // Very high score (bad split)
        return;
    }
    
    // Compute means
    let left_mean = left_sum / f32(left_count);
    let right_mean = right_sum / f32(right_count);
    
    // Second pass: compute MSE
    var mse: f32 = 0.0;
    
    for (var i: u32 = 0u; i < params.n_samples; i++) {
        let sample_idx = indices[i];
        let feature_value = data[sample_idx * params.n_features + params.feature_idx];
        let label = labels[sample_idx];
        
        var diff: f32;
        if (feature_value <= threshold) {
            diff = label - left_mean;
        } else {
            diff = label - right_mean;
        }
        
        mse += diff * diff;
    }
    
    // Store weighted MSE (weighted by group size for proper comparison)
    scores[threshold_idx] = mse;
}
