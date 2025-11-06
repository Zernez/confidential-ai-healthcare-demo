// WebGPU Compute Shader for averaging tree predictions
// Each workgroup processes 64 samples in parallel

struct Params {
    n_trees: u32,
    n_samples: u32,
    padding: vec2<u32>,
}

@group(0) @binding(0) var<storage, read> tree_predictions: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let sample_idx = global_id.x;
    
    // Boundary check
    if (sample_idx >= params.n_samples) {
        return;
    }
    
    // Calculate average of all tree predictions for this sample
    var sum: f32 = 0.0;
    let base_idx = sample_idx * params.n_trees;
    
    for (var tree_idx: u32 = 0u; tree_idx < params.n_trees; tree_idx = tree_idx + 1u) {
        sum = sum + tree_predictions[base_idx + tree_idx];
    }
    
    output[sample_idx] = sum / f32(params.n_trees);
}
