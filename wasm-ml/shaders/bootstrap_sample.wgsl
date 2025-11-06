// Bootstrap sampling shader
// Generates random indices for bootstrap sampling (sampling with replacement)

struct BootstrapParams {
    n_samples: u32,
    seed: u32,
}

@group(0) @binding(0) var<storage, read_write> indices: array<u32>;
@group(0) @binding(1) var<uniform> params: BootstrapParams;

// Simple XORshift PRNG
fn xorshift(seed: u32) -> u32 {
    var x = seed;
    x ^= x << 13u;
    x ^= x >> 17u;
    x ^= x << 5u;
    return x;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    // Boundary check
    if (idx >= params.n_samples) {
        return;
    }
    
    // Generate random index using XORshift
    // Use different seed for each thread
    var rng_state = params.seed + idx * 747796405u + 2891336453u;
    
    // Multiple rounds for better randomness
    rng_state = xorshift(rng_state);
    rng_state = xorshift(rng_state);
    
    // Map to [0, n_samples)
    let random_idx = rng_state % params.n_samples;
    
    indices[idx] = random_idx;
}
