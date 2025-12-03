# WASI GPU Interface

Portable GPU compute interface for ML workloads in WebAssembly.

## Overview

This WIT (WebAssembly Interface Types) package defines a portable interface for GPU compute operations. WASM modules can use this interface without knowing the underlying GPU backend.

## Architecture

```
┌─────────────────────────────────────┐
│          WASM Module                │
│  (wasm-ml, wasmwebgpu-ml, etc.)    │
└─────────────────┬───────────────────┘
                  │ wasi:gpu
                  ▼
┌─────────────────────────────────────┐
│        Host Runtime                 │
│  ┌─────────────┐ ┌───────────────┐  │
│  │ WebGPU/     │ │    CUDA       │  │
│  │ Vulkan      │ │  (cuBLAS)     │  │
│  └─────────────┘ └───────────────┘  │
└─────────────────────────────────────┘
```

## Interfaces

### `compute` - Low-level GPU operations

- `buffer-create` - Allocate GPU memory
- `buffer-write` - Upload data to GPU
- `buffer-read` - Download data from GPU
- `buffer-copy` - GPU-to-GPU copy
- `buffer-destroy` - Free GPU memory
- `sync` - Wait for GPU completion
- `get-device-info` - Query GPU capabilities

### `ml-kernels` - High-level ML operations

Pre-defined, optimized kernels for ML workloads:

| Kernel | Description | CUDA Backend | WebGPU Backend |
|--------|-------------|--------------|----------------|
| `kernel-bootstrap-sample` | Random sampling with replacement | cuRAND | PCG shader |
| `kernel-find-split` | Decision tree split finding | Custom reduction | WGSL reduction |
| `kernel-average` | Tree prediction averaging | Parallel reduce | WGSL reduce |
| `kernel-matmul` | Matrix multiplication | cuBLAS SGEMM | Tiled WGSL |
| `kernel-elementwise` | Element-wise ops (ReLU, etc.) | Custom kernel | WGSL shader |
| `kernel-reduce` | Array reductions (sum, max, etc.) | CUB reduce | WGSL reduce |
| `kernel-batch-predict` | Full forest inference | Warp-parallel | Per-sample parallel |

## Usage

### For WASM Module Authors (Rust)

```rust
// Cargo.toml
[dependencies]
wit-bindgen = "0.24"

// src/lib.rs
wit_bindgen::generate!({
    world: "ml-compute",
    path: "../wasi-gpu/wit",
});

use crate::wasi::gpu::compute::*;
use crate::wasi::gpu::ml_kernels::*;

fn train_tree(data: &[f32], labels: &[f32]) {
    // Create buffers
    let data_buf = buffer_create(data.len() as u64 * 4, BufferUsage::STORAGE).unwrap();
    buffer_write(data_buf, 0, bytemuck::cast_slice(data)).unwrap();
    
    // Run kernel
    let params = BootstrapParams { n_samples: 1000, seed: 42, max_index: 10000 };
    let output_buf = buffer_create(1000 * 4, BufferUsage::STORAGE).unwrap();
    kernel_bootstrap_sample(params, output_buf).unwrap();
    
    // Read results
    let result_bytes = buffer_read(output_buf, 0, 1000 * 4).unwrap();
    let indices: Vec<u32> = bytemuck::cast_slice(&result_bytes).to_vec();
}
```

### For WASM Module Authors (C++)

```cpp
// Use wit-bindgen to generate bindings
#include "wasi_gpu.h"

void train_tree(const float* data, size_t n_samples) {
    // Create buffer
    auto data_buf = wasi_gpu_compute_buffer_create(n_samples * sizeof(float), BUFFER_USAGE_STORAGE);
    wasi_gpu_compute_buffer_write(data_buf, 0, data, n_samples * sizeof(float));
    
    // Run kernel  
    bootstrap_params_t params = { .n_samples = 1000, .seed = 42, .max_index = n_samples };
    auto output_buf = wasi_gpu_compute_buffer_create(1000 * sizeof(uint32_t), BUFFER_USAGE_STORAGE);
    wasi_gpu_ml_kernels_kernel_bootstrap_sample(&params, output_buf);
    
    // Read results
    uint32_t indices[1000];
    wasi_gpu_compute_buffer_read(output_buf, 0, sizeof(indices), indices);
}
```

### For Host Runtime Implementers

Implement the `GpuBackend` trait:

```rust
pub trait GpuBackend: Send + Sync {
    fn get_device_info(&self) -> DeviceInfo;
    fn buffer_create(&self, size: u64, usage: BufferUsage) -> Result<BufferId, GpuError>;
    fn buffer_write(&self, id: BufferId, offset: u64, data: &[u8]) -> Result<(), GpuError>;
    fn buffer_read(&self, id: BufferId, offset: u64, size: u32) -> Result<Vec<u8>, GpuError>;
    fn kernel_bootstrap_sample(&self, params: BootstrapParams, output: BufferId) -> Result<(), GpuError>;
    fn kernel_find_split(&self, params: FindSplitParams, ...) -> Result<(), GpuError>;
    // ... other kernels
}
```

## Backend Implementations

### WebGPU Backend (Vulkan)

Uses `wgpu` crate with WGSL shaders. Works on systems with working Vulkan drivers.

### CUDA Backend

Uses CUDA runtime API with:
- cuBLAS for matrix operations
- cuRAND for random number generation
- Custom CUDA kernels for ML-specific operations
- Tensor Core support on H100

## File Structure

```
wasi-gpu/
├── wit/
│   ├── world.wit          # World definition
│   ├── compute.wit        # Low-level compute interface
│   └── ml-kernels.wit     # High-level ML kernels
├── README.md
└── LICENSE
```

## Versioning

This interface follows semantic versioning. Current version: `0.1.0`

## License

Apache-2.0
