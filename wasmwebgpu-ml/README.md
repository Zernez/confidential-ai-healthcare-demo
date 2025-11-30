# wasmwebgpu-ml

C++ implementation of the ML module with GPU acceleration via `wasi:gpu` interface.

## Architecture

```
┌─────────────────────────────────────┐
│    wasmwebgpu-ml (C++ WASM)         │
│  ┌─────────────────────────────────┐│
│  │ RandomForest / DecisionTree     ││
│  └────────────┬────────────────────┘│
│               │                      │
│  ┌────────────▼────────────────────┐│
│  │ GpuExecutor / GpuTrainer        ││
│  │ (wasi_gpu.h bindings)           ││
│  └────────────┬────────────────────┘│
└───────────────┼─────────────────────┘
                │ wasi:gpu imports
                ▼
┌─────────────────────────────────────┐
│      wasmtime-gpu-host              │
│  ┌─────────────┐ ┌───────────────┐  │
│  │   WebGPU    │ │     CUDA      │  │
│  │  (Vulkan)   │ │   (cuBLAS)    │  │
│  └─────────────┘ └───────────────┘  │
└─────────────────────────────────────┘
```

## Key Files

| File | Description |
|------|-------------|
| `src/wasi_gpu.h` | C bindings for wasi:gpu host functions |
| `src/gpu_executor.hpp/cpp` | High-level GPU wrapper (GpuExecutor, GpuTrainer, GpuPredictor) |
| `src/random_forest.hpp/cpp` | RandomForest implementation with GPU support |
| `src/main.cpp` | Benchmark application |
| `wit/*.wit` | WIT interface definitions |

## wasi:gpu Interface

The module imports two interfaces:

### `wasi:gpu/compute`

Low-level buffer management:
- `buffer-create` / `buffer-destroy`
- `buffer-write` / `buffer-read`
- `buffer-copy`
- `sync`
- `get-device-info`

### `wasi:gpu/ml-kernels`

High-level ML operations:
- `kernel-bootstrap-sample` - Random sampling with replacement
- `kernel-find-split` - Decision tree split finding
- `kernel-average` - Tree prediction averaging
- `kernel-matmul` - Matrix multiplication (cuBLAS/WGSL)
- `kernel-elementwise` - ReLU, sigmoid, etc.
- `kernel-reduce` - Sum, max, mean, variance

## Building

```bash
# Requires clang with wasm32-wasi target
mkdir build && cd build
cmake -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ ..
make

# Output: wasmwebgpu-ml-benchmark.wasm
```

## Running

```bash
# With wasmtime-gpu-host (from parent directory)
cd ../wasmtime-gpu-host
cargo run --release -- ../wasmwebgpu-ml/build/wasmwebgpu-ml-benchmark.wasm
```

## CPU Fallback

All GPU operations have CPU fallback implementations. If the host doesn't provide `wasi:gpu` or if GPU initialization fails, the module automatically falls back to CPU.

## Comparison with Rust Module

| Feature | wasm-ml (Rust) | wasmwebgpu-ml (C++) |
|---------|----------------|---------------------|
| Language | Rust | C++17 |
| Bindings | wit-bindgen | Manual C extern |
| Size | ~2 MB | ~500 KB |
| Compile | cargo component | clang --target=wasm32-wasi |
| Same wasi:gpu interface | ✓ | ✓ |
