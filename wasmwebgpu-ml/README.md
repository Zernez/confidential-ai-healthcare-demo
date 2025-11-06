# wasmwebgpu-ml - C++ + wasi:webgpu ML Benchmark

RandomForest implementation in C++ using wasi:webgpu for GPU acceleration, compiled to WebAssembly.

## 🎯 Purpose

This is a C++ implementation of the same ML pipeline as the Rust version (`wasm-ml/`) and Python version, designed for fair benchmarking of different approaches:

- **Python (RAPIDS)**: Native CUDA acceleration
- **Rust (wgpu)**: WebAssembly with wgpu library
- **C++ (wasi:webgpu)**: WebAssembly with wasi:webgpu standard ← *This project*

## 📊 Algorithm

- **Model**: RandomForest Regressor
- **Dataset**: Diabetes (442 samples, 10 features)
- **Parameters**: 
  - 200 trees
  - Max depth: 16
  - Train/test split: 80/20

## 🏗️ Architecture

```
wasmwebgpu-ml/
├── src/
│   ├── main.cpp              # Entry point
│   ├── dataset.hpp/cpp       # CSV loading, bootstrap sampling
│   ├── random_forest.hpp/cpp # RandomForest + DecisionTree
│   └── gpu_executor.hpp/cpp  # wasi:webgpu GPU interface
├── shaders/
│   ├── average.wgsl          # GPU prediction averaging
│   ├── bootstrap_sample.wgsl # GPU bootstrap sampling
│   └── find_split.wgsl       # GPU split finding
├── external/
│   ├── json.hpp              # nlohmann/json (header-only)
│   ├── csv.h                 # fast-cpp-csv-parser (header-only)
│   └── wasi/webgpu.h         # wasi:webgpu C bindings
└── CMakeLists.txt            # Build configuration
```

## 🚀 Building

### Prerequisites

1. **WASI SDK** (automatically installed by setup script)
2. **CMake** 3.20+
3. **WASM Runtime** (wasmtime or wasmer)

### Setup Environment

```bash
# From project root
./setup_wasi_cpp.sh

# Activate environment
source wasmwebgpu-ml/env.sh
```

### Build

```bash
cd wasmwebgpu-ml
./build.sh
```

### Run Benchmark

```bash
# From project root
./run_wasmwebgpu_benchmark.sh
```

## 🔧 Technical Details

### Language Features

- **C++17** standard
- **Header-only libraries** (zero external dependencies except toolchain)
- **WASI Preview 2** target (`wasm32-wasip2`)
- **WebGPU Compute Shaders** (WGSL)

### Optimizations

- `-O3` optimization level
- Link-Time Optimization (LTO)
- Dead code elimination
- Symbol stripping

### GPU Acceleration

The implementation uses wasi:webgpu for GPU operations:

1. **Training**:
   - Bootstrap sampling on GPU (parallel random sampling)
   - Split finding on GPU (parallel MSE computation)
   
2. **Inference**:
   - Tree prediction averaging on GPU (parallel averaging)

## 📈 Benchmarking

The implementation is designed for fair comparison:

- **Same dataset** (shared CSV files)
- **Same parameters** (200 trees, depth 16)
- **Same metrics** (MSE on test set)
- **Consistent methodology**

Output includes:
- Training time
- Inference time  
- Mean Squared Error
- Model size

## 🔍 Code Organization

### Core ML Components

- **`Dataset`**: CSV loading, data management, bootstrap sampling
- **`TreeNode`**: Decision tree node (leaf/internal)
- **`DecisionTree`**: Single decision tree with split finding
- **`RandomForest`**: Ensemble of trees with training/prediction

### GPU Integration

- **`GpuExecutor`**: WebGPU device management
- **WGSL Shaders**: Compute kernels for GPU operations
- **Async Operations**: Promise/future pattern for GPU calls

### Serialization

- **JSON format** (nlohmann/json)
- Human-readable
- Cross-platform compatible

## ⚙️ Configuration

Edit `src/main.cpp` to change:

```cpp
constexpr size_t N_ESTIMATORS = 200;  // Number of trees
constexpr size_t MAX_DEPTH = 16;       // Tree depth
constexpr size_t N_FEATURES = 10;      // Input features
```

## 🧪 Testing

### CPU-only Build (for debugging)

```bash
cd wasmwebgpu-ml/build
cmake .. -DBUILD_WASM=OFF -DBUILD_NATIVE=ON
make
./wasmwebgpu-ml-benchmark
```

### Verify Output

Expected MSE: ~3000-3500 (similar to Python/Rust versions)

## 📚 Dependencies

All dependencies are header-only:

- **[nlohmann/json](https://github.com/nlohmann/json)**: JSON serialization (v3.11.3)
- **[fast-cpp-csv-parser](https://github.com/ben-strasser/fast-cpp-csv-parser)**: CSV parsing
- **wasi:webgpu**: WebGPU C bindings (generated from WIT)

## 🐛 Debugging

Enable verbose output:
```bash
export WASMTIME_LOG=debug
wasmtime run --dir=. build/wasmwebgpu-ml-benchmark.wasm
```

Inspect WASM binary:
```bash
wasm-objdump -x build/wasmwebgpu-ml-benchmark.wasm
```

## 📊 Performance Notes

### Current Status

- ✅ CPU training implemented
- ✅ CPU inference implemented  
- 🚧 GPU training (wasi:webgpu integration in progress)
- 🚧 GPU inference (wasi:webgpu integration in progress)

### Expected Performance

Based on similar workloads:
- **Binary size**: ~200-500 KB (optimized)
- **Training time**: 2-5x slower than native (CPU)
- **Inference time**: <10ms for 88 samples
- **GPU speedup**: 2-10x vs CPU (when GPU enabled)

## 🔬 Comparison with Other Implementations

| Feature | Python (RAPIDS) | Rust (wgpu) | C++ (wasi:webgpu) |
|---------|----------------|-------------|-------------------|
| Language | Python | Rust | C++ |
| GPU API | CUDA | wgpu library | wasi:webgpu standard |
| Target | Native x86_64 | WASM (wasm32-wasi) | WASM (wasm32-wasip2) |
| Runtime | Direct | wasmtime | wasmtime/wasmer |
| Dependencies | Many (cuML, etc.) | Medium (wgpu crate) | Minimal (header-only) |
| Portability | NVIDIA GPUs only | Any GPU (via wgpu) | Any GPU (via WebGPU) |

## 🚦 Status

**Current Phase**: Implementation Complete (CPU), GPU Integration In Progress

- [x] Project structure
- [x] Dataset loading
- [x] RandomForest CPU implementation
- [x] Build system (CMake + WASI SDK)
- [x] Benchmarking scripts
- [ ] wasi:webgpu GPU integration
- [ ] Full testing on H100
- [ ] Performance optimization

## 📖 References

- [WASI WebGPU Proposal](https://github.com/WebAssembly/wasi-gfx)
- [WebGPU Specification](https://www.w3.org/TR/webgpu/)
- [WASI SDK](https://github.com/WebAssembly/wasi-sdk)
- [wasmtime](https://wasmtime.dev/)

## 🤝 Contributing

This is a benchmarking project. The goal is to maintain parity with the Python and Rust implementations for fair comparison.

When making changes:
1. Keep algorithm identical to other implementations
2. Maintain same parameters (n_estimators, max_depth, etc.)
3. Use same dataset (diabetes_train.csv, diabetes_test.csv)
4. Document any deviations

## 📝 License

Same as parent project.

## 🔗 Related Projects

- `../python-native/`: Native Python RAPIDS implementation
- `../wasm-ml/`: Rust + wgpu WebAssembly implementation
- `../docker/`: Docker containers for deployment
- `../infrastructure/`: Azure deployment configuration
