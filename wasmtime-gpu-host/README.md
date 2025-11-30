# wasmtime-gpu-host

Wasmtime host runtime che implementa l'interfaccia `wasi:gpu` per moduli WASM.

## Architettura

```
┌─────────────────────────────────────────────────────────────┐
│                    wasmtime-gpu-host                         │
│  ┌─────────────────────────────────────────────────────────┐│
│  │                    Wasmtime Engine                       ││
│  │  ┌───────────────┐  ┌───────────────────────────────┐   ││
│  │  │   WASI P1     │  │   wasi:gpu Host Functions     │   ││
│  │  │  (stdio,fs)   │  │  (buffer, ml-kernels)         │   ││
│  │  └───────────────┘  └───────────────────────────────┘   ││
│  └─────────────────────────────────────────────────────────┘│
│                              │                               │
│  ┌───────────────────────────▼───────────────────────────┐  │
│  │                    GpuBackend Trait                    │  │
│  └───────────────────────────┬───────────────────────────┘  │
│              ┌───────────────┴───────────────┐              │
│              ▼                               ▼              │
│  ┌───────────────────────┐   ┌───────────────────────────┐  │
│  │     CudaBackend       │   │     WebGpuBackend         │  │
│  │  ┌─────────────────┐  │   │  ┌─────────────────────┐  │  │
│  │  │  cuBLAS (SGEMM) │  │   │  │   wgpu (Vulkan)     │  │  │
│  │  │  cuRAND         │  │   │  │   WGSL shaders      │  │  │
│  │  │  PTX kernels    │  │   │  └─────────────────────┘  │  │
│  │  └─────────────────┘  │   └───────────────────────────┘  │
│  └───────────────────────┘                                  │
└─────────────────────────────────────────────────────────────┘
              │                               
              ▼                               
       ┌─────────────┐                
       │ NVIDIA H100 │                
       │   (CUDA)    │                
       │ Tensor Cores│                
       └─────────────┘                
```

## Implementazione GPU

### CudaBackend - Operazioni su GPU

| Kernel | Implementazione | Libreria |
|--------|-----------------|----------|
| `bootstrap_sample` | PTX kernel custom | CUDA (sm_80+) |
| `average_predictions` | PTX kernel custom | CUDA |
| `matmul` | **cuBLAS SGEMM** | Tensor Cores su H100 |
| `relu` | PTX kernel custom | CUDA |
| `sigmoid` | PTX kernel custom | CUDA |
| `reduce_sum` | PTX kernel custom | CUDA |
| `find_split` | cuBLAS dot products | Hybrid GPU/CPU |
| `add` | cuBLAS axpy | cuBLAS |

### PTX Kernels Embedded

I kernel CUDA sono scritti in PTX e compilati inline:

```ptx
// Bootstrap sample - PCG hash su GPU
.visible .entry bootstrap_sample(...)

// Average predictions - parallel reduction
.visible .entry average_predictions(...)

// ReLU activation
.visible .entry relu(...)

// Sigmoid activation  
.visible .entry sigmoid(...)

// Reduce sum
.visible .entry reduce_sum(...)
```

### cuBLAS per Tensor Cores

L'operazione `matmul` usa cuBLAS SGEMM che sfrutta automaticamente i Tensor Cores su H100:

```rust
self.blas.gemm(
    CUBLAS_OP_N, CUBLAS_OP_N,
    n, m, k,
    alpha,
    &b_f32, ldb,
    &a_f32, lda,
    beta,
    &mut c_f32, ldc,
)
```

## Usage

```bash
# Build con CUDA support
cargo build --release

# Esegui modulo WASM
./target/release/wasmtime-gpu-host module.wasm

# Forza backend specifico
./target/release/wasmtime-gpu-host --backend cuda module.wasm

# Con directory di lavoro
./target/release/wasmtime-gpu-host -d /path/to/data module.wasm

# Verbose logging
RUST_LOG=debug ./target/release/wasmtime-gpu-host module.wasm
```

## Build

### Requisiti

- CUDA Toolkit 11.x+ (per cudarc)
- Driver NVIDIA 525+ (per H100)
- Rust 1.70+

```bash
# Ubuntu - installa CUDA toolkit
sudo apt install nvidia-cuda-toolkit

# Build
cargo build --release
```

### Solo WebGPU (senza CUDA)

```bash
cargo build --release --no-default-features --features webgpu-only
```

## Performance su H100

Con CUDA backend su NVIDIA H100 NVL:

- **matmul 1024x1024**: ~0.5ms (Tensor Cores)
- **bootstrap_sample 10000**: ~0.1ms
- **average_predictions 1000x100**: ~0.05ms

## Configurazione

Variabili ambiente:

| Variabile | Descrizione |
|-----------|-------------|
| `RUST_LOG=debug` | Logging dettagliato |
| `CUDA_VISIBLE_DEVICES=0` | Seleziona GPU |

## File Structure

```
wasmtime-gpu-host/
├── Cargo.toml
├── src/
│   ├── main.rs           # CLI entry point
│   ├── host.rs           # wasi:gpu host functions
│   └── backend/
│       ├── mod.rs        # GpuBackend trait
│       ├── cuda.rs       # CudaBackend (cuBLAS/PTX)
│       └── webgpu.rs     # WebGpuBackend (wgpu)
└── wit/
    ├── world.wit
    ├── compute.wit
    └── ml-kernels.wit
```
