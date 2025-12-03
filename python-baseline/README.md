# Python Baseline Benchmark

Native Python implementation with cuML/RAPIDS for GPU acceleration, using the **same TEE attestation** as the WASM modules.

## Architecture

```
python-baseline/
├── src/
│   └── main.py              # Benchmark entry point
├── tee_attestation/         # PyO3 bindings for Rust TEE attestation
│   ├── Cargo.toml
│   ├── pyproject.toml
│   └── src/
│       └── lib.rs           # Rust -> Python bindings
├── build.sh                 # Build attestation module
├── run.sh                   # Run benchmark
└── requirements.txt
```

## Key Feature: Unified Attestation

The `tee_attestation` Python module uses **PyO3** to call the same Rust attestation code used by the WASM host runtime. This ensures:

- **Fair comparison**: All three implementations (Python, Rust WASM, C++ WASM) use identical attestation logic
- **Same output format**: JSON results are structured identically
- **Same TEE support**: AMD SEV-SNP, Intel TDX, NVIDIA GPU CC

## Building

```bash
# Install maturin (PyO3 build tool)
pip install maturin

# Build the attestation module
./build.sh
```

## Running

```bash
# Run benchmark
./run.sh

# Or directly
python3 src/main.py
```

## Output Format

Output matches WASM modules exactly:

```
╔══════════════════════════════════════════════════════════╗
║   Python Baseline Benchmark - Diabetes Prediction        ║
║   Native Python + cuML/RAPIDS + TEE Attestation          ║
╚══════════════════════════════════════════════════════════╝

=== GPU INFORMATION ===
[GPU] Device: NVIDIA H100 NVL
[GPU] Backend: cuda
[GPU] Memory: 95830 MB
[GPU] Hardware: YES ✓

=== TEE ATTESTATION ===
[TEE] Type: AMD SEV-SNP
[TEE] Supports attestation: YES
[TEE] VM attestation: OK (token: 16980 chars)
[TEE] GPU attestation: OK (token: 576 chars)
[TIMING] Attestation: 3500.00 ms

=== TRAINING ===
[TRAIN] Dataset: 353 samples, 10 features
[TRAIN] Model: RandomForest (200 trees, depth 16)
[TRAIN] Accelerator: GPU (cuML/RAPIDS)
[TIMING] Training: 500.00 ms

=== INFERENCE ===
[INFER] Test set: 89 samples
[INFER] Accelerator: GPU (cuML/RAPIDS)
[TIMING] Inference: 10.00 ms
[INFER] MSE: 2850.0000

### BENCHMARK_JSON ###
{"language":"python","gpu_device":"NVIDIA H100 NVL",...}
### END_BENCHMARK_JSON ###
```

## Dependencies

- **Python 3.9+**
- **Rust** (for building PyO3 bindings)
- **maturin** (`pip install maturin`)
- **cuML/RAPIDS** (for GPU acceleration, optional - falls back to scikit-learn)
- **scikit-learn** (CPU fallback)

## Comparison with WASM Modules

| Feature | Python Baseline | Rust WASM | C++ WASM |
|---------|----------------|-----------|----------|
| ML Framework | cuML/RAPIDS | Custom | Custom |
| GPU Backend | CUDA (native) | wasi:gpu | wasi:gpu |
| TEE Attestation | PyO3→Rust | wasmtime host | wasmtime host |
| Output Format | JSON | JSON | JSON |
