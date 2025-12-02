# TEE Attestation Python Bindings

PyO3 bindings for TEE attestation, providing the same functionality as the WASM host runtime.

## Building

```bash
# Requires Rust and maturin
pip install maturin
maturin develop --release
```

## Usage

```python
import tee_attestation

# Detect TEE type
info = tee_attestation.detect_tee()
print(f"TEE: {info.tee_type}, Attestation: {info.supports_attestation}")

# VM attestation (AMD SEV-SNP)
result = tee_attestation.attest_vm()
if result.success:
    print(f"Token: {result.token_length} chars")

# GPU attestation (NVIDIA CC)
result = tee_attestation.attest_gpu(0)
if result.success:
    print(f"GPU Token: {result.token_length} chars")

# Get GPU info
name, backend, memory = tee_attestation.get_gpu_info()
print(f"GPU: {name}, {memory} MB")
```

## Features

- **AMD SEV-SNP**: VM attestation via vTPM
- **Intel TDX**: Optional, enable with `--features attestation-tdx`
- **NVIDIA GPU CC**: Local attestation via `nvattest` CLI
