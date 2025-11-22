# wasmtime:attestation - TEE Runtime Extension

## Overview

`wasmtime:attestation` is a custom runtime extension for Wasmtime that provides **hardware-backed attestation** capabilities to WebAssembly guest modules. It enables confidential computing workloads to cryptographically verify both:
- **VM integrity** (Intel TDX / AMD SEV-SNP)
- **GPU integrity** (NVIDIA H100 via NRAS)

## Architecture

```
┌─────────────────────────────────────────────────────┐
│ WASM Guest Module                                   │
│  ├── Calls: attest_vm()                            │
│  ├── Calls: attest_gpu(gpu_index)                  │
│  └── Calls: verify_token(token)                    │
└─────────────────────────────────────────────────────┘
                    ↓ Host Functions
┌─────────────────────────────────────────────────────┐
│ Wasmtime Runtime (wasmtime-webgpu-host)            │
│  ├── TeeHost (attestation logic)                   │
│  │   ├── Intel TDX quote generation                │
│  │   ├── AMD SEV-SNP report generation             │
│  │   └── NVIDIA GPU NRAS attestation               │
│  ├── WebGpuHost (GPU compute)                      │
│  └── WasiCtx (standard WASI)                       │
└─────────────────────────────────────────────────────┘
                    ↓ System Calls
┌─────────────────────────────────────────────────────┐
│ Hardware & Drivers                                  │
│  ├── /dev/tdx_guest (TDX)                          │
│  ├── /dev/sev-guest (SEV-SNP)                      │
│  └── nvattest CLI / NVML (GPU)                     │
└─────────────────────────────────────────────────────┘
```

## Features

✅ **Multi-TEE Support**: Intel TDX, AMD SEV-SNP  
✅ **GPU Attestation**: NVIDIA H100 via NRAS  
✅ **Token Caching**: Avoid redundant attestations  
✅ **JSON Results**: Easy parsing for guests  
✅ **Multi-language**: Rust and C++ bindings  
✅ **Error Handling**: Detailed error messages  

## Host Functions

### 1. `attest_vm() -> JSON`
Generates a cryptographic attestation of the VM state.

**Returns**: `AttestationResult` JSON
```json
{
  "success": true,
  "token": "eyJhbGc...",
  "timestamp": 1763735568
}
```

**Supported TEEs**:
- Intel TDX (via `/dev/tdx_guest`)
- AMD SEV-SNP (via `/dev/sev-guest`)

---

### 2. `attest_gpu(gpu_index: u32) -> JSON`
Generates a cryptographic attestation of the GPU hardware state.

**Parameters**:
- `gpu_index`: GPU device index (0-based)

**Returns**: `AttestationResult` JSON
```json
{
  "success": true,
  "token": "eyJhbGc...",
  "timestamp": 1763735568
}
```

**Requirements**:
- NVIDIA H100 GPU in CC mode
- Driver R580+ with attestation support
- `nvattest` CLI or NVML library

---

### 3. `verify_token(token_ptr: *const u8, token_len: i32) -> bool`
Performs basic JWT token validation.

**Parameters**:
- `token_ptr`: Pointer to token string in WASM memory
- `token_len`: Length of token string

**Returns**: `1` if valid, `0` if invalid

**Note**: Currently performs structure validation only. Full signature verification requires public key from attestation service.

---

### 4. `clear_cache()`
Clears all cached attestation tokens.

**Use case**: Force re-attestation after VM/GPU state change.

---

## Guest Usage

### Rust (wasm-ml)

```rust
use wasm_ml::attestation::{attest_vm_token, attest_gpu_token, verify_attestation_token};

fn main() -> Result<(), Box<dyn Error>> {
    // Attest VM
    let vm_result = attest_vm_token()?;
    println!("VM token: {}", vm_result.token.unwrap());
    
    // Attest GPU
    let gpu_result = attest_gpu_token(0)?;
    println!("GPU token: {}", gpu_result.token.unwrap());
    
    // Verify tokens
    assert!(verify_attestation_token(&vm_result.token.unwrap()));
    assert!(verify_attestation_token(&gpu_result.token.unwrap()));
    
    // Proceed with ML workload
    run_ml_training();
    
    Ok(())
}
```

See: `wasm-ml/examples/attestation_example.rs`

---

### C++ (wasmwebgpu-ml)

```cpp
#include "attestation.hpp"

int main() {
    // Run full attestation workflow
    if (!wasmtime_attestation::attest_all(0)) {
        fprintf(stderr, "Attestation failed!\n");
        return 1;
    }
    
    // Proceed with ML workload
    run_ml_training();
    
    return 0;
}
```

See: `wasmwebgpu-ml/examples/main_with_attestation.cpp`

---

## Building

### Runtime (wasmtime-webgpu-host)

```bash
cd wasmtime-webgpu-host

# Compile with all attestation features
cargo build --release --features attestation-tdx,attestation,attestation-nvidia

# Run with WASM module
./target/release/wasmtime-webgpu-host \
    ../wasm-ml/target/wasm32-wasi/release/wasm_ml.wasm \
    --dir=../data
```

---

### Rust Guest (wasm-ml)

```bash
cd wasm-ml

# Build example with attestation
cargo build --release --target wasm32-wasi --example attestation_example

# Run with attestation-enabled runtime
../wasmtime-webgpu-host/target/release/wasmtime-webgpu-host \
    target/wasm32-wasi/release/examples/attestation_example.wasm \
    --dir=../data
```

---

### C++ Guest (wasmwebgpu-ml)

```bash
cd wasmwebgpu-ml

# Build with attestation example
mkdir -p build && cd build
cmake .. -DBUILD_WASM=ON
make

# Run
../../wasmtime-webgpu-host/target/release/wasmtime-webgpu-host \
    wasmwebgpu-ml-benchmark.wasm \
    --dir=../../data
```

---

## Testing

### On Development Machine (No TEE)

```bash
# Expected: Attestation will fail gracefully
./wasmtime-webgpu-host wasm_ml.wasm

# Output:
# ❌ No TEE attestation available (neither TDX nor SEV-SNP)
# ❌ VM attestation failed: No TEE available. Not running in confidential VM?
```

---

### On Azure Confidential VM (TDX/SEV-SNP)

```bash
# Should succeed with VM attestation
./wasmtime-webgpu-host wasm_ml.wasm

# Output:
# ✓ TDX attestation successful
# ✓ VM token verified!
```

---

### On Azure H100 VM (with GPU)

```bash
# Full attestation: VM + GPU
./wasmtime-webgpu-host wasm_ml.wasm

# Output:
# ✓ TDX attestation successful
# ✓ GPU attestation successful
# ✓ All attestations passed!
```

---

## Security Considerations

### Token Verification
Currently, `verify_token()` performs **structure validation only** (checks JWT format). For production use, implement:
- Signature verification with NRAS/Azure public keys
- Expiration checking
- Nonce validation
- Policy enforcement

### Caching
Tokens are cached to avoid redundant attestations. Clear cache if:
- VM state changes (e.g., measurements updated)
- GPU firmware updated
- Security policy requires fresh attestation

### Error Handling
Always check `success` field in `AttestationResult`:
```rust
let result = attest_vm_token()?;
if !result.success {
    return Err(result.error.unwrap_or("Unknown error".into()));
}
```

---

## Troubleshooting

### "No TEE attestation available"
**Cause**: Not running in a confidential VM  
**Solution**: Deploy to Azure DCasv5 VM with TDX/SEV-SNP enabled

### "GPU attestation failed"
**Cause**: GPU not in CC mode or driver issue  
**Solution**:
```bash
# Check GPU CC mode
nvidia-smi -q | grep 'Conf Compute'

# Check driver version (must be R580+)
nvidia-smi --query-gpu=driver_version --format=csv

# Verify nvattest CLI
nvattest --version
```

### "Failed to get WASM memory export"
**Cause**: Memory layout mismatch  
**Solution**: Ensure WASM module exports `memory` and uses linear memory

---

## Roadmap

- [ ] Full JWT signature verification
- [ ] Remote attestation service integration (Azure Attestation)
- [ ] Policy-based attestation (accept/reject based on measurements)
- [ ] Attestation report caching to disk
- [ ] Multi-GPU attestation
- [ ] Attestation event logging

---

## References

- [Intel TDX](https://www.intel.com/content/www/us/en/developer/tools/trust-domain-extensions/overview.html)
- [AMD SEV-SNP](https://www.amd.com/en/developer/sev.html)
- [NVIDIA Confidential Computing](https://docs.nvidia.com/confidential-computing/)
- [NVIDIA NRAS](https://docs.nvidia.com/attestation/)
- [Azure Confidential VMs](https://learn.microsoft.com/azure/confidential-computing/)

---

## License

Same as parent project.
