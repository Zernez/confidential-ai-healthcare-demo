# Quick Start - wasmtime:attestation

## 🚀 5-Minute Setup

### 1. Build Runtime (2 min)
```bash
cd wasmtime-webgpu-host
cargo build --release --features attestation-tdx,attestation,attestation-nvidia
```

### 2. Test Rust Example (1 min)
```bash
cd ../wasm-ml
cargo build --release --target wasm32-wasi --example attestation_example

# Run
../wasmtime-webgpu-host/target/release/wasmtime-webgpu-host \
    target/wasm32-wasi/release/examples/attestation_example.wasm
```

### 3. Test C++ Example (2 min)
```bash
cd ../wasmwebgpu-ml
# Add attestation.hpp to CMakeLists if needed
mkdir -p build && cd build
cmake .. -DBUILD_WASM=ON
make

# Run
../../wasmtime-webgpu-host/target/release/wasmtime-webgpu-host \
    wasmwebgpu-ml-benchmark.wasm
```

---

## 📝 Minimal Example

### Rust
```rust
use wasm_ml::attestation::{attest_vm_token, attest_gpu_token};

fn main() {
    // Attest
    let vm = attest_vm_token().expect("VM attestation failed");
    let gpu = attest_gpu_token(0).expect("GPU attestation failed");
    
    println!("✅ Attestation passed!");
    
    // Your ML code here
}
```

### C++
```cpp
#include "attestation.hpp"

int main() {
    // Attest
    if (!wasmtime_attestation::attest_all(0)) {
        return 1;
    }
    
    printf("✅ Attestation passed!\n");
    
    // Your ML code here
    return 0;
}
```

---

## 🐛 Troubleshooting

**"No TEE available"** → Normal on dev machine (no TDX/SEV-SNP)  
**"GPU attestation failed"** → Check `nvidia-smi`, driver version R580+  
**Compilation errors** → Make sure `attestation-rs` dependency is present  

---

## 📚 Full Documentation

See `ATTESTATION.md` for complete documentation.

---

## ✅ Expected Output

```
╔════════════════════════════════════════════════╗
║  Wasmtime with TEE Attestation + WebGPU       ║
║  • wasi:webgpu (GPU compute)                  ║
║  • wasmtime:attestation (VM + GPU)            ║
╚════════════════════════════════════════════════╝

Loading WASM: target/wasm32-wasi/release/examples/attestation_example.wasm
Initializing GPU backend...
GPU backend initialized
  GPU: NVIDIA H100 NVL
Initializing TEE attestation...
✓ wasi:webgpu functions registered
✓ wasmtime:attestation functions registered
Loading WASM module...
WASM module loaded
Instantiating module...
Running WASM...
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

╔════════════════════════════════════════════════╗
║  Confidential ML with TEE Attestation         ║
╚════════════════════════════════════════════════╝

━━━ Phase 1: Attestation ━━━

🔐 [1/4] Attesting VM (TDX/SEV-SNP)...
✓ VM attestation successful!
  Token length: 847 chars
  Timestamp: 1763735568

🔐 [2/4] Attesting GPU (NVIDIA H100)...
✓ GPU attestation successful!
  Token length: 1234 chars
  Timestamp: 1763735569

🔍 [3/4] Verifying VM token...
✓ VM token verified!

🔍 [4/4] Verifying GPU token...
✓ GPU token verified!

✅ All attestations passed! Proceeding with ML training...

━━━ Phase 2: ML Training ━━━
[Training output...]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WASM execution completed successfully
```

---

Happy Confidential Computing! 🔐🚀
