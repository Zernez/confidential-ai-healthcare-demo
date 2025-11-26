# Deprecated Scripts

These scripts are no longer needed and have been superseded by the updated build system.

## Why Deprecated?

| Old Script | Replaced By | Reason |
|------------|-------------|--------|
| `setup_wasi_cpp.sh` | `build_wasmwebgpu_ml.sh` | WASI SDK install is now integrated in build script |
| `setup_wasi_gfx.sh` | N/A | WIT bindings generation not needed; host functions are implemented in runtime |
| `setup_wasi_webgpu_beta.sh` | `build_all.sh` | Complete orchestration now handled by build_all.sh |

## Current Build System

Use these scripts instead:

```bash
# Setup dataset (sklearn diabetes)
./setup_data.sh

# Build everything
./build_all.sh

# Or build individually:
./build_webgpu_host.sh      # Runtime with attestation
./build_wasm.sh             # Rust WASM module
./build_wasmwebgpu_ml.sh    # C++ WASM module

# Run
./run_with_attestation.sh
```

## If You Need These Scripts

These scripts are kept for reference only. If you need specific functionality:

- **WASI SDK manual install**: See `build_wasmwebgpu_ml.sh` which auto-installs WASI SDK
- **WIT bindings**: The runtime (`wasmtime-webgpu-host`) already implements the host functions
- **C++ headers**: Run `build_wasmwebgpu_ml.sh` which downloads required headers

---
*Deprecated on: November 2025*
