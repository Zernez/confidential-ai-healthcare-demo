# Guida Implementazione WASM + WebGPU

## 📋 Panoramica

Questa implementazione integra RandomForest con accelerazione WebGPU nel progetto esistente.

```
conf-ai-healthcare-demo/
├── wasm-ml/              ← Nuovo modulo Rust WASM
│   ├── src/
│   │   ├── lib.rs       ← Entry point WASM
│   │   ├── random_forest.rs  ← Implementazione RF
│   │   ├── gpu_compute.rs    ← WebGPU compute
│   │   └── data.rs      ← Dataset handling
│   ├── shaders/
│   │   └── average.wgsl ← GPU shader per averaging
│   └── Cargo.toml       ← Config Rust
├── wasm_wrapper.py       ← Python wrapper
├── build_wasm.ps1        ← Build script Windows
└── docker/
    └── Dockerfile.wasm   ← Docker con WASM runtime
```

## 🚀 Setup Locale (Windows 11)

### 1. Installare Rust

```powershell
# Download e installa da https://rustup.rs
# Oppure via winget:
winget install Rustlang.Rustup
```

### 2. Configurare Toolchain

```powershell
# Aggiungere target WASM
rustup target add wasm32-wasi

# Verificare installazione
rustc --version
cargo --version
```

### 3. Build Modulo WASM

```powershell
# Navigare alla root del progetto
cd C:\Users\ferna\OneDrive\Documenti\ComputingContinuum\CPU+GPU\conf-ai-healthcare-demo

# Build in modalità release (ottimizzato)
.\build_wasm.ps1 -Release

# Oppure con test
.\build_wasm.ps1 -Release -Test
```

### 4. Verificare Build

```powershell
# Il file WASM sarà in:
# wasm-ml\target\wasm32-wasi\release\wasm_ml.wasm

# Verificare dimensione (dovrebbe essere <500KB in release)
Get-Item wasm-ml\target\wasm32-wasi\release\wasm_ml.wasm | Select-Object Length
```

## 🧪 Testing Locale

### Test Python Wrapper

```powershell
# Test base
python wasm_wrapper.py

# Test con comparison RAPIDS (se disponibile)
python -c "from wasm_wrapper import compare_with_rapids; compare_with_rapids()"
```

### Test Rust (senza WASM)

```powershell
cd wasm-ml
cargo test
```

## 🐳 Build Docker Image

### Build immagine con WASM runtime

```powershell
docker build -f docker/Dockerfile.wasm -t wasm-ml:latest .
```

### Test locale con Docker

```powershell
# Run container
docker run --gpus all -it wasm-ml:latest

# Dentro il container:
wasmtime --version
python3 wasm_wrapper.py
```

## ☁️ Deploy su Azure H100 VM

### 1. Preparazione

```powershell
# Login Azure
az login

# Seleziona subscription
az account set --subscription "YOUR_SUBSCRIPTION_ID"
```

### 2. Push Docker Image ad Azure Container Registry

```powershell
# Crea ACR (se non esiste)
az acr create --resource-group YOUR_RG --name YOUR_ACR --sku Premium

# Login ad ACR
az acr login --name YOUR_ACR

# Tag e push image
docker tag wasm-ml:latest YOUR_ACR.azurecr.io/wasm-ml:latest
docker push YOUR_ACR.azurecr.io/wasm-ml:latest
```

### 3. Deploy su VM

```powershell
# Usa script esistente modificato
# .\infrastructure\deploy.ps1 verrà aggiornato per supportare WASM
```

## 📊 Architettura di Esecuzione

### Flusso Training + Inferenza

```
┌─────────────────────────────────────────────┐
│  Python Script (main.py)                    │
│  └── train_model() / infer_model()          │
└──────────────┬──────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────┐
│  WASM Wrapper (wasm_wrapper.py)             │
│  └── WasmRandomForest                       │
└──────────────┬──────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────┐
│  WASM Runtime (wasmtime)                    │
│  └── Esegue wasm_ml.wasm                    │
└──────────────┬──────────────────────────────┘
               │
               ├──> Training (CPU)
               │    └── RandomForest::train()
               │
               └──> Inferenza (GPU)
                    └── GpuExecutor::predict()
                         └── WebGPU Compute Shader
                              └── H100 GPU
```

## 🔧 Configurazione WebGPU

### Enable WebGPU su Azure VM

```bash
# Verificare supporto GPU
nvidia-smi

# Installare driver Vulkan (per WebGPU backend)
sudo apt-get update
sudo apt-get install -y vulkan-tools libvulkan1

# Verificare Vulkan
vulkaninfo
```

### Test WebGPU

```python
# test_webgpu.py
from wasm_wrapper import WasmRandomForest
import numpy as np

# Dati di test
X = np.random.randn(100, 10).astype(np.float32)
y = np.random.randn(100).astype(np.float32)

rf = WasmRandomForest()
rf.train(X, y)

# Test GPU inference
predictions_gpu = rf.predict_gpu(X)
print(f"GPU predictions: {predictions_gpu[:5]}")

# Confronto con CPU
predictions_cpu = rf.predict_cpu(X)
print(f"CPU predictions: {predictions_cpu[:5]}")
```

## 📈 Performance Attese

### Hardware: Azure NC24ads A100 v4

| Operazione | CPU | GPU (WebGPU) | Speedup |
|------------|-----|--------------|---------|
| Training (200 trees) | 3-5s | N/A (CPU only) | - |
| Inference (100 samples) | 50ms | 5-10ms | 5-10x |
| Inference (1000 samples) | 500ms | 15-30ms | 15-30x |

### Ottimizzazioni

1. **Batch Inference**: Processa più sample contemporaneamente
2. **Tree Parallelization**: Esegui predizioni su alberi diversi in parallelo
3. **Memory Pooling**: Riusa buffer GPU tra inferenze

## 🐛 Troubleshooting

### Build Errors

```powershell
# Errore: "linking with `rust-lld` failed"
# Soluzione: Reinstalla Rust toolchain
rustup update
rustup target remove wasm32-wasi
rustup target add wasm32-wasi
```

### Runtime Errors

```powershell
# Errore: "WebGPU not available"
# Verifica:
1. Driver GPU aggiornati
2. Vulkan installato
3. Permessi GPU corretti
```

### Performance Issues

```python
# Se inferenza GPU non è più veloce di CPU:
# 1. Verifica batch size (minimo 50 samples)
# 2. Check GPU utilization: nvidia-smi -l 1
# 3. Profila con: wasmtime run --profile ...
```

## 📚 Risorse

- [WASI Spec](https://github.com/WebAssembly/WASI)
- [WebGPU Spec](https://www.w3.org/TR/webgpu/)
- [wgpu-rs](https://github.com/gfx-rs/wgpu)
- [Wasmtime](https://wasmtime.dev/)

## 🔄 Prossimi Passi

1. ✅ **Fase 1 completata**: Setup base Rust + WASI
2. ⏳ **Fase 2**: Testing completo e benchmarking
3. 📋 **Fase 3**: Integrazione con codice Python esistente
4. 🚀 **Fase 4**: Deploy e validazione su Azure H100

## 💬 Note

- La versione attuale implementa averaging su GPU
- Tree traversal è ancora su CPU (futura ottimizzazione)
- Performance migliori con batch grandi (>50 samples)
