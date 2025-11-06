# 📚 Guida agli Script - Python vs WASM

Questa guida spiega tutti gli script disponibili per eseguire i benchmark ML su Linux/Azure.

## 📊 Panoramica Script

### Python Nativo (RAPIDS/cuML)
- `run_local.sh` - Esecuzione diretta (richiede Conda + RAPIDS)
- `run.sh` - Esecuzione in Docker (build + run automatico)
- `run_python_benchmark.sh` - Benchmark pulito (rimuove modelli vecchi)

### WASM (Rust + WebGPU)
- `run_wasm_local.sh` - Esecuzione diretta (richiede Rust)
- `run_wasm_docker.sh` - Esecuzione in Docker (build + run automatico)
- `run_wasm_benchmark.sh` - Build completo + benchmark

### Utility
- `build_wasm.sh` - Build solo modulo WASM (no esecuzione)
- `export_diabetes_for_wasm.py` - Esporta dataset per WASM

---

## 🔧 Script Python Nativi

### 1. `run_local.sh` (Esecuzione Locale)

**Prerequisiti:**
- Conda installato
- Environment `rapids` attivato
- GPU NVIDIA con driver aggiornati

**Cosa fa:**
```bash
1. Esegue attestation (verifica sicurezza)
2. Lancia: python3 python-native/src/main.py
```

**Uso:**
```bash
# Attiva environment prima
conda activate rapids-25.10

# Esegui
./run_local.sh
```

**Output:**
```
[TRAINING] Avvio training su GPU (cuML RandomForest)...
[TRAINING] Completato.
[INFERENZA] Mean Squared Error (GPU): XXXX.XXXX
```

---

### 2. `run.sh` (Esecuzione Docker)

**Prerequisiti:**
- Docker installato
- NVIDIA Container Toolkit installato
- GPU NVIDIA disponibile

**Cosa fa:**
```bash
1. Build immagine Docker (Dockerfile)
   • Base: nvidia/cuda:13.0.0-runtime-ubuntu22.04
   • Installa Miniconda
   • Crea environment RAPIDS 25.10
   • Installa cuML, cuDF, cuPy

2. Rimuove container vecchi (se esistono)

3. Esegue attestation sull'host

4. Lancia container con:
   • --gpus all (accesso GPU)
   • --device /dev/nvidia* (device mapping)
   • Volume mount: $(pwd):/app
   
5. Esegue nel container:
   conda run -n rapids-25.10 python python-native/src/main.py
```

**Uso:**
```bash
./run.sh
```

**Vantaggi:**
- ✅ Ambiente isolato
- ✅ Nessuna configurazione host
- ✅ Ripetibile
- ❌ Più lento (build ogni volta)

---

### 3. `run_python_benchmark.sh` (Benchmark Pulito)

**Prerequisiti:**
- Python 3.x con dipendenze installate
- GPU NVIDIA (opzionale, dipende da main.py)

**Cosa fa:**
```bash
1. Rimuove model_diabetes_gpu.pkl (se esiste)
2. Esegue python3 main.py
3. Mostra risultati formattati
```

**Uso:**
```bash
./run_python_benchmark.sh
```

**Quando usare:**
- Per benchmark comparativi puliti
- Per evitare cache di modelli vecchi

---

## 🦀 Script WASM

### 1. `build_wasm.sh` (Build Solo)

**Prerequisiti:**
- Rust installato (`curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`)
- Target `wasm32-wasi` (`rustup target add wasm32-wasi`)

**Cosa fa:**
```bash
[1/6] Verifica Rust installato
[2/6] Verifica/installa target wasm32-wasi
[3/6] Opzionale --clean: pulisce build
[4/6] Opzionale --test: esegue test
[5/6] Build modulo WASM
[6/6] Mostra summary (path + size)
```

**Output:** `wasm-ml/target/wasm32-wasi/release/wasm_ml.wasm`

**Uso:**
```bash
# Debug build
./build_wasm.sh

# Release optimizzata
./build_wasm.sh --release

# Con test
./build_wasm.sh --test

# Clean + rebuild
./build_wasm.sh --clean --release
```

**NON installa Rust** - se manca, ti dice di installarlo manualmente.

---

### 2. `run_wasm_benchmark.sh` (Build + Benchmark Completo)

**Prerequisiti:**
- Rust installato
- Python 3.x (per export dataset)
- Dependencies Python: pandas, sklearn

**Cosa fa:**
```bash
[1/4] Esporta dataset da Python
      • python3 export_diabetes_for_wasm.py
      • Genera: wasm-ml/data/diabetes_train.csv
      •         wasm-ml/data/diabetes_test.csv

[2/4] Verifica CSV esistono
      • Conta righe train/test

[3/4] Build binario WASM
      • cd wasm-ml
      • cargo build --release --bin wasm-ml-benchmark
      • Genera: target/release/wasm-ml-benchmark

[4/4] Esegue benchmark
      • ./target/release/wasm-ml-benchmark
      • Training + Inferenza + MSE
```

**Uso:**
```bash
./run_wasm_benchmark.sh
```

**Primo run completo** - fa tutto da zero.

---

### 3. `run_wasm_local.sh` (Esecuzione Rapida)

**Prerequisiti:**
- Binario già buildato (`wasm-ml/target/release/wasm-ml-benchmark`)
- Dataset già esportato (CSV)

**Cosa fa:**
```bash
1. Controlla se binario esiste
   • NO → chiama run_wasm_benchmark.sh
   • SÌ → procede

2. Controlla se dataset esiste
   • NO → esporta da Python
   • SÌ → procede

3. Esegue benchmark
```

**Uso:**
```bash
./run_wasm_local.sh
```

**Run veloci successive** - salta build se non necessario.

---

### 4. `run_wasm_docker.sh` (Esecuzione Docker)

**Prerequisiti:**
- Docker installato
- NVIDIA Container Toolkit (opzionale, per GPU)

**Cosa fa:**
```bash
1. Build immagine Docker (Dockerfile.wasm)
   • Base: nvidia/cuda:12.2.0-runtime-ubuntu22.04
   • Installa Rust + Cargo
   • Installa target wasm32-wasi
   • Installa Wasmtime (WASI runtime)
   • Installa Python 3.11
   • Build modulo WASM
   • Installa dipendenze Python

2. Rimuove container vecchi

3. Lancia container con GPU
   • --gpus all
   • --device /dev/nvidia*
   • Volume mount per accesso codice

4. Esegue nel container:
   • export dataset
   • build binario
   • run benchmark
```

**Uso:**
```bash
./run_wasm_docker.sh
```

**Vantaggi:**
- ✅ Nessuna dipendenza locale
- ✅ Environment riproducibile
- ❌ Build più lenta

---

## 🐳 Confronto Dockerfile

### `Dockerfile` (Python RAPIDS)
```dockerfile
Base: nvidia/cuda:13.0.0-runtime-ubuntu22.04
Runtime: Miniconda + environment RAPIDS 25.10
Librerie: cuML, cuDF, cuPy (GPU-accelerated ML)
Linguaggio: Python 3.12
CUDA: 13.0
Entry: conda run -n rapids-25.10 python main.py
```

### `Dockerfile.wasm` (Rust WASM)
```dockerfile
Base: nvidia/cuda:12.2.0-runtime-ubuntu22.04
Runtime: Rust + Wasmtime (WASI)
Librerie: wgpu (WebGPU), custom RandomForest
Linguaggio: Rust → WASM + Python wrapper
CUDA: 12.2 (per WebGPU backend)
Entry: python3 main.py (che chiama WASM)
```

**Differenze chiave:**
| Aspetto | Python RAPIDS | WASM |
|---------|---------------|------|
| **ML Framework** | cuML (NVIDIA) | Custom Rust |
| **GPU API** | CUDA diretto | WebGPU/wgpu |
| **Training** | GPU completo | CPU (GPU in sviluppo) |
| **Inferenza** | GPU completo | CPU (GPU in sviluppo) |
| **Portabilità** | Solo CUDA | Cross-platform (teoricamente) |
| **Dimensione immagine** | ~5-8 GB | ~2-3 GB |

---

## 📋 Workflow Completo

### Setup Iniziale (una volta)

**Per Python:**
```bash
# Opzione A: Docker (consigliato su Azure)
docker pull nvidia/cuda:13.0.0-runtime-ubuntu22.04

# Opzione B: Locale
conda create -n rapids-25.10 -c rapidsai -c conda-forge rapids=25.10
```

**Per WASM:**
```bash
# Installa Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env

# Aggiungi target WASM
rustup target add wasm32-wasi

# Installa dipendenze Python per export dataset
pip install pandas scikit-learn
```

---

### Benchmark Comparativo

**Sequenza consigliata:**

```bash
# 1. Esegui Python benchmark
./run_python_benchmark.sh > results_python.txt

# 2. Esegui WASM benchmark
./run_wasm_benchmark.sh > results_wasm.txt

# 3. Confronta risultati
echo "=== Python ==="
grep "Mean Squared Error" results_python.txt
echo ""
echo "=== WASM ==="
grep "Mean Squared Error" results_wasm.txt
```

---

## 🔍 Troubleshooting

### Python

**Errore: "libcublas.so not found"**
```bash
# Verifica CUDA
nvidia-smi

# Verifica LD_LIBRARY_PATH
echo $LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

**Errore: "Conda environment not found"**
```bash
# Lista environment
conda env list

# Crea se manca
conda create -n rapids-25.10 -c rapidsai rapids=25.10
```

### WASM

**Errore: "cargo: command not found"**
```bash
# Installa Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```

**Errore: "target wasm32-wasi not found"**
```bash
rustup target add wasm32-wasi
```

**Errore: "CSV file not found"**
```bash
# Esporta manualmente
python3 export_diabetes_for_wasm.py
```

---

## ⚡ Performance Tips

### Python
- Usa `--release` per build ottimizzate Docker
- Abilita MPS (Multi-Process Service) su GPU condivise
- Pre-carica dataset in /dev/shm per I/O più veloce

### WASM
- Sempre usa `--release` per benchmark (`cargo build --release`)
- Usa batch prediction quando possibile
- Considera `--target-cpu=native` per ottimizzazioni CPU

---

## 📊 Script di Confronto Automatico

Creo anche uno script che esegue entrambi e compara:

```bash
# compare_benchmarks.sh (prossimo step)
./run_python_benchmark.sh
./run_wasm_benchmark.sh
# Parse e confronto MSE
```

Vuoi che creo questo script comparativo?
