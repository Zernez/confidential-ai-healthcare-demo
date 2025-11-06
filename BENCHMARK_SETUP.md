# WASM vs Python ML Benchmark

Questo setup permette di confrontare le performance di un'applicazione ML tra:
- **Python nativo** con RAPIDS (GPU accelerato via CUDA)
- **WebAssembly** con wgpu (GPU accelerato via WebGPU)

## 📋 Configurazione Identica

Entrambe le implementazioni usano:
- **Dataset**: sklearn diabetes (442 samples, 10 features)
- **Split**: 80/20 train/test con `random_state=42`
- **Modello**: RandomForest Regressor
  - `n_estimators`: 200
  - `max_depth`: 16
  - Task: Regressione (MSE)

## 🚀 Esecuzione

### Opzione 1: Benchmark Singoli

**Python (RAPIDS):**
```powershell
.\run_python_benchmark.ps1
```

**WASM:**
```powershell
.\run_wasm_benchmark.ps1
```

### Opzione 2: Confronto Completo

Esegui entrambi in sequenza per confrontare i risultati:

```powershell
# Python
.\run_python_benchmark.ps1

# WASM
.\run_wasm_benchmark.ps1
```

## 📊 Output Atteso

### Python (RAPIDS)
```
[TRAINING] Avvio training su GPU (cuML RandomForest)...
[TRAINING] Completato.
[TRAINING] Modello e test set salvati in model_diabetes_gpu.pkl
[INFERENZA] Predizione su test set (GPU)...
[INFERENZA] Campioni: 89
[INFERENZA] Mean Squared Error (GPU): XXXX.XXXX
```

### WASM
```
=== TRAINING PHASE ===
[TRAINING] Creating RandomForest with 200 estimators, max_depth 16
[TRAINING] Starting training on CPU...
[TRAINING] Training completed!
[TRAINING] Model saved to: data/model_diabetes_wasm.bin

=== INFERENCE PHASE ===
[INFERENCE] Running predictions on 89 test samples...
[INFERENCE] Samples: 89
[INFERENCE] Mean Squared Error (CPU): XXXX.XXXX
```

## 📁 File Generati

### Python
- `model_diabetes_gpu.pkl` - Modello cuML serializzato

### WASM
- `wasm-ml/data/diabetes_train.csv` - Dataset training
- `wasm-ml/data/diabetes_test.csv` - Dataset test
- `wasm-ml/data/model_diabetes_wasm.bin` - Modello Rust serializzato
- `wasm-ml/target/release/wasm-ml-benchmark.exe` - Binario compilato

## 🔄 Workflow Completo

### Python
```
main.py
  ↓
train_model.py → MLTrainer.train_and_split()
  ↓
  • Carica diabetes
  • Split 80/20
  • Training cuML RandomForest (GPU)
  • Salva model_diabetes_gpu.pkl
  ↓
infer_model.py → MLInferencer.run_inference()
  ↓
  • Carica model_diabetes_gpu.pkl
  • Inferenza su test set (GPU)
  • Calcola e stampa MSE
```

### WASM
```
run_wasm_benchmark.ps1
  ↓
export_diabetes_for_wasm.py
  ↓
  • Carica diabetes
  • Split 80/20 (stesso random_state=42)
  • Esporta CSV
  ↓
cargo build --release
  ↓
wasm-ml-benchmark.exe
  ↓
train_and_save()
  ↓
  • Carica diabetes_train.csv
  • Training RandomForest (CPU)
  • Salva model_diabetes_wasm.bin
  ↓
load_and_infer()
  ↓
  • Carica diabetes_test.csv
  • Carica model_diabetes_wasm.bin
  • Inferenza su test set (CPU)
  • Calcola e stampa MSE
```

## ⚙️ Requisiti

### Python
- Python 3.x
- cuML (RAPIDS)
- cuDF
- cuPy
- scikit-learn
- joblib
- NVIDIA GPU con CUDA

### WASM
- Rust toolchain (rustc, cargo)
- Target: native (non wasm32 per ora, binario Windows)
- No GPU richiesta per questa versione (CPU-only training)

## 🔍 Verifica Dati Identici

Per verificare che i dataset siano identici:

```powershell
# Conta righe
(Get-Content wasm-ml\data\diabetes_train.csv).Count  # Deve essere 354 (353 + header)
(Get-Content wasm-ml\data\diabetes_test.csv).Count   # Deve essere 90 (89 + header)
```

## ⚠️ Note Importanti

1. **Random State**: Entrambi usano `random_state=42` per garantire split identico
2. **Test Set**: Il test set è ESATTAMENTE lo stesso (esportato da Python)
3. **GPU vs CPU**: 
   - Python: Training e Inferenza su GPU (CUDA/cuML)
   - WASM: Training su CPU, Inferenza CPU (futura GPU via WebGPU)
4. **MSE**: Potrebbero esserci piccole differenze dovute a:
   - Implementazione algoritmo (cuML vs Rust custom)
   - Precisione numerica (float32 vs float64)
   - Ordine operazioni (GPU parallelismo)

## 📈 Cosa Confrontare

- ✅ **MSE**: Dovrebbe essere simile (±5-10%)
- ✅ **Tempo Training**: Python GPU vs Rust CPU
- ✅ **Tempo Inferenza**: Python GPU vs Rust CPU
- ⏳ **Memoria**: (non misurato in questa versione)

## 🚧 Limitazioni Attuali

### WASM Implementation
- ❌ GPU training non implementato
- ❌ GPU inference non implementato
- ✅ Stessi parametri modello
- ✅ Stesso dataset e split
- ✅ Stessa sequenza operazioni

### Prossimi Step
1. Implementare GPU inference via WebGPU
2. Implementare GPU training (se fattibile)
3. Aggiungere timing preciso
4. Deploy su Azure H100

## 📝 Debugging

Se l'MSE è molto diverso:
1. Verifica che i CSV siano stati generati correttamente
2. Controlla il numero di sample in train/test
3. Verifica che random_state sia 42 in entrambi
4. Controlla i parametri RandomForest (200 trees, depth 16)
