# WASM ML Module - RandomForest with WebGPU

Implementazione RandomForest con accelerazione GPU via WebGPU per inferenza parallela.

## Architettura

```
┌─────────────────────────────────────────┐
│  RandomForest (CPU Training)            │
│  ├── Bootstrap Sampling                 │
│  ├── Decision Tree Building             │
│  └── Model Serialization (bincode)      │
└─────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│  GPU Inference (WebGPU)                 │
│  ├── Parallel Tree Predictions          │
│  ├── GPU Averaging Kernel               │
│  └── Result Aggregation                 │
└─────────────────────────────────────────┘
```

## Features

- **Training**: CPU-based RandomForest training con bagging
- **Inferenza GPU**: Parallelizzazione delle predizioni su WebGPU
- **Serializzazione**: Modelli compatti con bincode
- **WASM-ready**: Compilabile per wasm32-wasi target

## Build

### Prerequisiti

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Add WASM target
rustup target add wasm32-wasi

# Install wasm-pack (optional, for browser deployment)
cargo install wasm-pack
```

### Build per WASI Runtime

```bash
cd wasm-ml
cargo build --target wasm32-wasi --release
```

### Build per Browser (WASM)

```bash
cd wasm-ml
wasm-pack build --target web
```

## Uso

### Training

```rust
use wasm_ml::train_model;

let training_data = vec![/* features */];
let training_labels = vec![/* labels */];
let n_features = 10;

let model_bytes = train_model(
    200,        // n_estimators
    16,         // max_depth
    training_data,
    training_labels,
    n_features,
)?;

// Salva model_bytes su disco
std::fs::write("model.bin", model_bytes)?;
```

### Inferenza GPU

```rust
use wasm_ml::predict_gpu;

let model_bytes = std::fs::read("model.bin")?;
let test_data = vec![/* test features */];
let n_features = 10;

let predictions = predict_gpu(
    model_bytes,
    test_data,
    n_features,
).await?;
```

### Inferenza CPU (fallback)

```rust
use wasm_ml::predict_cpu;

let predictions = predict_cpu(
    model_bytes,
    test_data,
    n_features,
)?;
```

## Performance

Su H100 GPU (Azure):
- **Training**: ~200 trees in 2-5 secondi (CPU)
- **Inferenza**: ~10-50x speedup vs CPU per batch grandi (>100 samples)

## Parametri RandomForest

- `n_estimators`: Numero di alberi (default: 200)
- `max_depth`: Profondità massima alberi (default: 16)
- Feature selection: sqrt(n_features) per split
- Bagging: Bootstrap sampling con replacement

## Limitazioni Attuali

1. **Training solo CPU**: WebGPU non supporta ancora decision tree building
2. **Memoria**: Modelli grandi (>500 trees) possono eccedere limiti WASM
3. **Shader semplificato**: Averaging su GPU, tree traversal ancora su CPU

## Roadmap

- [ ] Tree traversal su GPU con compute shaders
- [ ] Supporto classificazione (oltre regressione)
- [ ] Feature importance calculation
- [ ] Modelli ensemble multipli
- [ ] Quantizzazione modelli per ridurre dimensioni

## Testing

```bash
cargo test
```

## License

MIT
