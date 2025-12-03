# Benchmark Suite for Confidential AI Healthcare Demo

Sistema di benchmarking automatizzato per confrontare le performance di:
- **Python Native** (cuML/RAPIDS)
- **C++ WASM** (WebAssembly)
- **Rust WASM** (WebAssembly)

## Requisiti

```bash
pip install -r requirements.txt
```

## Utilizzo

### Eseguire benchmark completo (20 run)

```bash
cd benchmark-suite
./run_benchmark.sh
```

### Personalizzare numero di run

```bash
python main.py --runs 10 --warmup 2
```

### Rigenerare grafici da dati esistenti

```bash
python main.py --plot-only
python main.py --plot-only --input results/benchmark_report_XXXXXX.json
```

## Output

### Directory `results/`

```
results/
├── benchmark_report_YYYYMMDD_HHMMSS.json   # Dati completi
├── plots/
│   ├── violin_attestation.pdf              # Distribuzione attestazione
│   ├── violin_training.pdf                 # Distribuzione training
│   ├── violin_inference.pdf                # Distribuzione inference
│   ├── bar_comparison.pdf                  # Confronto metriche
│   ├── bar_total_time.pdf                  # Tempo totale stacked
│   └── combined_figure.pdf                 # Figura combinata per paper
└── latex/
    ├── table_module_sizes.tex              # Dimensioni moduli WASM
    ├── table_timing.tex                    # Timing con statistiche
    └── table_full_results.tex              # Risultati completi
```

## Struttura JSON Output

```json
{
  "timestamp": "2024-12-03T14:00:00",
  "config": {
    "num_runs": 20,
    "warmup_runs": 1
  },
  "system_info": {
    "gpu_name": "NVIDIA H100 NVL",
    "driver_version": "570.195.03"
  },
  "benchmarks": {
    "python": {
      "name": "python",
      "display_name": "Python Native",
      "num_runs": 20,
      "successful_runs": 20,
      "attestation": {
        "mean": 3500.0,
        "std": 150.0,
        "min": 3200.0,
        "max": 3800.0,
        "median": 3450.0,
        "cv": 4.3,
        "values": [...]
      },
      "training": {...},
      "inference": {...},
      "mse": {...},
      "module_size_bytes": 0
    },
    "cpp": {
      "module_size_bytes": 1234567,
      "module_size_kb": 1205.6,
      ...
    },
    "rust": {...}
  }
}
```

## Grafici Generati

### Violin Plot
Mostra la distribuzione completa dei tempi per ogni benchmark, evidenziando:
- Media e mediana
- Variabilità (forma del "violino")
- Outliers

### Bar Chart
Confronto diretto delle medie con barre di errore (deviazione standard).

### Combined Figure
Figura singola ottimizzata per paper (larghezza colonna ~3.5 pollici) con:
1. Violin plot training
2. Violin plot inference  
3. Stacked bar total time

## Tabelle LaTeX

Le tabelle generate sono pronte per l'inclusione in paper:

```latex
\input{results/latex/table_timing.tex}
\input{results/latex/table_module_sizes.tex}
```

## Metriche Statistiche

Per ogni metrica vengono calcolate:
- **Mean**: Media aritmetica
- **Std**: Deviazione standard
- **Min/Max**: Valori estremi
- **Median**: Valore mediano
- **P5/P95**: 5° e 95° percentile
- **CV**: Coefficiente di variazione (%)

Il CV (Coefficient of Variation) è utile per confrontare la variabilità relativa tra benchmark con scale diverse.
