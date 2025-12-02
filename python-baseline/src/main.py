#!/usr/bin/env python3
"""
Python Baseline Benchmark - Diabetes Prediction
Native Python + cuML/RAPIDS + TEE Attestation

Output format matches WASM modules (Rust and C++) for fair comparison.
"""

import os
import sys
import json
import time
from dataclasses import dataclass, asdict
from typing import Optional

# ═══════════════════════════════════════════════════════════════════════════
# Benchmark Results Structure (matches WASM output)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class BenchmarkResults:
    language: str = "python"
    gpu_device: str = ""
    gpu_backend: str = ""
    tee_type: str = ""
    gpu_available: bool = False
    tee_available: bool = False
    attestation_ms: float = 0.0
    training_ms: float = 0.0
    inference_ms: float = 0.0
    mse: float = 0.0
    train_samples: int = 0
    test_samples: int = 0
    
    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(',', ':'))


class Timer:
    def __init__(self):
        self.start_time = time.perf_counter()
    
    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self.start_time) * 1000
    
    def reset(self):
        self.start_time = time.perf_counter()


# ═══════════════════════════════════════════════════════════════════════════
# Main Benchmark
# ═══════════════════════════════════════════════════════════════════════════

def main():
    results = BenchmarkResults()
    
    # Header (matches WASM output)
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   Python Baseline Benchmark - Diabetes Prediction        ║")
    print("║   Native Python + cuML/RAPIDS + TEE Attestation          ║")
    print("╚══════════════════════════════════════════════════════════╝")
    
    # ═══════════════════════════════════════════════════════════════════
    # GPU Information
    # ═══════════════════════════════════════════════════════════════════
    
    print("\n=== GPU INFORMATION ===")
    
    try:
        import tee_attestation
        gpu_name, gpu_backend, gpu_memory = tee_attestation.get_gpu_info()
        results.gpu_available = True
        results.gpu_device = gpu_name
        results.gpu_backend = gpu_backend
        
        print(f"[GPU] Device: {gpu_name}")
        print(f"[GPU] Backend: {gpu_backend}")
        print(f"[GPU] Memory: {gpu_memory} MB")
        print(f"[GPU] Hardware: YES ✓")
    except Exception as e:
        print(f"[GPU] Not available: {e}")
        results.gpu_available = False
    
    # ═══════════════════════════════════════════════════════════════════
    # TEE Attestation (using Rust bindings - same as WASM host)
    # ═══════════════════════════════════════════════════════════════════
    
    print("\n=== TEE ATTESTATION ===")
    
    attestation_timer = Timer()
    
    try:
        import tee_attestation
        
        # Detect TEE type
        tee_info = tee_attestation.detect_tee()
        results.tee_type = tee_info.tee_type
        results.tee_available = tee_info.supports_attestation
        
        print(f"[TEE] Type: {tee_info.tee_type}")
        print(f"[TEE] Supports attestation: {'YES' if tee_info.supports_attestation else 'NO'}")
        
        # VM attestation
        vm_result = tee_attestation.attest_vm()
        if vm_result.success:
            print(f"[TEE] VM attestation: OK (token: {vm_result.token_length} chars)")
        else:
            print(f"[TEE] VM attestation: FAILED ({vm_result.error})")
        
        # GPU attestation
        gpu_result = tee_attestation.attest_gpu(0)
        if gpu_result.success:
            print(f"[TEE] GPU attestation: OK (token: {gpu_result.token_length} chars)")
        else:
            print(f"[TEE] GPU attestation: FAILED ({gpu_result.error})")
            
    except ImportError:
        print("[TEE] tee_attestation module not found - skipping attestation")
        print("[TEE] Build with: cd tee_attestation && maturin develop --release")
        results.tee_type = "N/A (module not built)"
        results.tee_available = False
    except Exception as e:
        print(f"[TEE] Attestation error: {e}")
        results.tee_type = f"Error: {e}"
        results.tee_available = False
    
    results.attestation_ms = attestation_timer.elapsed_ms()
    print(f"[TIMING] Attestation: {results.attestation_ms:.2f} ms")
    
    # ═══════════════════════════════════════════════════════════════════
    # Training Phase
    # ═══════════════════════════════════════════════════════════════════
    
    print("\n=== TRAINING ===")
    
    # Model parameters (MUST match WASM configuration)
    N_ESTIMATORS = 200
    MAX_DEPTH = 16
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    
    # Load dataset
    from sklearn.datasets import load_diabetes
    from sklearn.model_selection import train_test_split
    
    data = load_diabetes()
    X = data.data.astype('float32')
    y = data.target.astype('float32')
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    results.train_samples = len(X_train)
    results.test_samples = len(X_test)
    
    print(f"[TRAIN] Dataset: {results.train_samples} samples, {X_train.shape[1]} features")
    print(f"[TRAIN] Model: RandomForest ({N_ESTIMATORS} trees, depth {MAX_DEPTH})")
    
    train_timer = Timer()
    
    try:
        # Try GPU training with cuML/RAPIDS
        import cudf
        import cupy as cp
        from cuml.ensemble import RandomForestRegressor
        
        print("[TRAIN] Accelerator: GPU (cuML/RAPIDS)")
        
        # Convert to GPU
        X_train_gpu = cudf.DataFrame(X_train)
        y_train_gpu = cudf.Series(y_train)
        
        # Train model
        model = RandomForestRegressor(
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            random_state=RANDOM_STATE,
            n_bins=128,
            split_criterion="mse",
            n_streams=1
        )
        
        model.fit(X_train_gpu, y_train_gpu)
        
        # Store for inference
        _model = model
        _use_gpu = True
        
    except Exception as e:
        # Fallback to CPU
        print(f"[TRAIN] GPU not available ({e}), using CPU")
        print("[TRAIN] Accelerator: CPU (scikit-learn)")
        
        from sklearn.ensemble import RandomForestRegressor as SklearnRF
        
        model = SklearnRF(
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        
        _model = model
        _use_gpu = False
    
    results.training_ms = train_timer.elapsed_ms()
    print(f"[TIMING] Training: {results.training_ms:.2f} ms")
    
    # ═══════════════════════════════════════════════════════════════════
    # Inference Phase
    # ═══════════════════════════════════════════════════════════════════
    
    print("\n=== INFERENCE ===")
    print(f"[INFER] Test set: {results.test_samples} samples")
    
    infer_timer = Timer()
    
    if _use_gpu:
        import cudf
        import cupy as cp
        
        print("[INFER] Accelerator: GPU (cuML/RAPIDS)")
        
        X_test_gpu = cudf.DataFrame(X_test)
        y_test_gpu = cudf.Series(y_test)
        
        preds = _model.predict(X_test_gpu)
        
        # Calculate MSE on GPU
        preds_cp = cp.asarray(preds)
        y_test_cp = y_test_gpu.values
        mse = float(cp.mean((preds_cp - y_test_cp) ** 2))
    else:
        import numpy as np
        
        print("[INFER] Accelerator: CPU (scikit-learn)")
        
        preds = _model.predict(X_test)
        mse = float(np.mean((preds - y_test) ** 2))
    
    results.inference_ms = infer_timer.elapsed_ms()
    results.mse = mse
    
    print(f"[TIMING] Inference: {results.inference_ms:.2f} ms")
    print(f"[INFER] MSE: {results.mse:.4f}")
    
    # ═══════════════════════════════════════════════════════════════════
    # Benchmark Summary
    # ═══════════════════════════════════════════════════════════════════
    
    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║                    BENCHMARK RESULTS                      ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print(f"║  Language:       Python (cuML/RAPIDS)                     ║")
    print(f"║  Attestation:  {results.attestation_ms:10.2f} ms                           ║")
    print(f"║  Training:     {results.training_ms:10.2f} ms                           ║")
    print(f"║  Inference:    {results.inference_ms:10.2f} ms                           ║")
    print(f"║  MSE:          {results.mse:10.4f}                             ║")
    print("╚══════════════════════════════════════════════════════════╝")
    
    # JSON output for benchmark aggregation
    print("\n### BENCHMARK_JSON ###")
    print(results.to_json())
    print("### END_BENCHMARK_JSON ###")
    
    print("\n✅ Benchmark completed successfully!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
