#!/bin/bash
set -e

# ╔════════════════════════════════════════════════╗
# ║  Dataset Setup Script                          ║
# ║  Generates diabetes dataset for WASM modules  ║
# ╚════════════════════════════════════════════════╝

echo "╔════════════════════════════════════════════════╗"
echo "║  Dataset Setup - Diabetes Dataset             ║"
echo "╚════════════════════════════════════════════════╝"
echo ""

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="$PROJECT_ROOT/wasm-ml/data"
PYTHON_SCRIPT="$PROJECT_ROOT/export_diabetes_for_wasm.py"

# ─────────────────────────────────────────────────
# [1/4] Check Python installation
# ─────────────────────────────────────────────────
echo "[1/4] Checking Python installation..."

if ! command -v python3 &> /dev/null; then
    echo "Python3 not found. Please install Python 3.8+"
    exit 1
fi

PYTHON_VERSION=$(python3 --version)
echo " $PYTHON_VERSION"

# ─────────────────────────────────────────────────
# [2/4] Check required packages
# ─────────────────────────────────────────────────
echo ""
echo "[2/4] Checking required Python packages..."

# Check for sklearn
if ! python3 -c "import sklearn" 2>/dev/null; then
    echo "  Installing scikit-learn..."
    pip3 install scikit-learn --quiet
fi
echo "scikit-learn installed"

# Check for pandas
if ! python3 -c "import pandas" 2>/dev/null; then
    echo "  Installing pandas..."
    pip3 install pandas --quiet
fi
echo "pandas installed"

# Check for numpy
if ! python3 -c "import numpy" 2>/dev/null; then
    echo "  Installing numpy..."
    pip3 install numpy --quiet
fi
echo "numpy installed"

# ─────────────────────────────────────────────────
# [3/4] Create data directory
# ─────────────────────────────────────────────────
echo ""
echo "[3/4] Creating data directory..."

mkdir -p "$DATA_DIR"
echo "Directory created: $DATA_DIR"

# Also create symlink in project root for easier access
ROOT_DATA_LINK="$PROJECT_ROOT/data"
if [ ! -e "$ROOT_DATA_LINK" ]; then
    ln -sf "wasm-ml/data" "$ROOT_DATA_LINK"
    echo "✓ Symlink created: data -> wasm-ml/data"
fi

# Also create symlink for wasmwebgpu-ml if it doesn't exist
WASMWEBGPU_DATA="$PROJECT_ROOT/wasmwebgpu-ml/data"
if [ ! -e "$WASMWEBGPU_DATA" ]; then
    mkdir -p "$PROJECT_ROOT/wasmwebgpu-ml"
    ln -sf "$DATA_DIR" "$WASMWEBGPU_DATA" 2>/dev/null || cp -r "$DATA_DIR" "$WASMWEBGPU_DATA"
    echo "Symlink created: wasmwebgpu-ml/data -> wasm-ml/data"
fi

# ─────────────────────────────────────────────────
# [4/4] Generate dataset
# ─────────────────────────────────────────────────
echo ""
echo "[4/4] Generating diabetes dataset..."
echo ""

cd "$PROJECT_ROOT"

if [ -f "$PYTHON_SCRIPT" ]; then
    python3 "$PYTHON_SCRIPT"
else
    # Create the script inline if it doesn't exist
    echo "Creating export script..."
    python3 << 'PYTHON_EOF'
"""
Export diabetes dataset with exact same split as Python baseline
Uses sklearn.datasets.load_diabetes()
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
import os

def export_diabetes_dataset():
    print("[EXPORT] Loading sklearn diabetes dataset...")
    data = load_diabetes()
    X = data.data
    y = data.target
    
    print(f"[EXPORT] Dataset shape: {X.shape}")
    print(f"[EXPORT] Features: {list(data.feature_names)}")
    
    # Same split parameters as Python baseline
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    
    print(f"[EXPORT] Splitting with test_size={TEST_SIZE}, random_state={RANDOM_STATE}")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    print(f"[EXPORT] Train samples: {len(X_train)}")
    print(f"[EXPORT] Test samples: {len(X_test)}")
    
    # Create DataFrames with feature names
    feature_names = list(data.feature_names)
    
    train_df = pd.DataFrame(X_train, columns=feature_names)
    train_df['target'] = y_train
    
    test_df = pd.DataFrame(X_test, columns=feature_names)
    test_df['target'] = y_test
    
    # Ensure directory exists
    os.makedirs("wasm-ml/data", exist_ok=True)
    
    # Export to CSV
    train_path = "wasm-ml/data/diabetes_train.csv"
    test_path = "wasm-ml/data/diabetes_test.csv"
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"[EXPORT] ✓ Saved training data to: {train_path}")
    print(f"[EXPORT] ✓ Saved test data to: {test_path}")
    
    # Show preview
    print("\n[VERIFY] Train data preview:")
    print(train_df.head(3).to_string())
    print(f"\n[VERIFY] Target stats - Train: mean={y_train.mean():.2f}, std={y_train.std():.2f}")
    print(f"[VERIFY] Target stats - Test: mean={y_test.mean():.2f}, std={y_test.std():.2f}")

if __name__ == "__main__":
    export_diabetes_dataset()
PYTHON_EOF
fi

# ─────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────
echo ""
echo "╔════════════════════════════════════════════════╗"
echo "║  ✓ Dataset Setup Complete!                    ║"
echo "╚════════════════════════════════════════════════╝"
echo ""

# Verify files exist
if [ -f "$DATA_DIR/diabetes_train.csv" ] && [ -f "$DATA_DIR/diabetes_test.csv" ]; then
    TRAIN_ROWS=$(wc -l < "$DATA_DIR/diabetes_train.csv")
    TEST_ROWS=$(wc -l < "$DATA_DIR/diabetes_test.csv")
    TRAIN_SIZE=$(ls -lh "$DATA_DIR/diabetes_train.csv" | awk '{print $5}')
    TEST_SIZE=$(ls -lh "$DATA_DIR/diabetes_test.csv" | awk '{print $5}')
    
    echo "Generated files:"
    echo "  • $DATA_DIR/diabetes_train.csv"
    echo "    Rows: $((TRAIN_ROWS - 1)) samples + header"
    echo "    Size: $TRAIN_SIZE"
    echo ""
    echo "  • $DATA_DIR/diabetes_test.csv"
    echo "    Rows: $((TEST_ROWS - 1)) samples + header"
    echo "    Size: $TEST_SIZE"
    echo ""
    echo "Dataset info:"
    echo "  • Source: sklearn.datasets.load_diabetes()"
    echo "  • Features: 10 (age, sex, bmi, bp, s1-s6)"
    echo "  • Target: Disease progression measure"
    echo "  • Split: 80% train, 20% test (random_state=42)"
else
    echo "Error: CSV files not created"
    exit 1
fi

echo ""
echo "Next steps:"
echo "  1. Build WASM modules: ./build_all.sh --skip-cpp"
echo "  2. Run benchmark: ./run_with_attestation.sh --rust"
