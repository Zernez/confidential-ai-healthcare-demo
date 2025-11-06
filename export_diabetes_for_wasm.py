"""
Export diabetes dataset with exact same split as main.py
This ensures WASM benchmark uses identical train/test data
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

def export_diabetes_dataset():
    """Export diabetes dataset with same parameters as train_model.py"""
    
    print("[EXPORT] Loading diabetes dataset...")
    data = load_diabetes()
    X = data.data
    y = data.target
    
    print(f"[EXPORT] Dataset shape: {X.shape}")
    print(f"[EXPORT] Features: {data.feature_names}")
    
    # Same split parameters as train_model.py
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    
    print(f"[EXPORT] Splitting with test_size={TEST_SIZE}, random_state={RANDOM_STATE}")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    print(f"[EXPORT] Train samples: {len(X_train)}")
    print(f"[EXPORT] Test samples: {len(X_test)}")
    
    # Create DataFrames with feature names
    feature_names = data.feature_names
    
    train_df = pd.DataFrame(X_train, columns=feature_names)
    train_df['target'] = y_train
    
    test_df = pd.DataFrame(X_test, columns=feature_names)
    test_df['target'] = y_test
    
    # Export to CSV
    train_path = "wasm-ml/data/diabetes_train.csv"
    test_path = "wasm-ml/data/diabetes_test.csv"
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"[EXPORT] Saved training data to: {train_path}")
    print(f"[EXPORT] Saved test data to: {test_path}")
    
    # Verify export
    print("\n[VERIFY] Train data preview:")
    print(train_df.head())
    print(f"\n[VERIFY] Target stats - Train mean: {y_train.mean():.2f}, std: {y_train.std():.2f}")
    print(f"[VERIFY] Target stats - Test mean: {y_test.mean():.2f}, std: {y_test.std():.2f}")
    
    print("\n[EXPORT] ✅ Export completed successfully!")
    print(f"[EXPORT] Files ready for WASM benchmark")

if __name__ == "__main__":
    export_diabetes_dataset()
