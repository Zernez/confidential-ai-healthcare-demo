import joblib
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

import cudf
import cupy as cp
from cuml.ensemble import RandomForestRegressor

class MLTrainer:
    def __init__(self, model_path="model_diabetes_gpu.pkl"):
        self.model_path = model_path

    def train_and_split(self, test_size=0.2, random_state=42, n_estimators=200, max_depth=16):
        # Carico dataset (NumPy)
        data = load_diabetes()
        X = data.data
        y = data.target

        # Split (CPU/NumPy)
        X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Conversione a GPU (cuDF/CuPy)
        X_train = cudf.DataFrame(X_train_np)
        X_test = cudf.DataFrame(X_test_np)
        y_train = cudf.Series(y_train_np)
        y_test = cudf.Series(y_test_np)

        # Modello cuML (GPU)
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_bins=128,
            split_criterion="mse"
        )

        print("[TRAINING] Avvio training su GPU (cuML RandomForest)...")
        model.fit(X_train, y_train)
        print("[TRAINING] Completato.")

        # Salvo modello e test set (pickle/joblib)
        joblib.dump((model, X_test, y_test), self.model_path)
        print(f"[TRAINING] Modello e test set salvati in {self.model_path}")
