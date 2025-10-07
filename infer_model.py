import joblib

class MLInferencer:
    def __init__(self, model_path="model_diabetes_gpu.pkl"):
        self.model_path = model_path

    def run_inference(self):
        # Import RAPIDS qui (dopo che env vars sono impostate)
        import cupy as cp
        
        # Carico modello + test set (GPU types)
        model, X_test, y_test = joblib.load(self.model_path)

        print("[INFERENZA] Predizione su test set (GPU)...")
        preds = model.predict(X_test)  # cuDF -> CuPy/NumPy-like device outputs
        # Assicuro CuPy arrays per metrica
        preds_cp = cp.asarray(preds)
        y_test_cp = y_test.values  # cuDF Series -> CuPy

        # MSE su GPU
        mse = cp.mean((preds_cp - y_test_cp) ** 2)
        print(f"[INFERENZA] Campioni: {len(y_test_cp)}")
        print(f"[INFERENZA] Mean Squared Error (GPU): {float(mse):.4f}")
