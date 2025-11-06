import os
# Fix compatibilità Numba/CUDA per RAPIDS - DEVE essere impostato PRIMA degli import
# os.environ['NUMBA_CUDA_ENABLE_MINOR_VERSION_COMPATIBILITY'] = '1'
# # Disabilita patch cubinlinker che causa problemi con Numba 0.60.0
# os.environ['CUBINLINKER_DISABLE_PATCH'] = '1'

from train_model import MLTrainer
from infer_model import MLInferencer

def main():
    # Step 1: Training (GPU)
    trainer = MLTrainer()
    trainer.train_and_split()

    # Step 2: Inferenza (GPU)
    inferencer = MLInferencer()
    inferencer.run_inference()

if __name__ == "__main__":
    main()
