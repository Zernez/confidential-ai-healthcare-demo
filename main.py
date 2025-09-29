from attestation import NvidiaAttestation
from train_model import MLTrainer
from infer_model import MLInferencer

def main():
    # Step 1: Attestazione (centralizzata)
    attestor = NvidiaAttestation()
    if not attestor.perform_attestation():
        print("[MAIN] Attestazione fallita. Blocco esecuzione.")
        return

    print("[MAIN] Attestazione riuscita. Procedo...")

    # Step 2: Training (GPU)
    trainer = MLTrainer()
    trainer.train_and_split()

    # Step 3: Inferenza (GPU)
    inferencer = MLInferencer()
    inferencer.run_inference()

if __name__ == "__main__":
    main()
