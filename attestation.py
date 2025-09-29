import subprocess
import requests
import json

class NvidiaAttestation:
    """
    Gestione attestazione NVIDIA Confidential Computing su GPU H100.
    Richiede toolkit NVIDIA (es. nvidia-cc-tool) e accesso al NAS.
    """

    def __init__(self, nas_url="https://attestation.nvidia.com/v1/attest"):
        self.nas_url = nas_url

    def get_quote(self):
        try:
            result = subprocess.run(
                ["nvidia-cc-tool", "--get-quote"],
                capture_output=True, text=True, check=True
            )
            quote = result.stdout.strip()
            if not quote:
                raise RuntimeError("Quote vuoto restituito dal tool NVIDIA.")
            return quote
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Errore eseguendo nvidia-cc-tool: {e.stderr}")
        except FileNotFoundError:
            raise RuntimeError("nvidia-cc-tool non trovato. Assicurati che sia installato.")

    def send_to_nas(self, quote):
        try:
            response = requests.post(self.nas_url, json={"quote": quote})
            if response.status_code == 200:
                return response.json()
            else:
                raise RuntimeError(f"NAS errore: {response.status_code} {response.text}")
        except requests.RequestException as e:
            raise RuntimeError(f"Errore di rete verso NAS: {e}")

    def perform_attestation(self):
        print("[ATTESTATION] Avvio attestazione GPU NVIDIA...")
        quote = self.get_quote()
        print("[ATTESTATION] Quote ottenuto, invio al NAS...")
        attestation_result = self.send_to_nas(quote)
        print("[ATTESTATION] Risultato NAS:")
        print(json.dumps(attestation_result, indent=2))

        if attestation_result.get("secureMode", False) is True:
            print("[ATTESTATION] GPU in modalità confidential. OK.")
            return True
        else:
            print("[ATTESTATION] GPU NON in modalità confidential. FALLITA.")
            return False
