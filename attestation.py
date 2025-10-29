import requests
import json
from nv_attestation_sdk import attestation
import sys

class NvidiaAttestation:
    """
    Gestione attestazione NVIDIA Confidential Computing su GPU H100.
    Usa NVIDIA Attestation SDK per generare quote di attestazione.
    """

    def __init__(self, nas_url="https://nras.attestation.nvidia.com/v3/attest/gpu"):
        self.nas_url = nas_url
        if attestation is None:
            raise ImportError(
                "NVIDIA Attestation SDK non trovato. "
                "Installa con: pip install nv-attestation-sdk"
            )

    def get_quote(self, nonce=None):
        """
        Genera quote di attestazione usando NVIDIA Attestation SDK.
        
        Args:
            nonce: Nonce opzionale per la richiesta (bytes o None)
        
        Returns:
            Evidence contenente quote, certificate chain e altri dati
        """
        try:
            # Ottieni l'attestazione dalla GPU
            if nonce is None:
                nonce = b""  # Nonce vuoto di default
            
            evidence = attestation.Attestation().get_evidence(nonce)
            print(f"[ATTESTATION] Tipo evidence: {type(evidence)}")
            print(f"[ATTESTATION] Contenuto evidence: {evidence}")
            if not evidence:
                raise RuntimeError("Evidence vuoto restituito dall'SDK NVIDIA.")
            # Se evidence è bytes, non usare .get su di esso
            return evidence
            
        except Exception as e:
            raise RuntimeError(f"Errore generando quote con NVIDIA SDK: {str(e)}")

    def send_to_nas(self, evidence):
        """
        Invia evidence al NVIDIA Remote Attestation Service (NRAS).
        
        Args:
            evidence: Evidence object dall'SDK
            
        Returns:
            Risultato dell'attestazione dal servizio NVIDIA
        """
        try:
            # Converti evidence in formato inviabile
            payload = {
                "evidence": evidence.hex() if isinstance(evidence, bytes) else str(evidence),
                "arch": "HOPPER",  # H100 GPU architecture
                "gpu_attestation": True
            }
            
            response = requests.post(
                self.nas_url,
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                raise RuntimeError(
                    f"NRAS errore: {response.status_code} {response.text}"
                )
        except requests.RequestException as e:
            raise RuntimeError(f"Errore di rete verso NRAS: {e}")

    def perform_attestation(self, nonce=None):
        """
        Esegue il processo completo di attestazione.
        
        Args:
            nonce: Nonce opzionale per la richiesta
            
        Returns:
            True se attestazione riuscita, False altrimenti
        """
        print("[ATTESTATION] Avvio attestazione GPU NVIDIA con SDK...")
        
        try:
            evidence = self.get_quote(nonce)
            print("[ATTESTATION] Evidence ottenuto, invio al NRAS...")
            
            attestation_result = self.send_to_nas(evidence)
            print("[ATTESTATION] Risultato NRAS:")
            print(json.dumps(attestation_result, indent=2))

            # Verifica il risultato
            # Se la risposta è bytes, decodifica e carica come JSON
            if isinstance(attestation_result, bytes):
                try:
                    attestation_result = json.loads(attestation_result.decode())
                except Exception as e:
                    print(f"[ATTESTATION] Errore parsing risposta NRAS: {e}")
                    return False

            if isinstance(attestation_result, dict):
                status = attestation_result.get("status")
                verdict = attestation_result.get("nras_validation", {}).get("verdict")
                if status == "success" or verdict == "pass":
                    print("[ATTESTATION] GPU in modalità confidential. OK.")
                    return True
                else:
                    print("[ATTESTATION] GPU NON in modalità confidential. FALLITA.")
                    return False
            else:
                print(f"[ATTESTATION] Risposta NRAS inattesa: {type(attestation_result)}")
                return False
                
        except Exception as e:
            print(f"[ATTESTATION] Errore durante attestazione: {str(e)}")
            return False

if __name__ == "__main__":
    attestor = NvidiaAttestation()
    success = attestor.perform_attestation()
    sys.exit(0 if success else 1)
