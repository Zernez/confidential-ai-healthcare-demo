import requests
import json
from nv_attestation_sdk import attestation
from attestation_local import NvidiaAttestation as LocalAttestation
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
        Genera quote di attestazione usando NVIDIA Attestation SDK (API aggiornata).
        Returns:
            evidence: lista di dict con certificate/evidence/arch
        """
        try:
            client = attestation.Attestation()
            # Non usiamo add_verifier, lasciamo che l'SDK gestisca i default.
            # Passiamo il nonce direttamente a get_evidence.
            result = client.get_evidence(nonce=nonce)
            print(f"[ATTESTATION] Tipo evidence: {type(result)}")
            print(f"[ATTESTATION] Contenuto evidence: {result}")

            if not result or not isinstance(result, tuple) or len(result) < 2 or not result[0]:
                # Se il risultato è (False, []) o simile, l'attestazione è fallita
                raise RuntimeError("Evidence non valida o attestazione fallita dall'SDK NVIDIA.")

            # La nuova API potrebbe restituire (nonce, evidence_list)
            returned_nonce, evidence_list = result
            if not evidence_list or not isinstance(evidence_list, list) or len(evidence_list) == 0:
                raise RuntimeError("Lista evidence vuota restituita dall'SDK NVIDIA.")

            evidence_dict = evidence_list[0]
            evidence = evidence_dict.get("evidence")
            certificate = evidence_dict.get("certificate")
            arch = evidence_dict.get("arch")

            if not evidence:
                raise RuntimeError("Evidence vuota nel dizionario restituito dall'SDK.")

            return evidence, certificate, arch, returned_nonce
        except Exception as e:
            raise RuntimeError(f"Errore generando quote con NVIDIA SDK: {str(e)}")

    def send_to_nas(self, evidence, certificate, arch, nonce):
        """
        Invia evidence al NVIDIA Remote Attestation Service (NRAS).
        Args:
            evidence: evidence string
            certificate: certificate string
            arch: GPU architecture
        Returns:
            Risultato dell'attestazione dal servizio NVIDIA
        """
        try:
            payload = {
                "nonce": nonce,
                "evidence": evidence,
                "certificate": certificate,
                "arch": arch,
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
        Esegue il processo completo di attestazione (API aggiornata).
        Returns:
            True se attestazione riuscita, False altrimenti
        """
        print("[ATTESTATION] Avvio attestazione GPU NVIDIA con SDK...")
        try:
            evidence, certificate, arch, returned_nonce = self.get_quote(nonce)
            print("[ATTESTATION] Evidence ottenuto, invio al NRAS...")
            attestation_result = self.send_to_nas(evidence, certificate, arch, returned_nonce)
            print("[ATTESTATION] Risultato NRAS:")
            print(json.dumps(attestation_result, indent=2))

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
            print("[ATTESTATION] Fallback: eseguo attestazione locale semplificata...")
            try:
                simple_attestor = LocalAttestation()
                success = simple_attestor.perform_attestation()
                if success:
                    print("[ATTESTATION] Attestazione locale semplificata OK.")
                else:
                    print("[ATTESTATION] Attestazione locale semplificata FALLITA.")
                return success
            except Exception as e2:
                print(f"[ATTESTATION] Errore nel fallback locale: {e2}")
                return False

if __name__ == "__main__":
    attestor = NvidiaAttestation()
    success = attestor.perform_attestation()
    sys.exit(0 if success else 1)
