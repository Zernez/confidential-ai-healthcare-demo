import json
import sys
import secrets
from nv_attestation_sdk import attestation
from attestation_local import NvidiaAttestation as LocalAttestation

class NvidiaAttestation:
    """
    Gestione attestazione NVIDIA. Esegue sia l'attestazione remota (vs NRAS)
    che locale, in modo non bloccante.
    """

    def __init__(self, nas_url="https://nras.attestation.nvidia.com/v3/attest/gpu"):
        self.nas_url = nas_url
        if attestation is None:
            raise ImportError("NVIDIA Attestation SDK non trovato.")

    def _perform_remote_attestation(self) -> bool:
        """
        Esegue l'attestazione remota seguendo l'esempio ufficiale NVIDIA.
        L'SDK gestisce la comunicazione con NRAS.
        """
        print("\n[ATTESTATION] Avvio attestazione REMOTA GPU NVIDIA...")
        try:
            client = attestation.Attestation()
            client.set_nonce(secrets.token_hex(32))
            client.add_verifier(attestation.Devices.GPU, attestation.Environments.REMOTE, self.nas_url, "")
            
            print("[ATTESTATION] Raccolta evidence per attestazione remota...")
            evidence_list = client.get_evidence()
            
            print("[ATTESTATION] Invio evidence al servizio NRAS e attesa verdetto...")
            attestation_result = client.attest(evidence_list)
            
            if attestation_result:
                print("[ATTESTATION] VERDETTO REMOTO: Successo. La GPU è attestata da NVIDIA.")
                # Opzionale: ottenere il token se necessario
                # token = client.get_token()
                # print(f"[ATTESTATION] Token remoto: {token[:30]}...")
            else:
                print("[ATTESTATION] VERDETTO REMOTO: Fallimento.")

            return attestation_result
        
        except Exception as e:
            print(f"[ATTESTATION] ERRORE REMOTO: {str(e)}")
            return False

    def _perform_local_attestation(self) -> bool:
        """
        Esegue l'attestazione locale.
        """
        print("\n[ATTESTATION] Avvio attestazione LOCALE GPU NVIDIA...")
        try:
            local_attestor = LocalAttestation()
            success = local_attestor.perform_attestation()
            if success:
                print("[ATTESTATION] VERDETTO LOCALE: Successo.")
            else:
                print("[ATTESTATION] VERDETTO LOCALE: Fallimento.")
            return success
        
        except Exception as e:
            print(f"[ATTESTATION] ERRORE LOCALE: {e}")
            return False

    def perform_attestation(self):
        """
        Orchestra l'esecuzione di entrambe le attestazioni in modo non bloccante.
        """
        remote_success = self._perform_remote_attestation()
        local_success = self._perform_local_attestation()

        print("\n[ATTESTATION] Riepilogo:")
        print(f"- Attestazione Remota (NRAS): {'OK' if remote_success else 'FALLITA'}")
        print(f"- Attestazione Locale:        {'OK' if local_success else 'FALLITA'}")
        
        # La funzione non è bloccante e non restituisce un valore di successo aggregato,
        # ma stampa solo i risultati come richiesto.
        return True


if __name__ == "__main__":
    print("=== Inizio Processo di Attestazione GPU ===")
    attestor = NvidiaAttestation()
    attestor.perform_attestation()
    print("\n=== Processo di Attestazione Completato ===")
    # Esce con 0 per indicare che lo script è stato eseguito, come da richiesta non bloccante.
    sys.exit(0)