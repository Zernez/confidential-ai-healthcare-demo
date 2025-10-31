"""
Attestation NVIDIA per ambienti cloud basata su nv_attestation_sdk.
"""
import json
import logging
import os
from nv_attestation_sdk import attestation
from typing import Dict, Any, Optional
import secrets

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("attestation_local")

class NvidiaAttestation:
    """
    Classe per l'attestazione GPU NVIDIA utilizzando nv_attestation_sdk.
    """
    
    def __init__(self, nonce: Optional[str] = None):
        """
        Inizializza il client di attestazione.
        :param nonce: Un nonce esadecimale per la richiesta di attestazione. Se non fornito, ne viene generato uno.
        """
        self.client = attestation.Attestation()
        self.client.set_name("local_gpu_node")
        
        if nonce:
            self.nonce = nonce
        else:
            self.nonce = secrets.token_hex(32)
        
        self.client.set_nonce(self.nonce)
        self.client.set_claims_version("3.0")
        
        # Aggiunge il verifier per l'attestazione locale della GPU seguendo l'esempio ufficiale NVIDIA.
        # Formato: add_verifier(Device, Environment.LOCAL, OCSP_URL, RIM_URL)
        # Nota: non passare parametri extra (alcune versioni SDK si aspettano solo 4 argomenti per LOCAL).
        self.client.add_verifier(
            attestation.Devices.GPU,
            attestation.Environment.LOCAL,
            "",
            "",
        )
        
        logger.info("NvidiaAttestation client inizializzato.")
        logger.info(f"Nonce: {self.nonce}")

    def get_gpu_attestation(self) -> Dict[str, Any]:
        """
        Ottiene l'attestazione della GPU.
        Raccoglie le prove (evidence) e le attesta.
        """
        try:
            logger.info("Raccolta delle prove (evidence) dalla GPU...")
            evidence_list = self.client.get_evidence()
            logger.info("Prove raccolte con successo.")

            logger.info("Esecuzione dell'attestazione...")
            attestation_passed = self.client.attest(evidence_list)
            logger.info(f"Risultato attestazione: {attestation_passed}")

            if not attestation_passed:
                raise Exception("L'attestazione della GPU è fallita.")

            token = self.client.get_token()
            if not token:
                raise Exception("Impossibile ottenere il token di attestazione.")

            logger.info("Token di attestazione generato con successo.")
            
            # Decodifica il token per estrarre informazioni utili
            decoded_token = self.client.decode_token(token)

            attestation_data = {
                'attestation_passed': attestation_passed,
                'token': token,
                'decoded_token': decoded_token,
                'nonce': self.nonce,
                'attestation_type': 'hardware_local',
                'timestamp': self._get_timestamp()
            }
            
            return attestation_data

        except Exception as e:
            logger.error(f"Errore durante il processo di attestazione: {e}")
            return {'error': str(e), 'attestation_passed': False}

    def validate_token(self, token: str, policy_path: str) -> bool:
        """
        Valida un token di attestazione rispetto a una policy.

        :param token: Il token di attestazione da validare.
        :param policy_path: Il percorso del file JSON della policy di attestazione.
        :return: True se il token è valido, altrimenti False.
        """
        if not os.path.exists(policy_path):
            logger.error(f"File della policy non trovato a: {policy_path}")
            return False

        try:
            with open(policy_path, 'r') as f:
                policy = f.read()
            
            logger.info(f"Validazione del token con la policy: {policy_path}")
            
            # Alcune versioni dell'SDK accettano solo (policy) e usano l'ultimo token interno;
            # altre accettano (policy, token). Proviamo prima con la firma a un argomento.
            try:
                is_valid = self.client.validate_token(policy)
            except TypeError:
                # Fallback: versione SDK che richiede anche il token
                is_valid = self.client.validate_token(policy, token)
            
            logger.info(f"Risultato validazione: {is_valid}")
            return is_valid

        except Exception as e:
            logger.error(f"Errore durante la validazione del token: {e}")
            return False

    def _get_timestamp(self) -> str:
        """Ottiene il timestamp corrente in formato ISO."""
        from datetime import datetime
        return datetime.now().isoformat()

    def perform_attestation(self) -> bool:
        """
        Esegue l'attestazione completa e restituisce True se ha successo, altrimenti False.
        """
        try:
            attestation_data = self.get_gpu_attestation()
            
            if attestation_data.get('attestation_passed'):
                logger.info("Attestazione completata con successo.")
                return True
            else:
                logger.error(f"Attestazione fallita: {attestation_data.get('error', 'Errore sconosciuto')}")
                return False
            
        except Exception as e:
            logger.error(f"Errore critico durante l'attestazione: {e}")
            return False

def create_attestation_report() -> Dict[str, Any]:
    """
    Funzione helper per creare un report di attestazione.
    """
    attestation = NvidiaAttestation()
    attestation_result = attestation.get_gpu_attestation()
    
    report = {
        'status': 'success' if attestation_result.get('attestation_passed') else 'failure',
        'attestation': attestation_result,
        'gpu_ready': attestation_result.get('attestation_passed', False)
    }
    
    return report

if __name__ == "__main__":
    print("=== Test Attestazione NVIDIA con nv_attestation_sdk ===")
    
    try:
        report = create_attestation_report()
        print(json.dumps(report, indent=2))
        
        if report.get('gpu_ready'):
            print("\nGPU pronta e attestata per workload sicuri.")

            # --- Esempio di Validazione Token ---
            print("\n--- Inizio Validazione Token ---")
            
            # Usiamo la stessa istanza che ha generato il nonce per la validazione
            # In uno scenario reale, il nonce verrebbe passato al validatore
            attestation_instance = NvidiaAttestation(nonce=report['attestation']['nonce'])
            
            policy_file = "policy.json"
            
            # Crea un file di policy di esempio se non esiste.
            # ATTENZIONE: I valori devono essere adattati al tuo hardware specifico.
            # Puoi ottenere i valori corretti dal campo 'decoded_token' del report.
            if not os.path.exists(policy_file):
                print(f"Creazione di un file di policy di esempio: {policy_file}")
                # Estrai il driver version dal token decodificato per creare una policy funzionante
                driver_version_from_token = report['attestation']['decoded_token'].get('gpu_driver_version', '535.0.0')
                example_policy = {
                    "attestation_policy": {
                        "gpu.driver_version": driver_version_from_token
                    }
                }
                with open(policy_file, 'w') as f:
                    json.dump(example_policy, f, indent=2)
                print(f"Policy di esempio creata con driver_version: {driver_version_from_token}")

            token_to_validate = report['attestation']['token']
            is_token_valid = attestation_instance.validate_token(token_to_validate, policy_file)

            if is_token_valid:
                print("\nVALIDAZIONE TOKEN: Successo! Il token è valido e corrisponde alla policy.")
            else:
                print("\nVALIDAZIONE TOKEN: Fallita! Il token non è valido o non corrisponde alla policy.")
            # --- Fine Esempio Validazione ---

        else:
            print("\nAttestazione GPU fallita o GPU non pronta.")
            
    except Exception as e:
        print(f"Errore durante l'esecuzione del test di attestazione: {e}")
