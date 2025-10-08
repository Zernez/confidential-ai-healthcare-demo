"""
Attestation NVIDIA semplificata per ambienti cloud
Fallback compatibile senza dipendenze da funzioni CC avanzate
"""
import json
import subprocess
import logging
from typing import Dict, Any, Optional

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("attestation_simple")

class NvidiaAttestation:
    """
    Classe per attestation GPU NVIDIA semplificata
    Usa nvidia-smi e pynvml base senza funzioni CC avanzate
    """
    
    def __init__(self):
        self.gpu_info = None
        self._initialize_gpu()
    
    def _initialize_gpu(self):
        """Inizializza informazioni GPU"""
        try:
            # Prova prima con pynvml
            import pynvml
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            
            if device_count > 0:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                # --- FIX: Rimosso .decode('utf-8') ---
                name = pynvml.nvmlDeviceGetName(handle)
                # ------------------------------------
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                
                self.gpu_info = {
                    'name': name,
                    'memory_total': memory_info.total,
                    'memory_free': memory_info.free,
                    'count': device_count
                }
                logger.info(f"GPU inizializzata con pynvml: {name}")
            
            pynvml.nvmlShutdown()
            
        except Exception as e:
            logger.warning(f"pynvml fallito: {e}, uso nvidia-smi")
            self._get_gpu_info_from_nvidia_smi()
    
    def _get_gpu_info_from_nvidia_smi(self):
        """Fallback con nvidia-smi"""
        try:
            result = subprocess.run([
                'nvidia-smi', '--query-gpu=name,memory.total,memory.free,count',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, check=True)
            
            if result.stdout.strip():
                line = result.stdout.strip().split('\n')[0]
                name, mem_total, mem_free, count = line.split(', ')
                
                self.gpu_info = {
                    'name': name,
                    'memory_total': int(mem_total) * 1024 * 1024,  # MB to bytes
                    'memory_free': int(mem_free) * 1024 * 1024,
                    'count': 1
                }
                logger.info(f"GPU info da nvidia-smi: {name}")
                
        except Exception as e:
            logger.error(f"Errore nvidia-smi: {e}")
            self.gpu_info = {'error': str(e)}
    
    def get_gpu_attestation(self) -> Dict[str, Any]:
        """
        Ottiene attestation GPU semplificata
        """
        if not self.gpu_info:
            return {'error': 'GPU non inizializzata'}
        
        # Crea attestation base con info disponibili
        attestation_data = {
            'gpu_info': self.gpu_info,
            'driver_version': self._get_driver_version(),
            'cuda_version': self._get_cuda_version(),
            'attestation_type': 'simplified',
            'timestamp': self._get_timestamp()
        }
        
        # Prova attestation avanzata se disponibile
        try:
            advanced_data = self._try_advanced_attestation()
            if advanced_data:
                attestation_data.update(advanced_data)
                attestation_data['attestation_type'] = 'advanced'
        except Exception as e:
            logger.info(f"Attestation avanzata non disponibile: {e}")
        
        return attestation_data
    
    def _try_advanced_attestation(self) -> Optional[Dict[str, Any]]:
        """Prova attestation avanzata con nv-attestation-sdk"""
        try:
            from nv_attestation_sdk.attestation import Attestation
            
            attestation = Attestation()
            evidence = attestation.get_evidence()
            
            return {
                'evidence': evidence,
                'nras_url': attestation.nras_url if hasattr(attestation, 'nras_url') else None
            }
            
        except ImportError:
            logger.info("nv-attestation-sdk non disponibile")
            return None
        except Exception as e:
            logger.warning(f"Attestation SDK fallita: {e}")
            return None
    
    def _get_driver_version(self) -> str:
        """Ottiene versione driver"""
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], 
                                    capture_output=True, text=True, check=True)
            return result.stdout.strip()
        except:
            return "unknown"
    
    def _get_cuda_version(self) -> str:
        """Ottiene versione CUDA"""
        try:
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, check=True)
            # Estrai versione CUDA dalla output di nvidia-smi
            lines = result.stdout.split('\n')
            for line in lines:
                if 'CUDA Version:' in line:
                    return line.split('CUDA Version:')[1].strip().split()[0]
        except:
            pass
        return "unknown"
    
    def _get_timestamp(self) -> str:
        """Ottiene timestamp corrente"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def verify_gpu_ready(self) -> bool:
        """Verifica se GPU è pronta per ML workload"""
        if not self.gpu_info or 'error' in self.gpu_info:
            return False
        
        # Verifica memoria minima (1GB)
        min_memory = 1024 * 1024 * 1024
        if self.gpu_info.get('memory_free', 0) < min_memory:
            logger.warning("Memoria GPU insufficiente")
            return False
        
        return True
    
    def perform_attestation(self) -> bool:
        """Esegue attestation completa e ritorna successo/fallimento"""
        try:
            attestation_data = self.get_gpu_attestation()
            
            # Log attestation info
            logger.info(f"Attestation GPU: {self.gpu_info.get('name', 'Unknown')}")
            logger.info(f"Driver: {attestation_data.get('driver_version', 'Unknown')}")
            logger.info(f"CUDA: {attestation_data.get('cuda_version', 'Unknown')}")
            logger.info(f"Tipo: {attestation_data.get('attestation_type', 'Unknown')}")
            
            gpu_ready = self.verify_gpu_ready()
            if gpu_ready:
                logger.info("✓ Attestation completata con successo")
            else:
                logger.error("✗ GPU non pronta per ML workload")
            
            return gpu_ready
            
        except Exception as e:
            logger.error(f"Errore durante attestation: {e}")
            return False

def create_attestation_report() -> Dict[str, Any]:
    """Funzione helper per creare report attestation"""
    attestation = NvidiaAttestation()
    
    report = {
        'status': 'success' if attestation.verify_gpu_ready() else 'warning',
        'attestation': attestation.get_gpu_attestation(),
        'gpu_ready': attestation.verify_gpu_ready()
    }
    
    return report

if __name__ == "__main__":
    # Test dell'attestation
    print("=== Test Attestation NVIDIA ===")
    
    try:
        report = create_attestation_report()
        print(json.dumps(report, indent=2))
        
        if report['gpu_ready']:
            print("\n✓ GPU pronta per ML workload")
        else:
            print("\n⚠ GPU non pronta")
            
    except Exception as e:
        print(f"Errore: {e}")