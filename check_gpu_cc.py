#!/usr/bin/env python3
"""
Script per verificare se la GPU supporta Confidential Computing
"""
import subprocess
import sys

def check_nvidia_smi():
    """Verifica informazioni GPU tramite nvidia-smi"""
    try:
        # Verifica driver e GPU
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, check=True)
        print("=== nvidia-smi ===")
        print(result.stdout)
        
        # Verifica confidential computing specificamente
        result = subprocess.run(['nvidia-smi', '-q', '-d', 'COMPUTE'], capture_output=True, text=True, check=True)
        print("\n=== GPU Compute Info ===")
        print(result.stdout)
        
    except subprocess.CalledProcessError as e:
        print(f"Errore nvidia-smi: {e}")
        return False
    except FileNotFoundError:
        print("nvidia-smi non trovato - driver non installato?")
        return False
    return True

def check_pynvml():
    """Verifica pynvml e funzioni disponibili"""
    try:
        import pynvml
        print("\n=== pynvml check ===")
        
        # Inizializza NVML
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        print(f"GPU trovate: {device_count}")
        
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            print(f"GPU {i}: {name}")
            
            # Verifica se la funzione CC esiste
            if hasattr(pynvml, 'nvmlDeviceGetConfComputeGpuAttestationReport'):
                print(f"  ✓ Funzione CC attestation disponibile")
                try:
                    # Prova a chiamarla (potrebbe fallire se CC non è abilitato)
                    report = pynvml.nvmlDeviceGetConfComputeGpuAttestationReport(handle)
                    print(f"  ✓ CC attestation report ottenuto: {len(report)} bytes")
                except Exception as e:
                    print(f"  ⚠ CC attestation non disponibile: {e}")
            else:
                print(f"  ✗ Funzione CC attestation NON disponibile")
                
        pynvml.nvmlShutdown()
        
    except ImportError as e:
        print(f"Errore import pynvml: {e}")
        return False
    except Exception as e:
        print(f"Errore pynvml: {e}")
        return False
    return True

def check_cc_tools():
    """Verifica tool confidential computing"""
    tools = ['nvidia-cc-tool', 'nvmlmon']
    for tool in tools:
        try:
            result = subprocess.run([tool, '--help'], capture_output=True, text=True, check=True)
            print(f"\n✓ {tool} disponibile")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print(f"\n✗ {tool} non disponibile")

if __name__ == "__main__":
    print("=== Verifica Supporto GPU Confidential Computing ===\n")
    
    # Verifica nvidia-smi
    if not check_nvidia_smi():
        sys.exit(1)
    
    # Verifica pynvml
    if not check_pynvml():
        print("\nProvo ad installare nvidia-ml-py più recente...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--upgrade', 'nvidia-ml-py'])
        check_pynvml()
    
    # Verifica tool CC
    check_cc_tools()
    
    print("\n=== Fine verifica ===")