#!/bin/bash
set -e

# Vai nella cartella del progetto
cd "$(dirname "$0")"

# Fix compatibilità Numba/CUDA per RAPIDS 
# export NUMBA_CUDA_ENABLE_MINOR_VERSION_COMPATIBILITY=1
# export CUBINLINKER_DISABLE_PATCH=1

echo "Attivare l'environment Conda (rapids) prima dell'esecuzione."

echo "Avvio main.py..."

# if [ -f "docker/requirements.txt" ]; then
#     echo "Installo dipendenze da requirements.txt..."
#     python3 -m pip install --upgrade pip
#     python3 -m pip install -r docker/requirements.txt
# fi

# Esegui attestazione sull'host prima di avviare main.py
python3 attestation/attestation.py
if [ $? -ne 0 ]; then
    echo "Attestazione fallita sull'host. Blocco esecuzione."
    exit 1
fi

python3 python-native/src/main.py
