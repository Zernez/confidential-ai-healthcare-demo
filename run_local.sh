#!/bin/bash
set -e

# Vai nella cartella del progetto
cd "$(dirname "$0")"

echo "Attivare l'environment Conda (rapids) prima dell'esecuzione."

echo "Avvio main.py..."

if [ -f "docker/requirements.txt" ]; then
    echo "Installo dipendenze da requirements.txt..."
    python3 -m pip install --upgrade pip
    python3 -m pip install -r docker/requirements.txt
fi

python3 main.py
