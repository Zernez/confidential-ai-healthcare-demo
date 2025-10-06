#!/bin/bash
set -e

# Vai nella cartella del progetto
cd "$(dirname "$0")"

echo "Attivo ambiente Conda (rapids) se disponibile..."
if command -v conda &> /dev/null && conda info --envs | grep -q rapids; then
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate rapids
else
    echo "[WARN] Ambiente 'rapids' non trovato. Uso Python di sistema."
fi

echo "Avvio main.py..."

if [ -f "docker/requirements.txt" ]; then
    echo "Installo dipendenze da requirements.txt..."
    python3 -m pip install --upgrade pip
    python3 -m pip install -r docker/requirements.txt
fi

python3 main.py
