#!/bin/bash
set -e

# Installa Miniconda se non presente
if ! command -v conda &> /dev/null; then
    echo "Miniconda non trovato. Installazione..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p $HOME/miniconda
    export PATH="$HOME/miniconda/bin:$PATH"
    echo 'export PATH="$HOME/miniconda/bin:$PATH"' >> ~/.bashrc
fi

# Attiva Conda
source $(conda info --base)/etc/profile.d/conda.sh

# Rimuovi ambiente esistente se presente
if conda info --envs | grep -q rapids; then
    echo "Rimuovo ambiente rapids esistente..."
    conda remove -n rapids --all -y
fi

# Crea ambiente RAPIDS con Python 3.11 (più recente e stabile)
echo "Creazione ambiente rapids con Python 3.11..."
conda create -y -n rapids python=3.11
source $(conda info --base)/etc/profile.d/conda.sh
conda activate rapids

# Installa RAPIDS più recenti compatibili con Python 3.11
echo "Installazione RAPIDS 24.04 (Python 3.11 + Numba recente)..."
conda install -y -c rapidsai -c conda-forge -c nvidia cubinlinker ptxcompiler
conda install -y -c rapidsai -c conda-forge -c nvidia cudf=24.04 cuml=24.04

# Installa requirements.txt con pip
# if [ -f "docker/requirements.txt" ]; then
#     echo "Installazione dipendenze da requirements.txt..."
#     pip install -r docker/requirements.txt
# fi

echo "Ambiente rapids pronto! Ora puoi eseguire run_local.sh."
