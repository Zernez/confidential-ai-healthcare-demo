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

# Crea ambiente RAPIDS
conda create -y -n rapids python=3.10 -c rapidsai -c conda-forge
source $(conda info --base)/etc/profile.d/conda.sh
conda activate rapids
conda install -y -c rapidsai -c conda-forge cudf=24.06 cuml=24.06 cupy

# Attiva ambiente

echo "Ambiente rapids pronto! Ora puoi eseguire run_local.sh."
