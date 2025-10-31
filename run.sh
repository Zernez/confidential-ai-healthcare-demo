#!/bin/bash
set -e

# Nome immagine e container
IMAGE_NAME=conf-ai-demo
CONTAINER_NAME=conf-ai-container

echo "Costruzione immagine Docker..."
docker build -f docker/Dockerfile -t $IMAGE_NAME .

# Se esiste già un container con lo stesso nome, lo rimuovo
if [ "$(docker ps -aq -f name=$CONTAINER_NAME)" ]; then
    echo "Rimuovo container esistente $CONTAINER_NAME"
    docker rm -f $CONTAINER_NAME
fi

echo "Avvio container con GPU..."

# Mappa tutti i device NVIDIA visibili
EXTRA_DEVICES=""
for DEV in /dev/nvidia*; do
    if [ -e "$DEV" ]; then
        EXTRA_DEVICES="$EXTRA_DEVICES --device $DEV"
    fi
done

source /opt/conda/etc/profile.d/conda.sh
conda activate rapids-25.10

# Esegui attestazione sull'host prima di avviare il container
python3 attestation.py || { echo "Attestazione fallita sull'host. Blocco esecuzione."; exit 1; }

docker run --gpus all --name $CONTAINER_NAME \
    -v $(pwd):/app \
    -w /app \
    $EXTRA_DEVICES \
    $IMAGE_NAME \
    bash -c "conda run -n rapids-25.10 python main.py"
