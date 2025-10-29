#!/bin/bash
set -e

# Vai nella cartella del progetto
cd /home/azureuser/project

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

# Mappa device speciali NVIDIA per confidential computing se presenti
EXTRA_DEVICES=""
for DEV in /dev/nvidia-cc /dev/nvidia-uvm /dev/nvidia-uvm-tools; do
    if [ -e "$DEV" ]; then
        EXTRA_DEVICES="$EXTRA_DEVICES --device $DEV"
    fi
done

docker run --gpus all --name $CONTAINER_NAME \
    -v $(pwd):/app \
    -w /app \
    $EXTRA_DEVICES \
    $IMAGE_NAME \
    conda run -n rapids python main.py
