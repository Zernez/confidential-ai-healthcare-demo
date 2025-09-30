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
docker run --gpus all --name $CONTAINER_NAME \
    -v $(pwd):/app \
    -w /app \
    $IMAGE_NAME \
    conda run -n rapids python main.py
