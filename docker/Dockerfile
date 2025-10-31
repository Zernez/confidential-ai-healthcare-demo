# Base CUDA runtime
FROM nvidia/cuda:13.0.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    CONDA_DIR=/opt/conda \
    PATH=/opt/conda/bin:$PATH

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget bzip2 ca-certificates git build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p $CONDA_DIR && rm -f /tmp/miniconda.sh


# Create RAPIDS env (CUDA 12.2), install cudf/cuml/cupy + essentials
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
        conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Crea ambiente RAPIDS 25.10 con Python 3.12 e CUDA 13.0
RUN conda create -y -n rapids-25.10 -c rapidsai -c conda-forge -c nvidia \
        rapids=25.10 python=3.12 cuda-version=13.0 && \
        conda install -y -n rapids-25.10 -c rapidsai -c conda-forge \
            cudf=25.10 cuml=25.10 cuda-version=13.0 && \
        conda install -y -n rapids-25.10 -c conda-forge \
            pip joblib requests && \
        conda clean -a -y

WORKDIR /app

# Copy project
COPY . /app

# Optional: install any python-only requirements in the env
COPY docker/requirements.txt /app/requirements.txt
RUN conda init
RUN conda activate rapids-25.10
RUN pip install -r /app/requirements.txt
#RUN conda run -n rapids-25.10 pip install -r /app/requirements.txt

# Default command: run main in RAPIDS env
CMD ["python", "main.py"]
