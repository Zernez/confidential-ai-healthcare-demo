# Base CUDA runtime
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

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
RUN conda create -y -n rapids python=3.10 && \
    conda install -y -n rapids -c rapidsai -c conda-forge \
      cudf=24.06 cuml=24.06 cupy cuda-version=12.2 && \
    conda install -y -n rapids -c conda-forge \
      pip joblib requests && \
    conda clean -a -y

WORKDIR /app

# Copy project
COPY ../project .

# Optional: install any python-only requirements in the env
# RUN conda run -n rapids pip install -r requirements.txt

# Default command: run main in RAPIDS env
CMD ["conda", "run", "-n", "rapids", "python", "main.py"]
