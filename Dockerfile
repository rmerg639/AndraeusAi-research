# Andraeus AI - Reproducible Environment
# Copyright (c) 2025 Rocco Andraeus Sergi. All Rights Reserved.
#
# Build: docker build -t andraeus-ai .
# Run:   docker run --gpus all -it andraeus-ai

FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-venv \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Create working directory
WORKDIR /app

# Copy requirements first (for layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Install package in development mode
RUN pip install -e .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Default command
CMD ["python", "-c", "from andraeus import AndraeusConfig; print('Andraeus AI ready!')"]

# =============================================================================
# USAGE EXAMPLES
# =============================================================================
#
# Build the image:
#   docker build -t andraeus-ai .
#
# Run interactive shell with GPU:
#   docker run --gpus all -it andraeus-ai bash
#
# Run training:
#   docker run --gpus all -v $(pwd)/output:/app/output andraeus-ai \
#       python train_personal_ai.py
#
# Run evaluation:
#   docker run --gpus all andraeus-ai \
#       python evaluation/run_generalization_test.py
#
# =============================================================================
