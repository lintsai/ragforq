#!/bin/bash

# 瑽遣??撱箇? base image ?單
# ??image ???之??鞈游?隞塚??臭誑憭批?皜? CI/CD ??

set -e

echo "? Building base image with pre-installed dependencies..."

# ?萄遣?冽? Dockerfile
cat > Dockerfile.base << 'EOF'
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV TERM=xterm
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-venv \
    python3-pip \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Ensure python3 is linked to python
RUN ln -sf /usr/bin/python3 /usr/bin/python

# Install poetry
RUN pip install poetry

# Configure poetry
RUN poetry config virtualenvs.in-project true

# Set working directory
WORKDIR /app

# Create virtual environment
RUN python3 -m venv .venv && \
    . .venv/bin/activate && \
    pip install --upgrade pip setuptools wheel

# Install common ML packages that rarely change
RUN . .venv/bin/activate && \
    pip install --no-cache-dir \
        "numpy>=1.24.0,<2" \
        "huggingface_hub[hf_xet]" \
        "transformers>=4.35.0" \
        "torch>=2.0.0" torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 \
        "faiss-gpu>=1.7.0" \
        "bitsandbytes>=0.43.2" \
        "triton>=2.1.0,<3.0.0" \
        "vllm>=0.5.1" \
        "ray>=2.20.0"

# Clean up
RUN apt-get clean && rm -rf /var/lib/apt/lists/*
EOF

# 瑽遣 base image
BASE_IMAGE_NAME="ragforq-base:$(date +%Y%m%d)"
echo "Building base image: $BASE_IMAGE_NAME"

docker build -f Dockerfile.base -t $BASE_IMAGE_NAME .
docker tag $BASE_IMAGE_NAME ragforq-base:latest

echo "??Base image built successfully: $BASE_IMAGE_NAME"
echo "? To use this base image, update your main Dockerfile to use:"
echo "   FROM $BASE_IMAGE_NAME AS builder"

# 皜??冽??辣
rm Dockerfile.base

echo "?? You can now push this base image to your registry and use it in CI/CD"
