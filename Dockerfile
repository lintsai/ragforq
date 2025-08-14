# --- Dynamic Base Image Configuration ---
# Define build arguments to allow for dynamic base images.
# Defaults to a lightweight CPU setup.
ARG BUILDER_IMAGE=python:3.10-slim
ARG FINAL_IMAGE=python:3.10-slim
ARG ENABLE_GPU=false

# --- Stage 1: Build Stage ---
FROM ${BUILDER_IMAGE} AS builder

# Re-declare ARGs within the build stage to make them available.
ARG ENABLE_GPU

# Set TERM environment variable and prevent interactive prompts during build
ENV TERM=xterm 
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies. This list is compatible with both base images.
# The NVIDIA image is Ubuntu-based, python:slim is Debian-based. Both use apt-get.
RUN apt-get update && apt-get install -y 
    python3.10 
    python3.10-venv 
    python3-pip 
    build-essential 
    curl 
    git 
    && rm -rf /var/lib/apt/lists/*

# Ensure python3 is linked to python
RUN ln -sf /usr/bin/python3 /usr/bin/python

# Install poetry
RUN pip install poetry

# Configure poetry to create virtualenvs in the project directory
RUN poetry config virtualenvs.in-project true

# Set working directory
WORKDIR /app

# Copy project dependency files
COPY poetry.lock pyproject.toml ./

# Create a virtual environment and install project dependencies
RUN python -m venv .venv && 
    . .venv/bin/activate && 
    poetry install --no-root --no-interaction --no-ansi

# Activate venv and run all subsequent pip commands in a single RUN block for consistency
RUN . .venv/bin/activate && 
    pip install --no-cache-dir "numpy<2" --upgrade --force-reinstall && 
    pip install --no-cache-dir "huggingface_hub[hf_xet]" && 
    pip install --no-cache-dir "transformers>=4.35.0"

# Conditional installation of GPU/CPU packages based on the ENABLE_GPU flag
RUN . .venv/bin/activate && 
    if [ "$ENABLE_GPU" = "true" ]; then 
        echo "🔧 Installing GPU versions of PyTorch, FAISS, and quantization libraries..."; 
        pip uninstall -y torch torchvision torchaudio faiss-cpu || true; 
        pip install --no-cache-dir 
            "torch>=2.0.0" torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && 
        pip install --no-cache-dir 
            "faiss-gpu>=1.7.0" 
            "bitsandbytes>=0.43.2" 
            "triton>=2.1.0,<3.0.0" 
            "vllm>=0.5.1" 
            "ray>=2.20.0"; 
    else 
        echo "🔧 Installing CPU versions of PyTorch and FAISS..."; 
        pip uninstall -y faiss-gpu || true; 
        pip install --no-cache-dir 
            "torch>=2.0.0" torchvision torchaudio 
            "faiss-cpu>=1.7.0"; 
    fi

# Final numpy version check
RUN . .venv/bin/activate && pip install --no-cache-dir "numpy>=1.24.0,<2" --upgrade --force-reinstall


# --- Stage 2: Final Stage ---
FROM ${FINAL_IMAGE} AS final

# Re-declare ARG to use it for the debug echo
ARG ENABLE_GPU

# Debug: Show which version is being run
RUN echo "🔍 Final Stage - ENABLE_GPU=$ENABLE_GPU"

# Install runtime system dependencies
RUN apt-get update && apt-get install -y 
    python3.10 
    python3-pip 
    supervisor 
    curl 
    wget 
    && rm -rf /var/lib/apt/lists/*

# Ensure python3 is linked to python
RUN ln -sf /usr/bin/python3 /usr/bin/python

# Set environment variables for CUDA (will be unused in CPU image but harmless)
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Set working directory
WORKDIR /app

# Copy the entire project from the builder stage, including the .venv
COPY --from=builder /app /app

# Copy application code
COPY . .

# Copy production environment file to .env
COPY .env.production .env

# Create necessary directories
RUN mkdir -p /app/logs /app/db /app/models/cache

# Copy supervisor configuration
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Set permissions
RUN chmod +x /app/app.py

# Expose ports
EXPOSE 8000 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 
    CMD curl -f http://localhost:8000/ || exit 1

# Container start command
CMD ["sh", "-c", ". /app/.venv/bin/activate && echo '🚀 RAG for Q Container Starting...' && echo '🔍 Checking GPU support...' && python -c \"import torch; print(f'PyTorch Version: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU Count: {torch.cuda.device_count()}' if torch.cuda.is_available() else 'GPU Count: 0'); [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else None\" && echo '🎯 Starting services...' && exec /usr/bin/supervisord -n -c /etc/supervisor/conf.d/supervisord.conf"]

