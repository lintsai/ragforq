# --- Stage 1: Build Stage ---
FROM python:3.10-slim AS builder

# Build argument: Enable GPU support
ARG ENABLE_GPU=false

# Set TERM environment variable
ENV TERM=xterm

# Install system dependencies, including build tools
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install poetry
RUN pip install poetry

# Configure poetry to create virtualenvs in the project directory
RUN poetry config virtualenvs.in-project true

# Set working directory
WORKDIR /app

# Copy project dependency files
COPY poetry.lock pyproject.toml ./

# Install project dependencies using poetry
RUN poetry install --no-root --no-interaction --no-ansi

# Activate venv and run all subsequent pip commands in a single RUN block for consistency
RUN . .venv/bin/activate && \
    pip install --no-cache-dir "numpy<2" --upgrade --force-reinstall && \
    pip install --no-cache-dir "huggingface_hub[hf_xet]" && \
    pip install --no-cache-dir "transformers>=4.35.0"

# Conditional installation of GPU/CPU packages
RUN . .venv/bin/activate && \
    if [ "$ENABLE_GPU" = "true" ]; then \
        echo "🔧 Installing GPU versions of PyTorch, FAISS, and quantization libraries..."; \
        pip uninstall -y torch torchvision torchaudio faiss-cpu || true; \
        pip install --no-cache-dir \
            "torch>=2.0.0" torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
        pip install --no-cache-dir \
            "faiss-gpu>=1.7.0" \
            "bitsandbytes>=0.43.2" \
            "triton>=2.1.0,<3.0.0" \
            "vllm>=0.5.1" \
            "ray>=2.20.0"; \
    else \
        echo "🔧 Installing CPU versions of PyTorch and FAISS..."; \
        pip uninstall -y faiss-gpu || true; \
        pip install --no-cache-dir \
            "torch>=2.0.0" torchvision torchaudio \
            "faiss-cpu>=1.7.0"; \
    fi

# Final numpy version check
RUN . .venv/bin/activate && pip install --no-cache-dir "numpy>=1.24.0,<2" --upgrade --force-reinstall


# --- Stage 2: Final Stage ---
FROM python:3.10-slim

# Build argument: Enable GPU support
ARG ENABLE_GPU=false

# Debug: Show ENABLE_GPU value
RUN echo "🔍 Final Stage - ENABLE_GPU=$ENABLE_GPU"

# Install runtime system dependencies
RUN apt-get update && apt-get install -y \
    supervisor \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
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
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# Container start command with detailed GPU check
CMD ["sh", "-c", ". /app/.venv/bin/activate && echo '🚀 RAG for Q Container Starting...' && echo '🔍 Checking GPU support...' && python -c \"import torch; print(f'PyTorch Version: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU Count: {torch.cuda.device_count()}' if torch.cuda.is_available() else 'GPU Count: 0'); [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else None; import bitsandbytes; print('bitsandbytes imported successfully.')\" && echo '🎯 Starting services...' && exec /usr/bin/supervisord -n -c /etc/supervisor/conf.d/supervisord.conf"]
