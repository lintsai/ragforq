# --- Stage 1: Build Stage ---
FROM python:3.10-slim AS builder

# 構建參數：是否啟用 GPU 支援
ARG ENABLE_GPU=false

# 設定 TERM 環境變數（避免非互動式環境出錯）
ENV TERM=xterm

# 安裝系統依賴
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# 安裝 poetry
RUN pip install poetry

# 將 poetry 的配置設為在專案目錄內創建虛擬環境
RUN poetry config virtualenvs.in-project true

# 設置工作目錄
WORKDIR /app

# 複製專案依賴定義檔
COPY poetry.lock pyproject.toml ./

# 安裝專案依賴
RUN poetry install --no-root --no-interaction --no-ansi

# 先鎖定 numpy < 2，避免後續套件將其升級到 2.x
RUN . .venv/bin/activate && pip install --no-cache-dir "numpy<2" --upgrade --force-reinstall

# 根據 ENABLE_GPU 參數選擇 PyTorch 和 FAISS 版本
RUN if [ "$ENABLE_GPU" = "true" ]; then \
        echo "🔧 安裝 GPU 版本的 PyTorch 和 FAISS..."; \
        . .venv/bin/activate && pip uninstall -y torch torchvision torchaudio || true; \
        . .venv/bin/activate && pip install --no-cache-dir "torch>=2.0.0" torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121; \
        . .venv/bin/activate && pip uninstall -y faiss-cpu || true; \
        . .venv/bin/activate && pip install --no-cache-dir "faiss-gpu>=1.8.0"; \
    else \
        echo "🔧 安裝 CPU 版本的 PyTorch 和 FAISS..."; \
        . .venv/bin/activate && pip install --no-cache-dir "torch>=2.0.0" torchvision torchaudio; \
        . .venv/bin/activate && pip uninstall -y faiss-gpu || true; \
        . .venv/bin/activate && pip install --no-cache-dir "faiss-cpu>=1.8.0"; \
    fi

# 根據 GPU 支援安裝 vLLM（vLLM 需要 GPU）
RUN if [ "$ENABLE_GPU" = "true" ]; then \
        echo "🚀 Installing vLLM (GPU version)..."; \
        . .venv/bin/activate && pip install --no-cache-dir "vllm>=0.5.1"; \
        echo "🚀 Installing Ray (GPU version)..."; \
        . .venv/bin/activate && pip install --no-cache-dir "ray>=2.20.0"; \
        echo "🔧 Installing bitsandbytes for quantization..."; \
        . .venv/bin/activate && pip install --no-cache-dir "bitsandbytes>=0.43.2"; \
    else \
        echo "⚠️ Skipping vLLM installation (requires GPU support)"; \
        echo "🔧 Installing bitsandbytes (CPU fallback)..."; \
        . .venv/bin/activate && pip install --no-cache-dir "bitsandbytes>=0.43.2" || echo "⚠️ bitsandbytes installation failed (expected on CPU-only)"; \
    fi

# 安裝 HuggingFace 相關依賴以支援大型模型下載
RUN . .venv/bin/activate && pip install --no-cache-dir "huggingface_hub[hf_xet]"

# 確保使用與 pyproject.toml 一致的 transformers 版本
RUN . .venv/bin/activate && pip install --no-cache-dir "transformers>=4.35.0"

# 確保 numpy 版本一致
RUN . .venv/bin/activate && pip install --no-cache-dir "numpy>=1.24.0,<2" --upgrade --force-reinstall


# --- Stage 2: Final Stage ---
FROM python:3.10-slim

# 安裝系統依賴和 CUDA 相關工具
RUN apt-get update && apt-get install -y \
    supervisor \
    curl \
    wget \
    gnupg2 \
    && rm -rf /var/lib/apt/lists/*

# 將 poetry 也安裝到最終鏡像中，以便使用 poetry run
RUN pip install poetry

# 設置工作目錄
WORKDIR /app

# 從 builder 階段複製整個專案，包括完整的虛擬環境 .venv
COPY --from=builder /app /app

# 複製應用程式碼
COPY . .

# 複製 production 環境變數為預設 .env
COPY .env.production .env

# 創建必要的目錄
RUN mkdir -p /app/logs /app/db /app/models/cache

# 將我們的 supervisor 設定檔複製到容器的正確位置
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# 設置權限
RUN chmod +x /app/app.py

# 暴露端口
EXPOSE 8000 8501

# 健康檢查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# 容器啟動時執行的命令 - 包含詳細的 GPU 檢測
CMD ["sh", "-c", ". /app/.venv/bin/activate && echo '🚀 RAG for Q 容器啟動中...' && echo '🔍 檢查 GPU 支援...' && python -c \"import torch; print(f'PyTorch 版本: {torch.__version__}'); print(f'CUDA 可用: {torch.cuda.is_available()}'); print(f'GPU 數量: {torch.cuda.device_count()}' if torch.cuda.is_available() else 'GPU 數量: 0'); [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else None\" && echo '🎯 啟動服務...' && exec /usr/bin/supervisord -n -c /etc/supervisor/conf.d/supervisord.conf"]