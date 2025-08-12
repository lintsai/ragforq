# --- Stage 1: Build Stage ---
FROM python:3.10-slim as builder

# 設定 TERM 環境變數（避免非互動式環境出錯）
ENV TERM=xterm

# 安裝系統依賴
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
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

# 替換掉 faiss-cpu 為 GPU 版本（如果存在），並確保 numpy 不被升級
RUN . .venv/bin/activate && pip uninstall -y faiss-cpu || true
RUN . .venv/bin/activate && pip install --no-cache-dir faiss-gpu

# 安裝生產環境專用依賴（開發環境因兼容性問題移除的依賴），並在之後再鎖一次 numpy 版本
RUN . .venv/bin/activate && pip install --no-cache-dir "vllm>=0.2.0" "ray>=2.8.0"
RUN . .venv/bin/activate && pip install --no-cache-dir "numpy<2" --upgrade --force-reinstall


# --- Stage 2: Final Stage ---
FROM python:3.10-slim

# 安裝系統依賴
RUN apt-get update && apt-get install -y \
    supervisor \
    curl \
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

# 容器啟動時執行的命令
CMD ["/usr/bin/supervisord", "-n", "-c", "/etc/supervisor/conf.d/supervisord.conf"]