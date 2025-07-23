# --- Stage 1: Build Stage ---
FROM python:3.10-slim as builder

# 設定 TERM 環境變數（避免非互動式環境出錯）
ENV TERM=xterm

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

# 替換掉 faiss-cpu 為 GPU 版本
RUN . .venv/bin/activate && pip uninstall -y faiss-cpu
RUN . .venv/bin/activate && pip install faiss-gpu


# --- Stage 2: Final Stage ---
FROM python:3.10-slim

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

# 安裝 supervisor
RUN apt-get update && apt-get install -y supervisor

# 將我們的 supervisor 設定檔複製到容器的正確位置
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# 容器啟動時執行的命令
CMD ["/usr/bin/supervisord", "-n", "-c", "/etc/supervisor/conf.d/supervisord.conf"]