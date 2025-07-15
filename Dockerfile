# --- Stage 1: Build Stage ---
# 使用 Python 3.10-slim 作為建置環境
FROM python:3.10-slim as builder

# 安裝 poetry
RUN pip install pipx
RUN pipx install poetry

# 將 poetry 的配置設為不在專案目錄內創建虛擬環境
RUN poetry config virtualenvs.create false

# 設置工作目錄
WORKDIR /app

# 複製專案依賴定義檔
COPY poetry.lock pyproject.toml ./

# 先按照 lock 文件安裝所有依賴，包括 faiss-cpu
RUN poetry install --no-root --no-dev --no-interaction --no-ansi

# 替換掉 faiss-cpu
RUN pip uninstall -y faiss-cpu
RUN pip install faiss-gpu

# --- Stage 2: Final Stage ---
# 同樣使用 Python 3.10-slim 作為最終的運行環境
FROM python:3.10-slim

WORKDIR /app

# 從 builder 階段複製已安裝好的依賴
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages

# 複製應用程式碼
COPY . .

# 安裝 supervisor
RUN apt-get update && apt-get install -y supervisor

# 將我們的 supervisor 設定檔複製到容器的正確位置
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# 容器啟動時執行的命令
# 啟動 supervisord，-n 參數表示在前台運行，這是 Docker 容器所需要的
CMD ["/usr/bin/supervisord", "-n", "-c", "/etc/supervisor/conf.d/supervisord.conf"]