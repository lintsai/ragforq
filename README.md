# Q槽文件智能助手 (Q-Drive RAG Assistant)

基於 RAG 技術的企業級智能問答系統，專門用於檢索和查詢公司文件。

## ✨ 功能特點

- 🔍 **智能檢索**：基於語義搜索的文檔檢索
- 📄 **多格式支援**：PDF、Word、Excel、PowerPoint、文本文件等
- 🤖 **AI問答**：基於大型語言模型的智能問答
- 🌐 **Web界面**：簡潔直觀的用戶界面
- 🔧 **模型管理**：支援多模型管理和動態選擇
- ✏️ **內容維護**：可直接編輯向量資料庫中的文檔內容
- 🔐 **企業級安全**：完全本地部署，數據不外流

## 🏢 企業級特性

**完全本地化部署**：

- ✅ 所有 AI 推理在企業內部進行
- ✅ 零數據外流，不會向任何外部服務發送數據
- ✅ 離線運行，模型下載後可完全離線使用
- ✅ 符合企業數據安全和隱私要求

**支援的 AI 平台**：

- **Ollama**：本地推理，隱私保護，適合大多數場景
- **Hugging Face**：豐富模型，本地部署，適合實驗研究

## 🚀 快速開始

### 方法一：一鍵啟動（推薦）

```bash
poetry run python scripts/quick_start.py
```

### 方法二：手動啟動

```bash
# 1. 安裝依賴
pip install -r requirements.txt

# 2. 啟動服務
poetry run python app.py  # API 服務
poetry run streamlit run frontend/streamlit_app.py  # 前端服務
```

### 使用流程

1. **訪問系統**：http://localhost:8501
2. **完成設置**：選擇 AI 平台、模型和配置
3. **開始問答**：輸入問題獲得智能回答

## 🐳 Docker 部署

### GPU 支援

系統會自動檢測 GPU 可用性：

- **有 GPU**：容器啟動時會顯示 GPU 資訊並自動使用 GPU 加速
- **無 GPU**：自動回退到 CPU 模式，功能完全正常但速度較慢

**GPU 支援要求**：

1. 主機安裝 NVIDIA 驅動
2. 安裝 NVIDIA Container Toolkit：
   ```bash
   # Ubuntu/Debian 快速安裝
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   sudo apt-get update && sudo apt-get install -y nvidia-docker2
   sudo systemctl restart docker
   ```

**測試 GPU 支援**：

```bash
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

### 開發環境啟動

```bash
# 1. 停止現有容器
docker stop ragforq-test-container

# 2. 構建本地測試映像
docker build -t ragforq-local-test .

# 3. 運行開發容器
docker run --rm -d -p 8501:8501 -p 8000:8000 --name ragforq-test-container -v D:\source\ragforq\.env.local:/app/.env -v D:\data:/q_drive_data/MIC共用文件庫/05_MIC專案 -v D:\source\ragforq\vector_db:/app/vector_db -v D:\source\ragforq\models:/app/models -v D:\source\ragforq\backups:/app/backups -v D:\source\ragforq\logs:/app/logs ragforq-local-test
```

### 測試環境啟動 (支援 GPU)

```bash
# 1. 停止現有容器
docker stop ragforq-test-container

# 2. 構建本地測試映像
docker build --build-arg ENABLE_GPU=true -t ragforq-local-test .

# 3. 運行開發容器
docker run --rm -d --gpus all -p 8501:8501 -p 8000:8000 --name ragforq-test-container -v C:\Users\user\source\ragforq\.env.local:/app/.env -v C:\Users\user\source\ragforq:/ragforq -v C:\Users\user\source\ragforq\vector_db:/app/vector_db -v C:\Users\user\source\ragforq\models:/app/models -v C:\Users\user\source\ragforq\backups:/app/backups -v C:\Users\user\source\ragforq\logs:/app/logs ragforq-local-test
```

### 生產環境部署 (支援 GPU)

```bash
# 構建映像
docker build --build-arg ENABLE_GPU=true -t ragforq .

# 運行容器
  ragforq
docker run -d \
  --gpus all \
  --restart always \
  --network=host \
  --name ragforq \
  -v ~/rag_data/db:/app/db \
  -v ~/rag_data/documents:/app/documents \
  -v ~/rag_data/models:/app/models \
  -v ~/rag_data/backups:/app/backups \
  -v ~/ragforq_logs:/app/logs \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v /mnt/winshare/MIC:/mnt/winshare/MIC \
  -e ENVIRONMENT=production

# 如果只想使用特定 GPU
# docker run -d --gpus '"device=0,1"' --name ragforq ...
```

## 🎛️ 主要功能

### 智能問答

- 多語言支援（繁中、簡中、英文、泰文）
- 動態 RAG 和傳統 RAG 模式
- 來源文檔顯示和相關性說明

### 模型管理

- 支援不同模型組合的獨立向量資料庫
- 版本控制和並行訓練
- 智能模型選擇和狀態管理

### 內容維護

- 直接編輯向量資料庫內容
- 即時更新嵌入向量
- 支援新增、編輯、刪除文檔

### 系統監控

- 訓練進度監控
- 文件變更監控
- 故障診斷和恢復

## 📁 支援的文件格式

PDF, Word, Excel, PowerPoint, Markdown, 純文本, CSV, Visio

## 🔧 環境檢查

```bash
# 檢查依賴安裝
python scripts/check_dependencies.py

# 檢查 Hugging Face 環境
python scripts/check_hf_environment.py

# 檢查整體系統狀態
python tests/test_complete_system.py --check

# 檢查系統耦合狀態
python scripts/check_system_coupling.py
```

## 📚 文檔和參考資源

### 系統文檔

- **[docs/README.md](docs/README.md)** - 完整文檔目錄
- **[docs/enterprise_deployment.md](docs/enterprise_deployment.md)** - 企業級部署指南
- **[docs/huggingface_setup.md](docs/huggingface_setup.md)** - Hugging Face 平台設置

### 外部參考

- [Hugging Face 官方文檔](https://huggingface.co/docs)
- [Transformers 庫文檔](https://huggingface.co/docs/transformers)
- [vLLM 文檔](https://docs.vllm.ai/)

### 文檔版本說明

系統已升級為前端設置流程，部分基於環境變數配置的舊文檔已過時。請優先參考當前有效文檔。

## 🛠️ 故障排除

### 常見問題

- **服務無法啟動**：檢查端口佔用和依賴安裝
- **模型載入失敗**：檢查網路連接和磁盤空間
- **問答品質差**：檢查文檔索引和模型選擇

### 診斷工具

```bash
# 依賴檢查
python scripts/check_dependencies.py

# 系統檢查
python tests/test_complete_system.py --check

# 環境檢查
python scripts/check_hf_environment.py

# Docker 容器內 GPU 檢查
python scripts/check_gpu_docker.py
```

### GPU 相關問題

**問題：顯示 "使用 CPU" 而非 GPU**

解決方案：

1. 確保 Docker 運行時使用 `--gpus all` 參數
2. 檢查 NVIDIA Container Toolkit 是否正確安裝
3. 驗證主機 NVIDIA 驅動是否正常工作
4. 在容器內運行 `python scripts/check_gpu_docker.py` 檢查 GPU 狀態

**問題：GPU 記憶體不足**

解決方案：

1. 降低 `VLLM_GPU_MEMORY_UTILIZATION` (預設 0.9)
2. 選擇較小的模型 (如 Qwen2-0.5B 而非 7B)
3. 調整 `EMBEDDING_BATCH_SIZE` 和 `FILE_BATCH_SIZE`
4. 使用量化模型 (4-bit 量化)

### 日誌查看

- **應用日誌**：`logs/app.log`
- **索引日誌**：`logs/indexing.log`
- **錯誤日誌**：`logs/error.log`

---

**企業級 Q槽文件智能助手確保您的數據安全，同時提供強大的 AI 問答能力。**
