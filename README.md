# Q槽文件智能助手 (Q-Drive RAG Assistant)

這是一個基於RAG（檢索增強生成）技術的智能問答系統，專門用於檢索和查詢公司遠端Q槽上的文件。系統可以自動爬取Q槽文件，建立索引，並支持用戶通過自然語言提問來獲取文檔中的信息。

## 📋 目錄

- [功能特點](#-功能特點)
- [系統架構](#️-系統架構)
- [安裝部署](#-安裝部署)
- [本地Docker環境測試指南](#-本地docker環境測試指南)
- [快速開始](#-快速開始)
- [索引管理](#-索引管理)
- [故障排除與恢復](#️-故障排除與恢復)
- [API文檔](#-api文檔)
- [維護與監控](#-維護與監控)
- [項目結構](#-項目結構)
- [技術支持](#-技術支持)

## ✨ 功能特點

- 🔍 **智能檢索**：基於語義搜索的文檔檢索
- 📄 **多格式支持**：PDF、Word、Excel、PowerPoint、文本文件等
- 🤖 **AI問答**：基於大型語言模型的智能問答
- 🌐 **Web界面**：簡潔直觀的用戶界面
- 📊 **實時監控**：自動監控文件變更並更新索引
- 🔧 **故障恢復**：完善的索引恢復和診斷機制
- ⚡ **高性能**：並行處理和批量索引優化
- 🎛️ **模型管理**：支持多模型管理，可動態選擇不同的語言模型和嵌入模型組合
- 📚 **向量數據庫管理**：每個模型組合獨立的向量數據庫，支持並行訓練和使用

## 🏗️ 系統架構

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   前端界面      │    │   後端API       │    │   索引系統      │
│  (Streamlit)    │◄──►│  (FastAPI)      │◄──►│ (DocumentIndexer)│
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                ▲                       ▲
                                │                       │
                       ┌─────────────────┐    ┌─────────────────┐
                       │   RAG引擎       │    │   文件爬蟲      │
                       │ (Query Engine)  │    │ (FileCrawler)   │
                       └─────────────────┘    └─────────────────┘
                                ▲                       ▲
                                │                       │
                       ┌─────────────────┐    ┌─────────────────┐
                       │   向量數據庫    │    │     Q槽        │
                       │    (FAISS)      │    │   文件系統      │
                       └─────────────────┘    └─────────────────┘
```

## ⚙️ 環境配置 (Environment Configuration)

本專案針對不同的運行環境（本地開發、生產部署）使用不同的配置。請根據您的使用場景，將 `.env.example` 複製為對應的 `.env` 檔案，並根據以下指引進行設定。

### 1. 本地開發 (Local Development)

此配置適用於在個人電腦（尤其是沒有 GPU、記憶體有限的機器）上進行開發和功能測試。目標是低資源消耗和快速響應。

- **對應檔案:** `.env` (本地直接啟動) 和 `.env.local` (本地 Docker 測試)
- **核心配置:**
    - **LLM 模型:** `phi3:mini` (輕量級、高效的語言模型)
    - **Embedding 模型:** `nomic-embed-text` (輕量級、高效能的向量模型)
- **首次設定指令:**
    在開始開發前，請務必在您的 Ollama 中下載所需模型：
    ```bash
    ollama pull phi3:mini
    ollama pull nomic-embed-text
    ```

### 2. 生產環境 (Production)

此配置專為部署在擁有強大硬體（如多張 NVIDIA 4090 顯示卡）的伺服器上而設計。目標是最大化問答的準確度和效能。

- **對應檔案:** `.env.production`
- **核心配置:**
    - **LLM 模型:** `qwen2:72b` (頂級效能的 720 億參數語言模型)
    - **Embedding 模型:** `mxbai-embed-large` (頂級效能的向量模型)
- **首次設定指令:**
    在部署伺服器上，請確保已下載所需模型：
    ```bash
    ollama pull qwen2:72b
    ollama pull mxbai-embed-large
    ```

## 🚀 安裝部署

### 系統需求

- **Python**: 3.10+
- **Docker**: 最新版本
- **內存**: 建議 16GB 以上
- **磁盤**: 根據Q槽文件大小預留空間
- **網絡**: Q槽訪問權限及Ollama服務訪問權限

### 核心架構變更說明 (2025-07-21)

為了修復長時間運行的索引任務（如初始訓練、重建索引）會被中斷的問題，系統架構進行了以下關鍵修改：

1. **`api/main.py`**: 將觸發索引任務的API端點從直接使用 `subprocess.Popen` 啟動腳本，修改為使用 `subprocess.run(["supervisorctl", "start", "..."])`。
2. **`supervisord.conf`**:
   * 為 `initial_indexing`, `monitor_changes`, `reindex` 等耗時任務創建了獨立的 `[program]` 配置。
   * 添加了 `[unix_http_server]` 和 `[supervisorctl]` 配置塊，以允許 `supervisorctl` 客戶端與 `supervisord` 服務端正常通信。

這項修改將索引任務的生命週期與API服務的生命週期完全解耦。現在，索引任務由Supervisor直接管理，即使API服務因任何原因重啟，索引任務也能在背景穩定運行，不會再被中斷。

### 生產環境部署

生產環境的部署應使用Docker進行，以確保環境的一致性和管理的便捷性。詳細的本地測試流程請參考下一章節，生產部署與其類似，但需注意 `.env.production` 的配置。

---

## 🐳 本地Docker環境測試指南

本章節詳細記錄了如何在本地Windows環境下，使用Docker來模擬、測試和驗證生產環境的部署。

### 第 1 步：準備本地環境設定檔

1. **複製設定檔**：如果還沒有 `.env.local` 文件，請從 `.env.example` 複製一份。
2. **修改 `.env.local`**：這是最關鍵的步驟，確保容器可以正確連接到外部資源。
   * **Ollama Host**: 將 `localhost` 修改為 `host.docker.internal`，這是Docker容器訪問宿主機服務的特殊DNS名稱。
     ```env
     # 修改前
     # OLLAMA_HOST=http://localhost:11434

     # 修改後
     OLLAMA_HOST=http://host.docker.internal:11434
     ```
   * **數據源路徑**: 將Q槽的Windows路徑，修改為我們即將在容器中創建的掛載點路徑。
     ```env
     # 假設Q槽路徑為 D:\data\MIC共用文件庫\05_MIC專案
     # 修改前
     # TARGET_DIRECTORY=D:\data\MIC共用文件庫\05_MIC專案

     # 修改後
     TARGET_DIRECTORY=/q_drive_data/MIC共用文件庫/05_MIC專案
     ```

### 第 2 步：構建與啟動Docker容器

在專案根目錄下打開終端機 (PowerShell/CMD)，執行以下指令。

1. **構建Docker映像檔**:

   ```bash
   docker build -t ragforq-local-test .
   ```
2. **啟動Docker容器**:
   這是一個完整的啟動指令，包含了所有必要的Volume掛載，請根據您的實際本機路徑進行修改。

   ```bash
   # -v {本機路徑}:{容器內路徑}
   docker run --rm -d -p 8501:8501 -p 8000:8000 --name ragforq-test-container -v D:\source\ragforq\.env.local:/app/.env -v D:\data:/q_drive_data/MIC共用文件庫/05_MIC專案 -v D:\source\ragforq\vector_db:/app/vector_db ragforq-local-test
   ```

   * **-v D:\source\ragforq\.env.local:/app/.env**: 掛載本地設定檔，覆蓋映像檔中的版本。
   * **-v D:\data:/q_drive_data/MIC共用文件庫/05_MIC專案**: 掛載Q槽的根目錄，讓容器可以讀取源文件。
   * **-v D:\source\ragforq\vector_db:/app/vector_db**: 掛載向量數據庫目錄，實現數據持久化，避免容器停止後索引丟失。

### 第 3 步：在容器內進行驗證與除錯

1. **進入正在運行的容器**:

   ```bash
   docker exec -it ragforq-test-container /bin/bash
   ```
2. **查看由Supervisor管理的所有服務狀態**:

   ```bash
   supervisorctl status
   ```

   您應該能看到 `fastapi_backend` 和 `streamlit_frontend` 正在 `RUNNING`。
3. **實時查看某個服務的日誌**:

   ```bash
   # 查看fastapi的日誌
   supervisorctl tail -f fastapi_backend

   # 查看某個索引任務的日誌
   supervisorctl tail -f reindex
   ```
4. **手動啟動/停止/重啟服務**:
   這是驗證索引任務獨立性的核心。

   ```bash
   # 模擬API服務重啟
   supervisorctl restart fastapi_backend

   # 手動啟動一個任務
   supervisorctl start initial_indexing

   # 手動停止一個任務
   supervisorctl stop initial_indexing
   ```

### 第 4 步：測試後清理環境

1. **停止並自動移除容器**:

   ```bash
   docker stop ragforq-test-container
   ```
2. **移除本地測試用的映像檔 (可選)**:
   如果想釋放硬碟空間，可以刪除之前構建的映像檔。

   ```bash
   docker rmi ragforq-local-test
   ```

---

## 🚀 快速開始

### 第一次使用

1. **初始化索引**:

   ```bash
   python scripts/initial_indexing.py
   ```

   此過程會爬取Q槽上的所有支持格式的文件並建立索引。根據文件數量，可能需要數小時完成。
2. **監控索引進度**（另開終端）:

   ```bash
   python scripts/monitor_indexing.py --monitor
   ```

### 啟動服務

1. **啟動後端API**:

   ```bash
   python api/main.py
   # 或使用 uvicorn
   uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
   ```
2. **啟動前端界面**:

   ```bash
   streamlit run frontend/streamlit_app.py --server.port 8501
   ```
3. **訪問系統**:

   - 前端界面：http://localhost:8501
   - API文檔：http://localhost:8000/docs

### 使用示例

在前端界面中輸入問題:

- "公司的年假政策是什麼？"
- "如何申請報銷差旅費？"
- "產品退貨流程是怎樣的？"
- "最新的安全規範文件在哪裡？"

系統會搜索相關文檔並提供詳細回答，同時顯示參考來源。

### 支持的文件格式

| 格式       | 擴展名    | 說明              |
| ---------- | --------- | ----------------- |
| PDF        | .pdf      | 支持文本提取和OCR |
| Word       | .docx     | 支持文本和表格    |
| Excel      | .xlsx     | 支持工作表和數據  |
| PowerPoint | .pptx     | 支持幻燈片內容    |
| 文本       | .txt, .md | 純文本文件        |
| CSV        | .csv      | 結構化數據        |

## 🎛️ 向量模型管理系統

### 系統概述

本系統實現了完整的向量模型管理功能，支援動態選擇 Ollama 模型進行訓練和問答。每個模型組合（語言模型 + 嵌入模型）都有獨立的向量數據庫，支持並行訓練和使用。

### 主要功能

#### 1. 管理介面功能

- **動態模型選擇**：OLLAMA_MODEL 和 OLLAMA_EMBEDDING_MODEL 下拉選單，自動從 Ollama API 獲取可用模型
- **向量資料結構**：新的資料夾命名格式 `ollama@{OLLAMA_MODEL}@{OLLAMA_EMBEDDING_MODEL}`
  - 例如：`ollama@phi3_mini@nomic-embed-text_latest`
  - 每個模型組合都有獨立的向量資料夾，支援未來擴充不同種訓練方式
- **版本管理**：支援同一模型組合的多個版本
  - 版本格式：`ollama@{model}@{embedding}#{version}`
  - 例如：`ollama@phi3_mini@nomic-embed-text_latest#20250722`
  - 系統自動選擇最新可用版本進行問答
- **模型狀態檢查**：自動檢查向量資料是否已存在，顯示訓練狀態（訓練中/可用）
- **訓練鎖定機制**：使用 `.lock` 檔案防止同時進行多個訓練任務，訓練完成後自動移除鎖定檔案
- **模型信息記錄**：每個模型資料夾包含 `.model` 檔案記錄模型信息，包含版本和創建時間等元數據

#### 2. 問答介面功能

- **動態模型選擇**：問答介面新增模型下拉選單，支援「使用默認配置」選項
- **模型可用性檢查**：自動過濾掉訓練中的模型（有 `.lock` 檔案），只顯示有數據且可用的模型

#### 3. API 端點

```
GET /api/ollama/models          # 獲取 Ollama 可用模型列表
GET /api/vector-models          # 獲取向量數據庫模型列表
GET /api/usable-models          # 獲取可用於問答的模型列表
POST /admin/training/initial    # 開始初始訓練
POST /admin/training/incremental # 開始增量訓練
POST /admin/training/reindex    # 開始重新索引
POST /ask                       # 支援 selected_model 參數
```

### 向量數據結構

```
vector_db/
├── ollama@phi3_mini@nomic-embed-text_latest/
│   ├── .model                 # 模型信息檔案
│   ├── .lock                  # 訓練鎖定檔案（訓練時存在）
│   ├── index.faiss           # FAISS 索引檔案
│   ├── index.pkl             # 索引元數據
│   ├── indexed_files.pkl     # 已索引文件記錄
│   └── indexing_progress.json # 索引進度
├── ollama@phi3_mini@nomic-embed-text_latest#20250722/
│   ├── .model                 # 帶版本信息的模型檔案
│   ├── index.faiss           # 版本化的向量索引
│   ├── index.pkl             # 版本化的索引元數據
│   ├── indexed_files.pkl     # 版本化的已索引文件記錄
│   └── indexing_progress.json # 版本化的索引進度
└── ollama@qwen2_72b@mxbai-embed-large/
    ├── .model
    ├── index.faiss
    ├── index.pkl
    ├── indexed_files.pkl
    └── indexing_progress.json
```

#### .model 檔案格式

```json
{
  "OLLAMA_MODEL": "phi3:mini",
  "OLLAMA_EMBEDDING_MODEL": "nomic-embed-text:latest",
  "version": "20250722",
  "created_at": "2025-01-22T10:30:00"
}
```

**版本管理說明**：
- 基礎格式：`ollama@{model}@{embedding}`
- 版本格式：`ollama@{model}@{embedding}#{version}`
- 版本標識通常使用日期格式（如 `20250722`）
- 同一模型組合可以有多個版本，系統會自動選擇最新可用版本

### 使用流程

#### 1. 模型訓練流程

1. **進入管理員後台**：輸入管理員 Token，進入「模型訓練管理」區域
2. **選擇模型**：從 OLLAMA_MODEL 和 OLLAMA_EMBEDDING_MODEL 下拉選單選擇模型
3. **檢查狀態**：系統自動顯示當前模型組合的狀態
   - 新模型組合：顯示「將創建新的向量資料夾」
   - 已有數據：顯示「可進行增量訓練或重新索引」
   - 訓練中：顯示「正在訓練中」並禁用操作按鈕
4. **開始訓練**：點擊相應的訓練按鈕（初始訓練/增量訓練/重新索引）

#### 2. 問答使用流程

1. **選擇模型**：在問答介面選擇要使用的模型（選單會自動過濾掉訓練中的模型）
2. **進行問答**：輸入問題並提交，系統使用選定模型的向量數據和對應的語言模型進行問答

### 命令行工具

#### 使用模型訓練管理器

```bash
# 初始訓練
python scripts/model_training_manager.py initial \
  --ollama-model phi3:mini \
  --ollama-embedding-model nomic-embed-text

# 增量訓練
python scripts/model_training_manager.py incremental \
  --ollama-model phi3:mini \
  --ollama-embedding-model nomic-embed-text

# 重新索引
python scripts/model_training_manager.py reindex \
  --ollama-model phi3:mini \
  --ollama-embedding-model nomic-embed-text
```

#### 添加模型信息到現有資料夾

```bash
python scripts/add_model_files.py
```

### 技術實現

#### 核心組件

- **VectorDBManager**：管理向量數據庫文件夾、處理模型信息和狀態、提供鎖定機制
- **ModelTrainingManager**：管理模型訓練流程，支援初始訓練、增量訓練、重新索引，集成向量數據庫管理器
- **OllamaUtils**：與 Ollama API 交互，獲取可用模型列表，檢查模型可用性

### 配置變更

#### 移除的配置項
- `OLLAMA_MODEL`：不再使用固定配置，改為動態選擇
- `OLLAMA_EMBEDDING_MODEL`：不再使用固定配置，改為動態選擇

#### 保留的配置項
- `OLLAMA_HOST`：Ollama 服務地址
- `VECTOR_DB_PATH`：向量數據庫基礎路徑
- 其他文件處理和系統配置

### 注意事項

1. **模型可用性**：確保選擇的 Ollama 模型已經下載並可用
2. **資源管理**：不同模型組合會佔用不同的磁盤空間
3. **訓練時間**：初始訓練可能需要較長時間，請耐心等待
4. **並發限制**：同一模型組合不能同時進行多個訓練任務
5. **備份建議**：重要的向量數據建議定期備份

### 故障排除

#### 常見問題

1. **模型列表為空**：檢查 Ollama 服務是否正在運行，確認 OLLAMA_HOST 配置正確
2. **訓練失敗**：檢查磁盤空間是否充足，確認 Q 槽是否可訪問，查看日誌文件獲取詳細錯誤信息
3. **問答無響應**：確認選擇的模型有向量數據，檢查模型是否正在訓練中，驗證 Ollama 模型是否可用

#### 日誌查看
- 索引日誌：`logs/indexing.log`
- 應用日誌：`logs/app.log`
- 管理員後台提供日誌下載功能

### 未來擴展

系統設計支援未來的擴展需求：

1. **多種訓練方式**：資料夾結構支援不同的訓練方式前綴
2. **模型版本管理**：可以擴展支援模型版本控制
3. **分散式訓練**：可以擴展支援多機器訓練
4. **自動化調度**：可以添加定時訓練和自動更新功能

## 🔧 索引管理

### 初始化索引（舊版）

```bash
# 全新建立索引
python scripts/initial_indexing.py
```

### 監控索引狀態

```bash
# 實時監控
python scripts/monitor_indexing.py --monitor

# 查看當前狀態
python scripts/monitor_indexing.py --status

# 查看詳細進度
python scripts/monitor_indexing.py --progress
```

### 恢復中斷的索引

```bash
# 診斷問題
python scripts/diagnose_indexing.py

# 穩定恢復（推薦）
python scripts/stable_resume_indexing.py

# 標準恢復
python scripts/resume_indexing.py
```

### 文件變更監控

```bash
# 啟動文件監控
python scripts/monitor_changes.py --interval 3600
```

### 重新索引

```bash
# 完全重建索引
python scripts/reindex.py
```

## 🛠️ 故障排除與恢復

### 索引中斷恢復指南

當索引程序在處理過程中突然中斷時，可以使用以下方法進行恢復:

#### 🔍 第一步：診斷問題

運行診斷腳本檢查系統狀態:

```bash
python scripts/diagnose_indexing.py
```

診斷內容包括:

- 向量數據庫文件狀態
- 索引進度記錄
- 系統資源使用情況
- Q槽網絡連接
- Ollama服務狀態
- 日誌文件分析

#### 📊 第二步：監控當前狀態

```bash
# 實時監控（每5秒刷新）
python scripts/monitor_indexing.py --monitor

# 查看當前狀態
python scripts/monitor_indexing.py --status

# 查看詳細進度
python scripts/monitor_indexing.py --progress
```

#### 🔄 第三步：選擇恢復方法

根據診斷結果，選擇合適的恢復方法:

**方法1：穩定恢復（推薦）**

```bash
python scripts/stable_resume_indexing.py
```

特點:

- 小批次處理（每批5個文件）
- 動態資源監控
- 自動錯誤恢復
- 安全中斷處理
- 詳細進度保存

**方法2：標準恢復**

```bash
python scripts/resume_indexing.py
```

**方法3：原始腳本繼續**

```bash
python scripts/initial_indexing.py
```

原始腳本會自動檢測已處理的文件並跳過。

### 常見問題

#### 1. 內存不足

**症狀**：索引過程中斷，系統響應緩慢

**解決方案**:

```bash
# 檢查內存使用
free -h

# 清理內存
sudo sync && sudo sysctl vm.drop_caches=3

# 調整配置（編輯 config/config.py）
FILE_BATCH_SIZE = 5      # 減小批次大小
EMBEDDING_BATCH_SIZE = 8 # 減小嵌入批次
MAX_WORKERS = 4          # 減少並行線程
```

#### 2. 網絡連接問題

**症狀**：無法訪問Q槽文件

**解決方案**:

```bash
# 檢查Q槽掛載
ls -la /mnt/winshare/MIC/

# 重新掛載
sudo umount /mnt/winshare/MIC/
sudo mount -t cifs //server/path /mnt/winshare/MIC/ -o credentials=/path/to/creds
```

#### 3. Ollama服務問題

**症狀**：嵌入生成失敗

**解決方案**:

```bash
# 檢查服務狀態
curl http://localhost:11434/api/tags

# 重啟服務
sudo systemctl restart ollama

# 檢查模型
ollama list
ollama pull llama3.2  # 如果模型不存在
```

#### 4. 索引進程卡死

**緊急處理**:

```bash
# 查找並終止進程
ps aux | grep python | grep indexing
kill -9 <PID>

# 清理臨時文件
rm -f /tmp/tmp*

# 重置進度（謹慎使用）
python scripts/monitor_indexing.py --reset
```

### 性能優化

#### 系統級優化

```bash
# 增加文件描述符限制
ulimit -n 65536

# 調整虛擬內存設置
echo 'vm.swappiness=10' | sudo tee -a /etc/sysctl.conf
sudo sysctl -p
```

#### 監控資源使用

```bash
# 實時監控系統資源
htop

# 監控磁盤I/O
iotop

# 監控網絡連接
netstat -an | grep 11434
```

### 日誌分析

```bash
# 查看最新日誌
tail -f logs/indexing.log

# 搜索錯誤
grep -i error logs/indexing.log

# 搜索特定文件處理記錄
grep "索引文件" logs/indexing.log | tail -20
```

## 🔌 API文檔

### 主要端點

| 方法 | 端點        | 描述     |
| ---- | ----------- | -------- |
| GET  | `/health` | 健康檢查 |
| GET  | `/`       | 系統狀態 |
| POST | `/ask`    | 問答接口 |
| GET  | `/stats`  | 索引統計 |

### 問答接口示例

```bash
curl -X POST "http://localhost:8000/ask" \
     -H "Content-Type: application/json" \
     -d '{"question": "公司的年假政策是什麼？"}'
```

響應格式:

```json
{
  "answer": "根據公司政策...",
  "sources": [
    {
      "file_path": "/path/to/document.pdf",
      "page": 1,
      "relevance_score": 0.95
    }
  ],
  "processing_time": 2.3
}
```

## 📁 項目結構

```
q-drive-rag-assistant/
├── api/                    # FastAPI後端
│   ├── main.py            # 主API服務
│   └── main_pro.py        # 專業版API
├── config/                # 配置文件
│   └── config.py          # 主配置
├── frontend/              # 前端界面
│   └── streamlit_app.py   # Streamlit應用
├── indexer/               # 索引系統
│   ├── document_indexer.py # 文檔索引器
│   └── file_crawler.py    # 文件爬蟲
├── logs/                  # 日誌文件
│   └── indexing.log       # 索引日誌
├── rag_engine/            # RAG引擎
│   └── rag_engine.py      # 查詢引擎
├── scripts/               # 實用腳本
│   ├── initial_indexing.py      # 初始化索引
│   ├── stable_resume_indexing.py # 穩定恢復
│   ├── monitor_indexing.py      # 監控工具
│   ├── diagnose_indexing.py     # 診斷工具
│   └── monitor_changes.py       # 文件監控
├── utils/                 # 工具函數
│   └── file_parsers.py    # 文件解析器
├── vector_db/             # 向量數據庫
│   ├── index.faiss        # FAISS索引
│   ├── index.pkl          # 索引元數據
│   ├── indexed_files.pkl  # 已索引文件記錄
│   └── indexing_progress.json # 進度記錄
├── .env                   # 環境變量
├── requirements.txt       # 依賴列表
└── README.md             # 本文檔
```

## 🔄 維護與監控

### 定期維護

1. **每日檢查**:

   ```bash
   python scripts/monitor_indexing.py --status
   ```
2. **每週清理**:

   ```bash
   # 清理日誌文件
   find logs/ -name "*.log" -mtime +7 -delete

   # 檢查磁盤空間
   df -h
   ```
3. **每月優化**:

   ```bash
   # 重建索引（如果需要）
   python scripts/reindex.py

   # 系統資源檢查
   python scripts/diagnose_indexing.py
   ```

### 備份策略

重要文件備份:

- `vector_db/` 目錄（向量數據庫）
- `.env` 文件（配置）
- `logs/indexing.log`（日誌記錄）

### 監控告警

建議設置以下監控:

- 磁盤空間使用率 > 90%
- 內存使用率 > 85%
- 索引進程異常退出
- Q槽連接中斷

## 📚 腳本使用快速參考

| 腳本       | 用途             | 命令                                              |
| ---------- | ---------------- | ------------------------------------------------- |
| 初始化索引 | 全新建立索引     | `python scripts/initial_indexing.py`            |
| 穩定恢復   | 從中斷點安全恢復 | `python scripts/stable_resume_indexing.py`      |
| 標準恢復   | 一般恢復索引     | `python scripts/resume_indexing.py`             |
| 診斷工具   | 檢查系統狀態     | `python scripts/diagnose_indexing.py`           |
| 實時監控   | 監控索引進度     | `python scripts/monitor_indexing.py --monitor`  |
| 查看狀態   | 顯示當前狀態     | `python scripts/monitor_indexing.py --status`   |
| 詳細進度   | 查看詳細進度     | `python scripts/monitor_indexing.py --progress` |
| 重置進度   | 重置索引進度     | `python scripts/monitor_indexing.py --reset`    |
| 文件監控   | 監控文件變更     | `python scripts/monitor_changes.py`             |
| 重新索引   | 完全重建索引     | `python scripts/reindex.py`                     |

## 🆘 緊急故障處理

### 索引程序卡死

```bash
# 查找並終止進程
ps aux | grep python | grep indexing
kill -9 <PID>

# 清理臨時文件
rm -f /tmp/tmp*
```

### 系統資源不足

```bash
# 檢查並清理內存
free -h
sudo sync && sudo sysctl vm.drop_caches=3

# 檢查磁盤空間
df -h
```

### 服務異常

```bash
# 檢查Ollama服務
curl http://localhost:11434/api/tags
sudo systemctl restart ollama

# 檢查Q槽掛載
ls -la /mnt/winshare/MIC/
```

---

## 📞 技術支持

### 故障排除流程

遇到問題時，請按以下順序進行排查:

1. **診斷問題** → `python scripts/diagnose_indexing.py`
2. **檢查系統資源** → 內存、磁盤、網絡
3. **驗證服務狀態** → Ollama、Q槽掛載
4. **選擇恢復方法** → 穩定恢復 > 標準恢復 > 原始腳本
5. **監控恢復進度** → `python scripts/monitor_indexing.py --monitor`

### 日誌分析

```bash
# 查看最新日誌
tail -f logs/indexing.log

# 搜索錯誤
grep -i error logs/indexing.log

# 搜索處理記錄
grep "索引文件" logs/indexing.log | tail -20
```

### 聯繫方式

- 查看項目源碼了解技術細節
- 檢查 `logs/indexing.log` 獲取詳細錯誤信息
- 運行診斷腳本獲取系統狀態報告

---

**最後更新**: 2025-07-21
**版本**: v2.1
**維護者**: MIC IT Team & Gemini
