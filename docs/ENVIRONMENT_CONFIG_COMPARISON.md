# 環境配置對比表

本文檔說明不同環境下的RAG系統優化配置差異。

## 配置文件說明

| 文件 | 用途 | 環境 |
|------|------|------|
| `.env.production` | 生產環境配置 | 正式部署環境 |
| `.env.local` | 本地Docker環境配置 | Docker容器開發 |
| `.env` | 本地開發環境配置 | 直接本地開發 |
| `.env.example` | 配置模板 | 參考和新部署 |

## 關鍵配置對比

### 1. 批處理設置

| 配置項 | 生產環境 | 開發環境 | Docker環境 | 說明 |
|--------|----------|----------|------------|------|
| `FILE_BATCH_SIZE` | 10 | 5 | 5 | 文件批處理大小，生產環境可以更大 |
| `EMBEDDING_BATCH_SIZE` | 32 | 16 | 16 | 嵌入批處理大小，根據資源調整 |
| `MAX_FILE_SIZE_MB` | 50 | 20 | 20 | 最大文件大小限制 |

### 2. 超時設置

| 配置項 | 生產環境 | 開發環境 | Docker環境 | 說明 |
|--------|----------|----------|------------|------|
| `OLLAMA_REQUEST_TIMEOUT` | 300 | 120 | 120 | 請求超時，生產環境更長 |
| `OLLAMA_EMBEDDING_TIMEOUT` | 180 | 60 | 60 | 嵌入超時 |
| `OLLAMA_ANSWER_GENERATION_TIMEOUT` | 300 | 120 | 120 | 回答生成超時 |
| `OLLAMA_RELEVANCE_TIMEOUT` | 120 | 60 | 60 | 相關性分析超時 |
| `OLLAMA_CONNECTION_TIMEOUT` | 60 | 30 | 30 | 連接超時 |

### 3. 動態RAG設置

| 配置項 | 生產環境 | 開發環境 | Docker環境 | 說明 |
|--------|----------|----------|------------|------|
| `DYNAMIC_MAX_SCAN_FILES` | 5000 | 1000 | 1000 | 最大掃描文件數 |
| `DYNAMIC_SCAN_DEPTH` | 8 | 5 | 5 | 掃描深度 |
| `DYNAMIC_CACHE_DURATION` | 300 | 300 | 300 | 緩存持續時間 |

### 4. 重試機制

| 配置項 | 所有環境 | 說明 |
|--------|----------|------|
| `OLLAMA_MAX_RETRIES` | 3 | 最大重試次數 |
| `OLLAMA_RETRY_DELAY` | 5 (生產) / 3 (開發) | 重試延遲秒數 |

### 5. 編碼處理

| 配置項 | 所有環境 | 說明 |
|--------|----------|------|
| `AUTO_ENCODING_DETECTION` | true | 自動編碼檢測 |
| `USE_CHARDET` | true | 使用chardet庫 |
| `CLEAN_GARBLED_TEXT` | true | 清理亂碼文本 |

## 環境特定配置

### 生產環境 (`.env.production`)

```bash
# 高性能配置
FILE_BATCH_SIZE=10
EMBEDDING_BATCH_SIZE=32
MAX_FILE_SIZE_MB=50

# 較長超時時間
OLLAMA_REQUEST_TIMEOUT=300
OLLAMA_EMBEDDING_TIMEOUT=180
OLLAMA_ANSWER_GENERATION_TIMEOUT=300

# 完整文件掃描
DYNAMIC_MAX_SCAN_FILES=5000
DYNAMIC_SCAN_DEPTH=8

# 網絡路徑
Q_DRIVE_PATH="/mnt/winshare/MIC/"
OLLAMA_HOST="http://localhost:11434"
```

### 開發環境 (`.env`)

```bash
# 較小批處理，快速測試
FILE_BATCH_SIZE=5
EMBEDDING_BATCH_SIZE=16
MAX_FILE_SIZE_MB=20

# 較短超時時間，快速反饋
OLLAMA_REQUEST_TIMEOUT=120
OLLAMA_EMBEDDING_TIMEOUT=60
OLLAMA_ANSWER_GENERATION_TIMEOUT=120

# 有限文件掃描
DYNAMIC_MAX_SCAN_FILES=1000
DYNAMIC_SCAN_DEPTH=5

# 本地路徑
Q_DRIVE_PATH="C:/Users/user/source/ragforq/test_files/"
OLLAMA_HOST="http://localhost:11434"
```

### Docker環境 (`.env.local`)

```bash
# 容器資源限制下的配置
FILE_BATCH_SIZE=5
EMBEDDING_BATCH_SIZE=16
MAX_FILE_SIZE_MB=20

# 適中超時時間
OLLAMA_REQUEST_TIMEOUT=120
OLLAMA_EMBEDDING_TIMEOUT=60
OLLAMA_ANSWER_GENERATION_TIMEOUT=120

# 有限文件掃描
DYNAMIC_MAX_SCAN_FILES=1000
DYNAMIC_SCAN_DEPTH=5

# Docker網絡配置
Q_DRIVE_PATH="/ragforq/test_files/"
OLLAMA_HOST="http://host.docker.internal:11434"
```

## 配置選擇指南

### 根據系統資源選擇

**高配置系統（16GB+ RAM, 8+ CPU核心）：**
- 使用生產環境配置
- 可以增加 `EMBEDDING_BATCH_SIZE` 到 64
- 可以增加 `FILE_BATCH_SIZE` 到 15

**中等配置系統（8-16GB RAM, 4-8 CPU核心）：**
- 使用開發環境配置
- 保持默認批處理大小
- 監控內存使用情況

**低配置系統（<8GB RAM, <4 CPU核心）：**
- 減少批處理大小：`FILE_BATCH_SIZE=3`, `EMBEDDING_BATCH_SIZE=8`
- 減少文件掃描：`DYNAMIC_MAX_SCAN_FILES=500`
- 增加超時時間補償性能不足

### 根據網絡環境選擇

**穩定網絡環境：**
- 使用默認超時設置
- 可以適當減少重試次數

**不穩定網絡環境：**
- 增加所有超時時間 50%
- 增加重試次數到 5
- 增加重試延遲到 10 秒

### 根據文件特性選擇

**大量小文件：**
- 增加 `FILE_BATCH_SIZE`
- 減少 `MAX_FILE_SIZE_MB`
- 增加 `DYNAMIC_MAX_SCAN_FILES`

**少量大文件：**
- 減少 `FILE_BATCH_SIZE`
- 增加 `MAX_FILE_SIZE_MB`
- 增加超時時間

## 配置更新流程

1. **備份現有配置**
   ```bash
   cp .env .env.backup
   ```

2. **應用新配置**
   ```bash
   # 根據環境選擇對應的配置文件
   cp .env.production .env  # 生產環境
   # 或
   cp .env.example .env     # 新部署
   ```

3. **驗證配置**
   ```bash
   python scripts/rag_diagnostic_tool.py
   ```

4. **重啟服務**
   ```bash
   # 重啟RAG服務以應用新配置
   docker-compose restart  # Docker環境
   # 或重啟相關進程
   ```

## 監控和調優

### 性能監控指標

- **內存使用率**：應保持在 80% 以下
- **CPU使用率**：批處理期間可達 100%，平時應在 50% 以下
- **磁盤I/O**：監控向量數據庫讀寫性能
- **網絡延遲**：監控Ollama服務響應時間

### 調優建議

1. **如果經常超時**：
   - 增加相應的超時時間
   - 減少批處理大小
   - 檢查網絡連接

2. **如果內存不足**：
   - 減少 `EMBEDDING_BATCH_SIZE`
   - 減少 `FILE_BATCH_SIZE`
   - 減少 `DYNAMIC_MAX_SCAN_FILES`

3. **如果處理速度慢**：
   - 增加批處理大小（在內存允許範圍內）
   - 使用SSD存儲
   - 升級硬件配置

4. **如果相關性不佳**：
   - 檢查文件編碼設置
   - 調整 `DYNAMIC_SCAN_DEPTH`
   - 重新建立索引

---

**注意**：修改配置後務必重啟相關服務以確保配置生效。建議在測試環境驗證配置效果後再應用到生產環境。