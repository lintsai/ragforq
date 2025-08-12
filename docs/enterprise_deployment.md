# 企業級部署指南

## 📋 概述

Q槽文件智能助手是專為企業環境設計的完全本地化 AI 解決方案，確保數據安全和隱私保護。

## 🔒 企業級特性

### 數據安全保障
- ✅ **完全本地部署**: 所有數據處理在企業內部進行
- ✅ **零數據外流**: 不會向任何外部服務發送數據
- ✅ **離線運行**: 模型下載後可完全離線使用
- ✅ **自主可控**: 企業完全掌控系統和數據

### 支援的 AI 平台

#### 1. Ollama 平台（推薦企業使用）
- **完全本地**: 模型和推理完全在本地進行
- **易於管理**: 簡單的模型管理和部署
- **資源效率**: 針對企業硬體優化
- **穩定可靠**: 成熟的本地推理解決方案

#### 2. Hugging Face 平台（企業本地化）
- **本地模型庫**: 從 Hugging Face 下載模型到本地
- **離線推理**: 下載後完全本地運行，無需網路
- **豐富選擇**: 大量開源模型可選
- **高性能**: 支援 vLLM 高性能推理引擎

**重要說明**: 兩個平台都是完全本地化的，符合企業數據安全要求。

## 🏢 部署架構

### 推薦部署架構

```
┌─────────────────────────────────────────────────────────────┐
│                    企業內網環境                              │
│                                                             │
│  ┌─────────────────┐    ┌─────────────────┐                │
│  │   用戶終端      │    │   管理員終端    │                │
│  │  (瀏覽器訪問)   │    │  (系統管理)     │                │
│  └─────────────────┘    └─────────────────┘                │
│           │                       │                        │
│           └───────────┬───────────┘                        │
│                       │                                    │
│  ┌─────────────────────────────────────────────────────────┐│
│  │              應用伺服器 (.121)                          ││
│  │                                                         ││
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    ││
│  │  │   前端服務  │  │   API服務   │  │  AI推理引擎 │    ││
│  │  │ (Port 8501) │  │ (Port 8000) │  │ (本地模型)  │    ││
│  │  └─────────────┘  └─────────────┘  └─────────────┘    ││
│  │                                                         ││
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    ││
│  │  │  向量資料庫 │  │  文件索引   │  │   日誌系統  │    ││
│  │  │   (FAISS)   │  │  (本地儲存) │  │ (本地日誌)  │    ││
│  │  └─────────────┘  └─────────────┘  └─────────────┘    ││
│  └─────────────────────────────────────────────────────────┘│
│                       │                                    │
│  ┌─────────────────────────────────────────────────────────┐│
│  │              共享文件系統                               ││
│  │         /mnt/winshare/MIC (Q槽映射)                    ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

### 網路安全
- **內網部署**: 系統部署在企業內網，外部無法訪問
- **防火牆保護**: 可配置防火牆規則限制訪問
- **VPN 訪問**: 支援通過企業 VPN 遠程訪問
- **權限控制**: 管理員 Token 保護敏感功能

## 🚀 部署步驟

### 1. 環境準備

#### 硬體需求
- **CPU**: 16核心以上（推薦）
- **記憶體**: 64GB RAM（推薦）
- **GPU**: NVIDIA RTX 3090/4090 或 A100（可選但推薦）
- **儲存**: 500GB+ SSD
- **網路**: 千兆內網

#### 軟體需求
- **作業系統**: Ubuntu 20.04+ 或 CentOS 8+
- **Docker**: 最新版本
- **NVIDIA Driver**: 如果使用 GPU

### 2. 系統部署

#### 使用 Docker 部署（推薦）

```bash
# 1. 克隆專案
git clone <repository-url>
cd ragforq

# 2. 配置環境變數
cp .env.production .env
# 編輯 .env 文件，設置正確的路徑和配置

# 3. 構建 Docker 映像
docker build -t ragforq-enterprise .

# 4. 運行容器
docker run -d \
  --name ragforq-production \
  --restart always \
  --gpus all \
  -p 8000:8000 \
  -p 8501:8501 \
  -v /path/to/documents:/mnt/winshare/MIC \
  -v /path/to/db:/app/db \
  -v /path/to/cache:/app/models/cache \
  ragforq-enterprise
```

#### 手動部署

```bash
# 1. 安裝 Python 環境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 2. 安裝依賴
pip install -r requirements.txt

# 3. 配置環境
cp .env.production .env
# 編輯配置文件

# 4. 啟動服務
python scripts/quick_start.py
```

### 3. 初始設置

1. **訪問系統**: http://your-server:8501
2. **選擇平台**: 推薦選擇 Ollama（企業環境）
3. **下載模型**: 系統會引導下載所需模型
4. **配置完成**: 跟隨設置流程完成配置

## 🔧 企業級配置

### 安全配置

#### 1. 管理員 Token 設置
```bash
# 在 .env 文件中設置強密碼
ADMIN_TOKEN="your-secure-admin-token-here"
```

#### 2. 網路訪問控制
```bash
# 限制訪問 IP（可選）
APP_HOST="192.168.1.100"  # 指定服務器 IP
```

#### 3. 日誌配置
```bash
# 設置日誌級別
LOG_LEVEL="INFO"  # DEBUG, INFO, WARNING, ERROR
```

### 性能優化

#### 1. GPU 配置
```bash
# GPU 記憶體使用率
VLLM_GPU_MEMORY_UTILIZATION="0.9"

# 最大序列長度
VLLM_MAX_MODEL_LEN="4096"
```

#### 2. 並行處理
```bash
# 工作線程數
MAX_WORKERS="8"

# 批處理大小
EMBEDDING_BATCH_SIZE="32"
FILE_BATCH_SIZE="50"
```

### 儲存配置

#### 1. 文檔路徑
```bash
# Q槽映射路徑
Q_DRIVE_PATH="/mnt/winshare/MIC"
DISPLAY_DRIVE_NAME="Q:/MIC共用文件庫"
```

#### 2. 資料庫路徑
```bash
# 向量資料庫路徑
VECTOR_DB_PATH="/app/db"

# 模型緩存路徑
HF_MODEL_CACHE_DIR="/app/models/cache"
```

## 📊 監控和維護

### 系統監控

#### 1. 健康檢查
```bash
# 檢查系統狀態
python tests/test_complete_system.py --check

# 檢查服務狀態
curl http://localhost:8000/
curl http://localhost:8501/
```

#### 2. 日誌監控
```bash
# 查看應用日誌
tail -f logs/app.log

# 查看索引日誌
tail -f logs/indexing.log

# 查看錯誤日誌
tail -f logs/error.log
```

### 定期維護

#### 1. 模型更新
- 定期檢查模型版本
- 測試新模型性能
- 逐步升級生產模型

#### 2. 索引維護
- 定期重建索引
- 清理過期數據
- 監控索引性能

#### 3. 系統備份
```bash
# 備份向量資料庫
tar -czf vector_db_backup.tar.gz /app/db

# 備份配置文件
tar -czf config_backup.tar.gz config/ .env

# 備份日誌
tar -czf logs_backup.tar.gz logs/
```

## 🔍 故障排除

### 常見問題

#### 1. 服務無法啟動
- 檢查端口是否被佔用
- 確認配置文件正確
- 查看日誌文件

#### 2. 模型載入失敗
- 檢查磁盤空間
- 確認網路連接（首次下載）
- 檢查 GPU 記憶體

#### 3. 文檔索引問題
- 確認文檔路徑正確
- 檢查文件權限
- 查看索引日誌

### 技術支援

#### 診斷信息收集
```bash
# 系統信息
python scripts/check_system_coupling.py

# 環境信息
python scripts/check_hf_environment.py

# 日誌收集
tar -czf diagnostic_logs.tar.gz logs/ config/
```

## 📞 聯繫支援

如需技術支援，請提供：
1. 系統環境信息
2. 錯誤日誌文件
3. 配置文件（去除敏感信息）
4. 問題重現步驟

---

**企業級 Q槽文件智能助手確保您的數據安全，同時提供強大的 AI 問答能力。**