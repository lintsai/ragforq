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

## 🔧 索引管理

### 初始化索引

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
