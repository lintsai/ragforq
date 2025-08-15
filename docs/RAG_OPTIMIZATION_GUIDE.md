# RAG系統優化指南

本指南提供了解決RAG系統常見問題的完整方案。

## 問題與解決方案概覽

### 1. 傳統RAG重新索引問題

**問題症狀：**
- 索引過程中斷或失敗
- 大文件處理超時
- 編碼錯誤導致索引失敗

**解決方案：**
- ✅ 實現斷點續傳機制
- ✅ 添加文件預處理和編碼檢測
- ✅ 優化批處理大小和超時設置
- ✅ 增強錯誤恢復能力

### 2. 傳統RAG超時問題

**問題症狀：**
- Ollama模型響應超時
- 查詢優化過程中斷
- 回答生成失敗

**解決方案：**
- ✅ 增加超時時間配置
- ✅ 實現重試機制
- ✅ 優化批處理策略
- ✅ 添加並發控制

### 3. 傳統RAG相關性問題

**問題症狀：**
- 檢索結果不相關
- 重複文檔內容
- 相似度評分不準確

**解決方案：**
- ✅ 實現動態相似度閾值
- ✅ 添加文檔去重機制
- ✅ 優化文檔排序算法
- ✅ 改進上下文格式化

### 4. 動態RAG文件掃描問題

**問題症狀：**
- Q槽文件掃描不完整
- 重要文件未被發現
- 掃描深度不足

**解決方案：**
- ✅ 增加掃描深度和文件數量限制
- ✅ 實現優先目錄掃描
- ✅ 優化文件緩存機制
- ✅ 添加智能文件過濾

### 5. 動態RAG優化效果問題

**問題症狀：**
- 查詢重寫過於簡化
- 上下文信息不足
- 回答質量不如傳統RAG

**解決方案：**
- ✅ 增強查詢重寫策略
- ✅ 改進文檔選擇算法
- ✅ 豐富上下文格式化
- ✅ 參考傳統RAG最佳實踐

### 6. 文件編碼問題

**問題症狀：**
- 文件內容顯示亂碼
- 編碼檢測失敗
- 非UTF-8文件處理錯誤

**解決方案：**
- ✅ 實現多編碼自動檢測
- ✅ 集成chardet編碼檢測庫
- ✅ 添加亂碼清理功能
- ✅ 支持常見中文編碼

## 使用方法

### 1. 應用優化配置

```bash
# 自動應用所有優化配置
python -c "from config.rag_optimization import apply_rag_optimizations; apply_rag_optimizations()"
```

### 2. 運行系統診斷

```bash
# 基本診斷
python scripts/rag_diagnostic_tool.py

# 診斷並自動修復
python scripts/rag_diagnostic_tool.py --auto-fix

# 保存診斷結果
python scripts/rag_diagnostic_tool.py --output diagnostic_report.json
```

### 3. 手動配置優化

編輯 `.env.production` 文件，確保包含以下優化配置：

```bash
# 超時優化
OLLAMA_REQUEST_TIMEOUT=300
OLLAMA_EMBEDDING_TIMEOUT=180
OLLAMA_MAX_RETRIES=3

# 批處理優化
FILE_BATCH_SIZE=10
EMBEDDING_BATCH_SIZE=32
MAX_FILE_SIZE_MB=50

# 編碼處理
AUTO_ENCODING_DETECTION=true
USE_CHARDET=true

# 動態RAG優化
DYNAMIC_MAX_SCAN_FILES=5000
DYNAMIC_SCAN_DEPTH=8
```

### 4. 重建索引（如需要）

```bash
# 清理舊索引
rm -rf vector_db/*

# 重新索引
python scripts/rebuild_index.py
```

## 性能監控

### 1. 檢查系統狀態

```python
from scripts.rag_diagnostic_tool import RAGDiagnosticTool

diagnostic = RAGDiagnosticTool()
results = diagnostic.run_full_diagnostic()
print(f"系統狀態: {results['system_status']}")
```

### 2. 監控關鍵指標

- **Q槽訪問狀態**：確保網絡連接正常
- **向量數據庫完整性**：檢查索引文件存在
- **文件編碼問題**：監控亂碼文件數量
- **內存使用率**：避免超過80%
- **磁盤空間**：確保有足夠存儲空間

### 3. 日誌監控

```bash
# 查看索引日誌
tail -f logs/indexing.log

# 查看應用日誌
tail -f logs/app.log

# 查看錯誤日誌
tail -f logs/error.log
```

## 故障排除

### 常見問題解決

1. **索引中斷**
   ```bash
   # 檢查進度文件
   cat vector_db/indexing_progress.json
   
   # 繼續索引
   python scripts/continue_indexing.py
   ```

2. **編碼問題**
   ```bash
   # 安裝編碼檢測庫
   pip install chardet
   
   # 檢查問題文件
   python scripts/check_encoding.py
   ```

3. **超時問題**
   ```bash
   # 增加超時時間
   export OLLAMA_REQUEST_TIMEOUT=600
   
   # 減少批處理大小
   export FILE_BATCH_SIZE=5
   ```

4. **動態RAG掃描問題**
   ```bash
   # 清理文件緩存
   python -c "from rag_engine.dynamic_rag_base import SmartFileRetriever; SmartFileRetriever().file_cache.clear()"
   
   # 重新掃描
   python scripts/rescan_files.py
   ```

## 最佳實踐

### 1. 定期維護

- 每週運行一次系統診斷
- 定期清理無效索引記錄
- 監控磁盤空間使用情況
- 更新過期文件索引

### 2. 性能優化

- 根據系統資源調整批處理大小
- 使用SSD存儲向量數據庫
- 定期重啟Ollama服務
- 監控內存使用情況

### 3. 數據質量

- 定期檢查文件編碼問題
- 清理重複和無效文件
- 驗證索引完整性
- 測試查詢質量

### 4. 備份策略

- 定期備份向量數據庫
- 保存索引記錄文件
- 備份配置文件
- 記錄重要變更

## 配置參考

### 推薦配置（生產環境）

```bash
# 超時設置
OLLAMA_REQUEST_TIMEOUT=300
OLLAMA_EMBEDDING_TIMEOUT=180
OLLAMA_QUERY_OPTIMIZATION_TIMEOUT=90
OLLAMA_ANSWER_GENERATION_TIMEOUT=300
OLLAMA_RELEVANCE_TIMEOUT=120
OLLAMA_CONNECTION_TIMEOUT=60

# 重試設置
OLLAMA_MAX_RETRIES=3
OLLAMA_RETRY_DELAY=5

# 批處理設置
FILE_BATCH_SIZE=10
EMBEDDING_BATCH_SIZE=32
MAX_FILE_SIZE_MB=50

# 動態RAG設置
DYNAMIC_MAX_SCAN_FILES=5000
DYNAMIC_SCAN_DEPTH=8
DYNAMIC_CACHE_DURATION=300

# 編碼設置
AUTO_ENCODING_DETECTION=true
USE_CHARDET=true
CLEAN_GARBLED_TEXT=true
```

### 測試環境配置

```bash
# 較小的批處理大小用於測試
FILE_BATCH_SIZE=5
EMBEDDING_BATCH_SIZE=16
MAX_FILE_SIZE_MB=20

# 較短的超時時間用於快速測試
OLLAMA_REQUEST_TIMEOUT=120
OLLAMA_EMBEDDING_TIMEOUT=60

# 較少的文件掃描用於測試
DYNAMIC_MAX_SCAN_FILES=1000
DYNAMIC_SCAN_DEPTH=5
```

## 支持與幫助

如果遇到問題，請：

1. 首先運行診斷工具
2. 檢查相關日誌文件
3. 參考本指南的故障排除部分
4. 聯繫技術支持團隊

---

**注意：** 應用這些優化後，建議重啟RAG服務以確保所有配置生效。