# 測試文件說明

## 動態RAG功能測試

### 功能確認

✅ **1. 動態RAG對所有語言模型可用**
- 前端會調用 `/api/ollama/models/categorized` 獲取所有可用的Ollama模型
- 用戶可以選擇任何可用的語言模型和嵌入模型組合
- 不限制特定模型，所有Ollama中的模型都可以使用

✅ **2. 動態RAG包含語言選擇功能**
- 支援語言：繁體中文、简体中文、English、ไทย、Dynamic
- Dynamic選項會讓AI根據問題內容自動選擇合適的語言回答
- 語言參數正確傳遞到後端API

✅ **3. 後端API正確處理動態RAG請求**
- `QuestionRequest` 包含 `use_dynamic_rag` 和 `ollama_embedding_model` 參數
- `get_rag_engine` 函數正確處理動態RAG初始化
- RAG引擎工廠支援Dynamic語言映射到 `DynamicRAGEngine`

### 測試文件組織

#### 🚀 推薦測試文件（部署前使用）
1. **test_dynamic_rag_simple.py** - 動態RAG簡化測試（API級別）
2. **test_dynamic_rag_comprehensive.py** - 動態RAG全面測試（所有組合）
3. **test_models.py** - 基本模型可用性測試

#### 🔧 開發測試文件
4. **test_dynamic_rag.py** - 動態RAG引擎組件測試
5. **test_dynamic_rag_minimal.py** - 最小化組件測試
6. **test_dynamic_rag_local.py** - 本地環境測試
7. **test_dynamic_rag_full.py** - 完整功能測試

#### 🌐 前端和API測試
8. **test_frontend.py** - 前端功能測試
9. **test_frontend_fix.py** - 前端修復測試
10. **test_api_endpoints.py** - API端點測試

#### 🗄️ 資料庫維護測試
11. **test_content_maintenance.py** - 內容維護測試
12. **test_vector_db_maintenance.py** - 向量資料庫維護測試

### 使用方法

```bash
# 簡化測試（推薦用於部署環境）
python tests/test_dynamic_rag_simple.py

# 全面測試（測試所有模型和語言組合）
python tests/test_dynamic_rag_comprehensive.py

# 基本模型測試
python tests/test_models.py
```

### 部署前檢查清單

- [ ] API服務正常運行 (port 8000)
- [ ] Ollama服務正常運行 (port 11434)
- [ ] 至少有一個語言模型和一個嵌入模型可用
- [ ] 測試文件可以正常執行
- [ ] 前端可以正確顯示模型選項
- [ ] 動態RAG功能可以正常回答問題

### 已知問題

1. **效能問題**：本地環境可能因為硬體限制導致響應緩慢
2. **模型載入**：首次使用某個模型組合時需要較長載入時間
3. **記憶體使用**：多個模型同時載入可能消耗大量記憶體

### 測試文件清理

如果你想清理不必要的測試文件，可以運行：

```bash
python tests/cleanup_tests.py
```

這個腳本會：
- 保留核心測試文件
- 詢問是否刪除可選的開發測試文件
- 顯示清理結果

### 建議

- 在生產環境中使用較小的模型（如0.5b參數的模型）以提高響應速度
- 考慮使用GPU加速來改善效能
- 監控記憶體使用情況，必要時重啟服務
- 部署前運行 `test_dynamic_rag_simple.py` 確保基本功能正常