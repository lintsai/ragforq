# Dynamic RAG 使用指南

## 🎉 修復完成

Dynamic RAG 的前端界面問題已經修復！現在你可以完整使用 Dynamic RAG 功能了。

## 🚀 快速開始

### 1. 啟動服務

```bash
# 終端1: 啟動API服務
python app.py

# 終端2: 啟動前端服務
streamlit run frontend/streamlit_app.py
```

### 2. 使用Dynamic RAG

1. **打開前端界面**: 訪問 `http://localhost:8501`

2. **選擇Dynamic RAG模式**:
   - 在左側邊欄找到「設置」區域
   - 選擇「RAG模式」為「Dynamic RAG」
   - 系統會顯示提示：💡 Dynamic RAG 無需預先建立向量資料庫，查詢時即時檢索文件

3. **配置模型**:
   - **語言模型**: 選擇 `qwen2.5:0.5b-instruct` 或 `qwen2:0.5b-instruct`
   - **嵌入模型**: 選擇 `nomic-embed-text:latest`

4. **開始問答**:
   - 在聊天框中輸入問題
   - 系統會自動進行動態檢索和回答

## 📋 可用模型

根據測試結果，你的系統中有以下可用模型：

### 語言模型 (用於生成回答)
- `qwen2:0.5b-instruct` - 輕量級中文語言模型
- `qwen2.5:0.5b-instruct` - 改進版中文語言模型 (推薦)

### 嵌入模型 (用於文本向量化)
- `nomic-embed-text:latest` - 高質量嵌入模型

## 🔧 API使用

如果你想通過API直接調用Dynamic RAG：

```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "什麼是技術？",
    "use_dynamic_rag": true,
    "selected_model": "qwen2.5:0.5b-instruct",
    "ollama_embedding_model": "nomic-embed-text:latest",
    "language": "繁體中文",
    "include_sources": true
  }'
```

## 💻 程式化使用

```python
from rag_engine.dynamic_rag_engine import DynamicRAGEngine

# 創建Dynamic RAG引擎
engine = DynamicRAGEngine(
    ollama_model="qwen2.5:0.5b-instruct",
    ollama_embedding_model="nomic-embed-text:latest",
    language="繁體中文"
)

# 回答問題
answer = engine.answer_question("什麼是人工智能？")
print(answer)

# 獲取帶來源的回答
answer, sources, documents = engine.get_answer_with_sources("技術的好處是什麼？")
print(f"回答: {answer}")
print(f"來源: {sources}")
```

## 🎯 Dynamic RAG vs 傳統RAG

| 特性 | 傳統RAG | Dynamic RAG |
|------|---------|-------------|
| **初始化時間** | 長（需建立索引） | 短（立即可用） ✅ |
| **存儲需求** | 高（向量資料庫） | 低（僅文件） ✅ |
| **實時性** | 低（需重建索引） | 高（立即生效） ✅ |
| **維護成本** | 高（索引管理） | 低（無需維護） ✅ |
| **查詢延遲** | 低（預建索引） | 中等（即時處理） |
| **適用規模** | 大型 | 小到中型 |

## 🔍 工作原理

Dynamic RAG 的工作流程：

1. **智能文件檢索**: 根據查詢關鍵詞智能選擇相關文件
2. **動態內容處理**: 即時解析和分割文件內容
3. **實時向量化**: 將查詢和文檔內容轉換為向量
4. **相似度計算**: 計算查詢與文檔的相似度
5. **回答生成**: 基於最相關的內容生成回答

## 📊 性能特點

### 優勢
- ✅ **無需預建索引**: 節省大量時間和存儲空間
- ✅ **實時文件更新**: 文件變更立即生效
- ✅ **智能檢索**: 自動選擇最相關的文件
- ✅ **資源效率**: 按需處理，節省資源

### 適用場景
- 📁 文件經常變更的環境
- 💾 存儲空間有限的系統
- ⚡ 需要快速部署的場景
- 📚 小到中型文件庫（< 1000個文件）

## 🛠️ 故障排除

### 常見問題

1. **"獲取Ollama模型時出錯"**
   - ✅ **已修復**: 更新了API端點和前端邏輯
   - 確保Ollama服務正在運行: `curl http://localhost:11434/api/tags`

2. **找不到相關文件**
   - 檢查Q槽路徑配置
   - 確認文件類型在支援列表中
   - 嘗試使用更具體的關鍵詞

3. **回答質量不佳**
   - 嘗試使用 `qwen2.5:0.5b-instruct` 模型
   - 調整查詢關鍵詞
   - 檢查文件內容是否相關

### 測試工具

```bash
# 測試基本功能
python scripts/test_dynamic_rag_simple.py

# 測試前端修復
python scripts/test_frontend_fix.py

# 測試最小化功能
python scripts/test_dynamic_rag_minimal.py
```

## 📈 使用建議

### 1. 模型選擇
- **推薦語言模型**: `qwen2.5:0.5b-instruct` (更好的中文支持)
- **推薦嵌入模型**: `nomic-embed-text:latest` (高質量向量化)

### 2. 查詢優化
- 使用具體、明確的問題
- 包含相關的專業術語
- 避免過於寬泛的查詢

### 3. 文件組織
- 使用有意義的文件名
- 合理組織目錄結構
- 定期清理無用文件

## 🎊 總結

Dynamic RAG 現在已經完全可用！主要特點：

- ✅ **前端界面修復**: 解決了模型獲取錯誤
- ✅ **API端點完善**: 新增分類模型端點
- ✅ **完整測試驗證**: 所有功能測試通過
- ✅ **即時可用**: 無需預建向量資料庫

你現在可以：
1. 立即使用Dynamic RAG進行問答
2. 享受實時文件更新的便利
3. 節省大量存儲空間和維護成本
4. 在傳統RAG和Dynamic RAG之間靈活選擇

🚀 **開始使用Dynamic RAG，體驗無需預建索引的智能問答吧！**