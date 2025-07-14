# Q槽文件智能助手 (Q-Drive RAG Assistant)

這是一個基於RAG（檢索增強生成）技術的智能問答系統，專門用於檢索和查詢公司遠端Q槽上的文件。

## 功能特點

- 自動爬取並索引Q槽文件
- 支持多種文件格式（PDF、Word、Excel、文本文件等）
- 高效的語義搜索
- 基於大型語言模型的智能問答
- 簡潔直觀的Web界面
- 實時監控Q槽文件變更，自動更新索引

## 系統需求

- Python 3.9+
- 網絡訪問權限（用於連接Q槽）

## 安裝方法

1. 克隆此倉庫：

   ```
   git clone [倉庫地址]
   cd q-drive-rag-assistant
   ```
2. 安裝依賴：

   ```
   pip install -r requirements.txt
   ```
3. 創建並配置環境變量文件：

   - 複製 `.env.example`為 `.env`
   - 填寫必要的配置信息，包括Q槽路徑
4. 運行初始索引：

   ```
   python scripts/initial_indexing.py
   ```
5. 啟動服務：

   ```
   python app.py
   ```

   或使用FastAPI的Uvicorn伺服器：

   ```
   uvicorn api.main:app --reload
   ```
6. 啟動前端界面：

   ```
   streamlit run frontend/streamlit_app.py
   ```

## 使用方法

1. 訪問Web界面（默認地址：http://localhost:8501）
2. 在搜索框中輸入問題
3. 系統將從Q槽文件中檢索相關信息並生成回答

## 維護與更新

- 系統會自動監控Q槽文件變更並更新索引
- 可以通過運行 `python scripts/reindex.py`手動觸發完整重新索引

## 結構說明

- `indexer/`: 文件索引和向量存儲相關代碼
- `rag_engine/`: RAG查詢引擎實現
- `api/`: FastAPI後端API
- `frontend/`: Streamlit前端界面
- `scripts/`: 實用腳本（索引、監控等）
- `config/`: 配置文件
- `utils/`: 通用工具函數
