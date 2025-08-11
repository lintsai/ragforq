# 項目依賴說明

## 依賴檢查狀態

✅ **所有依賴都已正確配置在 pyproject.toml 中**

## 核心依賴

### Web框架
- `streamlit ^1.47.0` - 前端界面框架
- `fastapi ^0.111.0` - 後端API框架
- `uvicorn ^0.29.0` - ASGI服務器

### 機器學習和RAG
- `sentence-transformers ^2.7.0` - 文本嵌入模型
- `langchain ^0.3.26` - LangChain核心庫
- `langchain-ollama ^0.3.5` - Ollama集成
- `langchain-community ^0.3.27` - LangChain社區組件
- `faiss-cpu ^1.8.0` - 向量數據庫

### 文檔處理
- `unstructured ^0.14.4` - 通用文檔解析
- `pypdf ^4.2.0` - PDF處理
- `pymupdf ^1.26.3` - PDF處理（備用）
- `python-docx ^1.2.0` - Word文檔處理
- `openpyxl ^3.1.5` - Excel處理
- `xlrd ^2.0.2` - Excel讀取
- `olefile ^0.47` - OLE文件處理

### 數據處理
- `numpy ^1.24.0` - 數值計算
- `requests ^2.31.0` - HTTP請求
- `pydantic ^2.0.0` - 數據驗證
- `annotated-types ^0.7.0` - 類型註解

### 工具庫
- `python-dotenv ^1.0.1` - 環境變量管理
- `tqdm ^4.66.0` - 進度條
- `psutil ^7.0.0` - 系統監控
- `pytz ^2025.2` - 時區處理
- `streamlit-autorefresh ^1.0.1` - Streamlit自動刷新

## 安裝方法

### 使用Poetry（推薦）
```bash
# 安裝所有依賴
poetry install

# 檢查依賴狀態
python scripts/check_dependencies.py
```

### 使用pip
```bash
# 從requirements.txt安裝（如果有的話）
pip install -r requirements.txt

# 或者手動安裝核心依賴
pip install streamlit fastapi uvicorn sentence-transformers faiss-cpu langchain langchain-ollama
```

## 版本要求

- **Python**: 3.10.x（嚴格要求）
- **操作系統**: Windows/Linux/macOS
- **記憶體**: 建議8GB以上
- **硬碟**: 建議10GB以上可用空間

## 依賴檢查

運行以下命令檢查所有依賴是否正確安裝：

```bash
python scripts/check_dependencies.py
```

預期輸出應該顯示所有依賴都成功安裝（100%成功率）。

## 常見問題

### 1. Pydantic版本衝突
如果遇到Pydantic版本問題，項目已配置自動處理v1/v2兼容性。

### 2. FAISS安裝問題
- Windows: 使用 `faiss-cpu`
- 如需GPU支持: 使用 `faiss-gpu`

### 3. 文檔處理問題
- 確保安裝了所有文檔處理庫
- Windows用戶可能需要額外的Visual C++運行庫

### 4. Ollama連接問題
- 確保Ollama服務運行在 `http://localhost:11434`
- 檢查防火牆設置

## 更新依賴

```bash
# 更新所有依賴到最新兼容版本
poetry update

# 更新特定依賴
poetry update streamlit

# 檢查過時的依賴
poetry show --outdated
```