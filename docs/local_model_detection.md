# Hugging Face 本地模型檢測

## 概述

現在系統已經支援像 Ollama 一樣的本地模型檢測功能。系統會自動掃描本地已下載的 Hugging Face 模型，而不需要在線下載或硬編碼模型列表。

## 功能特點

- ✅ **純本地模式**: 只顯示 `models/cache` 目錄中實際存在的模型
- ✅ **自動檢測**: 掃描 Hugging Face 緩存目錄中的已下載模型
- ✅ **智能分類**: 自動識別語言模型和嵌入模型
- ✅ **模型信息**: 顯示模型大小、路徑等詳細信息
- ✅ **即用即顯**: 下載完成的模型立即出現在選擇列表中

## 快速開始

### 完整流程示例

```bash
# 1. 安裝 Hugging Face CLI 工具
pip install huggingface-hub[cli]

# 2. 下載推薦的模型組合（標準配置）
hf download Qwen/Qwen2-0.5B-Instruct --cache-dir ./models/cache
hf download sentence-transformers/paraphrase-multilingual-mpnet-base-v2 --cache-dir ./models/cache

# 3. 檢查下載的模型
python scripts/list_models.py

# 4. 啟動應用，選擇 Hugging Face 平台
python scripts/quick_start.py
```

## 使用方式

### 1. 檢查本地可用模型

```bash
# 列出所有本地可用的模型
python scripts/list_models.py
```

### 2. 下載模型

#### 方法一：使用 Hugging Face CLI（推薦）

```bash
# 安裝工具
pip install huggingface-hub[cli]

# 本地環境模型下載
hf download Qwen/Qwen2-0.5B-Instruct --cache-dir ./models/cache
hf download openai/gpt-oss-20b --cache-dir ./models/cache
hf download sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 --cache-dir ./models/cache

# 生產環境模型下載
hf download openai/gpt-oss-20b --cache-dir ~/rag_data/models/cache
hf download sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 --cache-dir ~/rag_data/models/cache
hf download sentence-transformers/paraphrase-multilingual-mpnet-base-v2 --cache-dir ~/rag_data/models/cache
```

**CLI 工具優勢**:
- ✅ **官方支援**: Hugging Face 官方維護
- ✅ **斷點續傳**: 支援下載中斷後繼續
- ✅ **並行下載**: 自動並行下載多個文件
- ✅ **進度顯示**: 實時顯示下載進度
- ✅ **版本控制**: 支援指定特定版本或分支

#### 方法二：使用項目內建腳本

```bash
# 使用 Hugging Face CLI 下載模型（推薦）
hf download Qwen/Qwen2.5-0.5B-Instruct --cache-dir ./models/cache
hf download sentence-transformers/paraphrase-multilingual-mpnet-base-v2 --cache-dir ./models/cache
```

### 3. 在應用中使用

當你啟動應用並選擇 Hugging Face 平台時，系統會：

1. 自動掃描本地已下載的模型
2. 在模型選擇界面**只顯示**本地實際存在的模型
3. 如果沒有本地模型，會顯示空列表並提示先下載模型

## 模型檢測邏輯

### 支援的緩存結構

系統支援兩種 Hugging Face 緩存結構：

1. **標準 Hugging Face 緩存結構**:
   ```
   models/cache/
   ├── models--Qwen--Qwen2-0.5B-Instruct/
   │   └── snapshots/
   │       └── [hash]/
   │           ├── config.json
   │           ├── pytorch_model.bin
   │           └── tokenizer.json
   └── models--sentence-transformers--paraphrase-multilingual-mpnet-base-v2/
   ```

2. **直接存放結構**:
   ```
   models/cache/
   ├── Qwen2-0.5B-Instruct/
   │   ├── config.json
   │   └── pytorch_model.bin
   └── paraphrase-multilingual-mpnet-base-v2/
   ```

### 模型類型檢測

系統使用以下方法檢測模型類型：

1. **名稱匹配**: 檢查模型名稱中的關鍵詞
   - 嵌入模型: `sentence-transformers`, `embedding`, `embed`, `mpnet`, `minilm`
   - 語言模型: `gpt`, `llama`, `qwen`, `bert`

2. **配置文件分析**: 讀取 `config.json` 中的 `architectures` 字段

3. **默認分類**: 無法確定時默認為語言模型

## 配置

### 緩存目錄設置

在 `.env` 文件中設置模型緩存目錄：

```bash
HF_MODEL_CACHE_DIR="./models/cache"
```

### 推薦模型配置

當沒有本地模型時，系統會顯示以下推薦模型：

**語言模型**:
- `Qwen/Qwen2-0.5B-Instruct` - 輕量級，適合開發測試
- `openai/gpt-oss-20b` - 高性能，適合生產環境

**嵌入模型**:
- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` - 輕量級
- `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` - 高精度

## 故障排除

### 模型檢測不到

1. **檢查緩存目錄**:
   ```bash
   python scripts/test_local_models.py
   ```

2. **檢查目錄結構**:
   確保模型文件包含必要的配置文件（`config.json`, `tokenizer.json` 等）

3. **手動指定路徑**:
   如果模型存放在非標準位置，可以修改 `HF_MODEL_CACHE_DIR` 環境變量

### 模型類型錯誤

如果模型被錯誤分類，可以：

1. 檢查模型名稱是否包含正確的關鍵詞
2. 檢查 `config.json` 文件是否存在且格式正確
3. 在 `utils/huggingface_utils.py` 中調整檢測邏輯

## 開發者信息

### 相關文件

- `utils/huggingface_utils.py` - 本地模型檢測核心邏輯
- `utils/platform_manager.py` - 平台管理器，整合本地模型檢測
- `scripts/list_models.py` - 列出本地模型腳本

### API 接口

```python
from utils.huggingface_utils import huggingface_utils

# 獲取所有本地模型
models = huggingface_utils.get_local_models()

# 獲取語言模型
language_models = huggingface_utils.get_language_models()

# 獲取嵌入模型
embedding_models = huggingface_utils.get_embedding_models()

# 檢查特定模型是否可用
is_available = huggingface_utils.is_model_available("Qwen/Qwen2-0.5B-Instruct")
```