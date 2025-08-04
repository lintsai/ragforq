# 多語言RAG引擎架構

## 概述

這是一個重構後的多語言RAG（Retrieval-Augmented Generation）引擎架構，支持四種語言的獨立處理：繁體中文、简体中文、English和泰文。

## 架構特點

### 1. Interface設計模式
- `RAGEngineInterface`: 抽象基類，定義所有RAG引擎必須實現的方法
- 每種語言都有獨立的實現類，確保語言特定的優化

### 2. 支持的語言
- **繁體中文** (`TraditionalChineseRAGEngine`)
- **简体中文** (`SimplifiedChineseRAGEngine`) 
- **English** (`EnglishRAGEngine`)
- **泰文** (`ThaiRAGEngine`)

### 3. 工廠模式管理
- `RAGEngineFactory`: 負責創建和管理不同語言的RAG引擎
- 支持引擎緩存，提高性能
- 自動語言標準化和驗證

## 主要改進

### 原有架構問題
- 單一RAG引擎處理所有語言
- 查詢流程：中文 → 英文 → 目標語言
- 語言轉換可能導致信息丟失

### 新架構優勢
- 每種語言獨立處理：查詢（任何語言）→ 同語言優化檢索 → 目標語言回答
- 語言特定的優化策略和回答生成
- 更好的維護性和擴展性
- 保持語言一致性，避免翻譯造成的信息丟失
- 支持最多5筆關聯檔案顯示
- 當找不到相關文檔時，提供基於常識的回答

## 使用方法

### 基本使用

```python
from rag_engine.rag_engine_factory import get_rag_engine_for_language
from indexer.document_indexer import DocumentIndexer

# 創建文檔索引器
document_indexer = DocumentIndexer()

# 獲取繁體中文RAG引擎
engine = get_rag_engine_for_language(
    language="繁體中文",
    document_indexer=document_indexer,
    ollama_model="llama3.2:3b"
)

# 使用引擎回答問題
answer = engine.answer_question("什麼是ITPortal？")
print(answer)
```

### 支持的語言檢查

```python
from rag_engine.rag_engine_factory import get_supported_languages, validate_language

# 獲取支持的語言列表
languages = get_supported_languages()
print(f"支持的語言: {languages}")

# 驗證語言支持
is_supported = validate_language("繁體中文")
print(f"繁體中文支持: {is_supported}")
```

### 緩存管理

```python
from rag_engine.rag_engine_factory import clear_rag_engine_cache

# 清理所有緩存
clear_rag_engine_cache()

# 清理特定語言的緩存
clear_rag_engine_cache(language="繁體中文")

# 清理特定語言和模型的緩存
clear_rag_engine_cache(language="繁體中文", ollama_model="llama3.2:3b")
```

## API集成

### 更新的API端點

新架構已集成到API中，支持語言參數：

```python
# POST /ask
{
    "question": "什麼是ITPortal？",
    "language": "繁體中文",  # 新增語言參數
    "include_sources": true,
    "use_query_rewrite": true,
    "selected_model": "model_name"
}
```

### 新增端點

```python
# GET /api/supported-languages
# 返回支持的語言列表
{
    "supported_languages": ["繁體中文", "简体中文", "English", "泰文", "ไทย"]
}
```

## 語言映射

系統支持多種語言標識符的映射：

```python
LANGUAGE_MAPPING = {
    "繁體中文": "traditional_chinese",
    "中文": "traditional_chinese",  # 默認為繁體中文
    "简体中文": "simplified_chinese",
    "簡體中文": "simplified_chinese",
    "English": "english",
    "english": "english",
    "泰文": "thai",
    "ไทย": "thai",
    "thai": "thai"
}
```

## 實現細節

### 每個語言引擎實現的方法

1. **`get_language()`**: 返回引擎支持的語言
2. **`rewrite_query()`**: 語言特定的查詢優化（保持原語言）
3. **`answer_question()`**: 使用目標語言回答問題
4. **`generate_relevance_reason()`**: 生成相關性理由
5. **`_generate_general_knowledge_answer()`**: 基於常識的回答（當找不到文檔時）
6. **錯誤處理方法**: 語言特定的錯誤訊息

### 通用方法（繼承自基類）

1. **`retrieve_documents()`**: 文檔檢索
2. **`format_context()`**: 上下文格式化
3. **`format_sources()`**: 來源格式化
4. **`get_answer_with_sources()`**: 帶來源的問答
5. **`get_answer_with_query_rewrite()`**: 帶查詢重寫的問答

## 測試

運行測試腳本來驗證新架構：

```bash
python test_new_rag_architecture.py
```

測試內容包括：
- 語言支持驗證
- 引擎創建測試
- 緩存功能測試
- 錯誤處理測試

## 擴展新語言

要添加新語言支持：

1. 創建新的語言引擎類，繼承`RAGEngineInterface`
2. 實現所有抽象方法
3. 在`RAGEngineFactory`中添加語言映射和引擎類映射
4. 更新測試和文檔

## 特殊功能

### 常識回答功能
當系統在文檔中找不到相關資料時，會自動提供基於一般IT常識的回答：

- **智能回退**: 自動檢測無相關文檔的情況
- **常識回答**: 基於一般IT知識提供有用的回答
- **免責聲明**: 明確標示回答來源，避免誤解
- **多語言支持**: 每種語言都有對應的常識回答實現

### 查詢優化策略
- **語言一致性**: 查詢優化保持原語言，避免翻譯丟失信息
- **完整描述生成**: 將問題轉換為完整的描述性語句，而非簡單關鍵詞
- **語意擴展**: 智能擴展相關專業術語和同義詞
- **表達方式多樣化**: 包含文檔中可能出現的不同表達方式
- **上下文保持**: 保留問題的核心概念和語意完整性

### 相關度排序機制
- **智能檢索**: 檢索更多文檔段落以提供充足選擇
- **文件去重**: 每個文件只保留最相關的段落
- **相關度排序**: 按文件的最高相關度分數排序
- **精準篩選**: 確保返回相關度最高的5個不同文件

### 回答生成機制
- **統一文檔來源**: 回答生成和來源顯示使用相同的檢索文檔
- **充分利用內容**: 使用最多8個相關文檔段落生成回答
- **智能長度控制**: 限制總上下文長度避免超出模型限制
- **來源標註**: 在上下文中標註每個段落的文件來源

## 注意事項

1. **模型兼容性**: 確保使用的Ollama模型支持目標語言
2. **性能考慮**: 引擎會被緩存，但首次創建可能需要時間
3. **錯誤處理**: 不支持的語言會自動回退到繁體中文
4. **泰文特殊處理**: 泰文引擎包含額外的字符驗證邏輯
5. **常識回答**: 當找不到文檔時會提供常識回答，但會明確標示來源

## 遷移指南

從舊架構遷移到新架構：

1. 更新API調用，添加`language`參數
2. 使用`get_rag_engine_for_language()`替代直接創建`RAGEngine`
3. 更新錯誤處理邏輯以適應新的語言特定錯誤訊息
4. 測試所有支持的語言功能