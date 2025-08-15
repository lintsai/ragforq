# 動態RAG常識回答功能

本文檔說明動態RAG系統的常識回答功能，確保與傳統RAG具有相同的智能回答能力。

## 功能概述

動態RAG現在具備與傳統RAG相同的常識回答功能：

- ✅ **智能常識回答**：當找不到相關文檔時，基於一般IT知識提供有用回答
- ✅ **多語言支持**：支援繁體中文、簡體中文、英文、泰文的常識回答
- ✅ **免責聲明**：明確標示回答來源，避免誤解
- ✅ **回退機制**：提供通用回退回答作為最後保障

## 實現細節

### 1. 常識回答方法

```python
def _generate_general_knowledge_answer(self, question: str) -> str:
    """生成常識回答 - 當找不到相關文檔時，基於常識提供回答"""
```

**功能特點：**
- 根據語言自動選擇對應的提示模板
- 基於一般IT知識提供有用回答
- 自動添加免責聲明
- 支援語言檢測和回退

### 2. 回退機制

```python
def _get_general_fallback(self, query: str) -> str:
    """獲取通用回退回答"""
```

**使用場景：**
- 當LLM生成的回答過短時
- 當常識回答生成失敗時
- 作為最後的回答保障

### 3. 語言特定實現

#### 繁體中文
```python
general_prompt = """你是一個IT領域的專家助手。雖然在QSI內部文檔中找不到相關資料，但請基於一般IT常識來回答以下問題。

問題：{question}

請注意：
1. 使用繁體中文回答
2. 基於一般IT知識提供有用的回答
3. 明確說明這是基於常識的回答，不是來自QSI內部文檔
4. 如果是QSI特定的問題，建議聯繫相關部門

繁體中文回答："""
```

#### 簡體中文
```python
general_prompt = """你是一个IT领域的专家助手。虽然在QSI内部文档中找不到相关资料，但请基于一般IT常识来回答以下问题。

问题：{question}

请注意：
1. 使用简体中文回答
2. 基于一般IT知识提供有用的回答
3. 明确说明这是基于常识的回答，不是来自QSI内部文档
4. 如果是QSI特定的问题，建议联系相关部门

简体中文回答："""
```

#### 英文
```python
general_prompt = """You are an IT expert assistant. Although no relevant information was found in QSI internal documents, please provide an answer based on general IT knowledge.

Question: {question}

Please note:
1. Answer in English only
2. Provide useful answers based on general IT knowledge
3. Clearly state this is based on general knowledge, not from QSI internal documents
4. If it's QSI-specific, suggest contacting relevant departments

English answer:"""
```

#### 泰文
```python
general_prompt = """คุณเป็นผู้ช่วยผู้เชี่ยวชาญด้าน IT แม้ว่าจะไม่พบข้อมูลที่เกี่ยวข้องในเอกสารภายใน QSI แต่กรุณาให้คำตอบโดยอิงจากความรู้ทั่วไปด้าน IT

คำถาม: {question}

กรุณาทราบ:
1. ตอบเป็นภาษาไทยเท่านั้น
2. ให้คำตอบที่เป็นประโยชน์โดยอิงจากความรู้ทั่วไปด้าน IT
3. ระบุอย่างชัดเจนว่านี่เป็นคำตอบจากความรู้ทั่วไป ไม่ใช่จากเอกสารภายใน QSI
4. หากเป็นคำถามเฉพาะของ QSI แนะนำให้ติดต่อแผนกที่เกี่ยวข้อง

คำตอบภาษาไทย:"""
```

## 免責聲明

每種語言都會自動添加對應的免責聲明：

### 繁體中文
```
※ 注意：以上回答基於一般IT常識，非來自QSI內部文檔。如需準確資訊，請聯繫相關部門。
```

### 簡體中文
```
※ 注意：以上回答基于一般IT常识，非来自QSI内部文档。如需准确信息，请联系相关部门。
```

### 英文
```
※ Note: The above answer is based on general IT knowledge, not from QSI internal documents. For accurate information, please contact relevant departments.
```

### 泰文
```
※ หมายเหตุ: คำตอบข้างต้นอิงจากความรู้ทั่วไปด้าน IT ไม่ใช่จากเอกสารภายใน QSI หากต้องการข้อมูลที่แม่นยำ กรุณาติดต่อแผนกที่เกี่ยวข้อง
```

## 觸發條件

常識回答功能在以下情況下會被觸發：

1. **文件檢索失敗**：`retrieve_relevant_files()` 返回空列表
2. **文檔處理失敗**：`process_files()` 返回空列表
3. **相似度過低**：所有文檔的相似度都低於閾值
4. **回答生成失敗**：LLM生成的回答過短（< 5字符）

## 使用示例

### 基本使用

```python
from rag_engine.rag_engine_factory import get_rag_engine_for_language

# 創建動態RAG引擎
engine = get_rag_engine_for_language(
    language="Dynamic_繁體中文",
    document_indexer=None,
    ollama_model="llama3.2:3b",
    ollama_embedding_model="nomic-embed-text"
)

# 詢問一個文檔中沒有的問題
question = "什麼是人工智能？"
answer = engine.answer_question(question)

print(answer)
# 輸出將包含基於常識的回答和免責聲明
```

### 直接調用常識回答

```python
# 直接調用常識回答方法
knowledge_answer = engine._generate_general_knowledge_answer("什麼是雲端運算？")
print(knowledge_answer)
```

### 測試回退功能

```python
# 測試回退功能
fallback_answer = engine._get_general_fallback("測試查詢")
print(fallback_answer)
```

## 與傳統RAG的比較

| 特性 | 傳統RAG | 動態RAG | 說明 |
|------|---------|---------|------|
| 常識回答 | ✅ | ✅ | 兩者都支援 |
| 多語言支持 | ✅ | ✅ | 支援相同語言 |
| 免責聲明 | ✅ | ✅ | 自動添加 |
| 回退機制 | ✅ | ✅ | 多層保障 |
| 語言檢測 | ✅ | ✅ | 自動檢測目標語言 |
| 提示模板 | 固定 | 可配置 | 動態RAG更靈活 |

## 測試方法

使用提供的測試腳本驗證功能：

```bash
# 運行常識回答功能測試
python scripts/test_dynamic_rag_knowledge_answer.py
```

測試內容包括：
- 多語言常識回答測試
- 回退功能測試
- 與傳統RAG的比較測試

## 配置選項

可以通過環境變量調整相關設置：

```bash
# 常識回答超時時間
OLLAMA_ANSWER_GENERATION_TIMEOUT=300

# 最大重試次數
OLLAMA_MAX_RETRIES=3

# 重試延遲
OLLAMA_RETRY_DELAY=5
```

## 故障排除

### 常見問題

1. **常識回答生成失敗**
   - 檢查LLM模型是否正常運行
   - 確認網絡連接穩定
   - 檢查超時設置是否合理

2. **語言不一致**
   - 確認 `get_language()` 方法返回正確語言
   - 檢查語言映射配置
   - 驗證提示模板語言設置

3. **回答質量不佳**
   - 調整提示模板內容
   - 嘗試不同的LLM模型
   - 檢查溫度參數設置

### 日誌監控

關鍵日誌訊息：
```
動態RAG常識回答生成失敗: [錯誤訊息]
動態RAG常識回答處理失敗: [錯誤訊息]
生成回答過程中發生錯誤: [錯誤訊息]
```

## 最佳實踐

1. **提示優化**：根據實際使用情況調整提示模板
2. **錯誤處理**：確保所有異常都有適當的回退機制
3. **性能監控**：監控常識回答的生成時間和成功率
4. **用戶反饋**：收集用戶對常識回答質量的反饋

## 總結

動態RAG現在具備與傳統RAG相同的常識回答能力：

- ✅ **功能完整性**：支援所有傳統RAG的常識回答功能
- ✅ **多語言一致性**：四種語言都有對應的實現
- ✅ **錯誤處理**：完善的回退和錯誤恢復機制
- ✅ **用戶體驗**：明確的免責聲明和有用的回答

這確保了無論使用傳統RAG還是動態RAG，用戶都能獲得一致的高質量問答體驗。