# 動態RAG重構架構說明

本文檔說明動態RAG系統的重構架構，將語言特定的功能從基礎類移到各自的引擎中。

## 重構原因

### 原有問題
- ❌ 所有語言的常識回答邏輯都擠在 `DynamicRAGEngineBase` 中
- ❌ 違反了單一職責原則
- ❌ 代碼維護困難，語言特定邏輯混雜
- ❌ 不符合面向對象設計原則

### 重構目標
- ✅ 將語言特定功能移到各自的引擎中
- ✅ 保持基礎類的通用性
- ✅ 提高代碼可維護性
- ✅ 符合開放封閉原則

## 新架構設計

### 基礎類 (`DynamicRAGEngineBase`)

**職責：**
- 提供通用的動態RAG功能
- 定義抽象方法供子類實現
- 處理文件檢索和向量化

**關鍵方法：**
```python
class DynamicRAGEngineBase(RAGEngineInterface):
    def _generate_general_knowledge_answer(self, question: str) -> str:
        """生成常識回答 - 子類實現"""
        # 默認實現，子類應該重寫此方法
        return "默認回答..."
    
    def _get_general_fallback(self, query: str) -> str:
        """獲取通用回退回答 - 子類應該重寫此方法"""
        # 默認實現，子類應該重寫此方法
        return "默認回退回答..."
```

### 語言特定引擎

每個語言引擎都實現自己的常識回答功能：

#### 1. 繁體中文引擎 (`DynamicTraditionalChineseRAGEngine`)

```python
def _generate_general_knowledge_answer(self, question: str) -> str:
    """當找不到相關文檔時，基於常識提供繁體中文回答"""
    general_prompt = """你是一個IT領域的專家助手。雖然在QSI內部文檔中找不到相關資料，但請基於一般IT常識來回答以下問題。

問題：{question}

請注意：
1. 使用繁體中文回答
2. 基於一般IT知識提供有用的回答
3. 明確說明這是基於常識的回答，不是來自QSI內部文檔
4. 如果是QSI特定的問題，建議聯繫相關部門

繁體中文回答："""
    
    # 添加繁體中文免責聲明
    disclaimer = "\n\n※ 注意：以上回答基於一般IT常識，非來自QSI內部文檔。如需準確資訊，請聯繫相關部門。"

def _get_general_fallback(self, query: str) -> str:
    """獲取繁體中文通用回退回答"""
    return f"根據一般IT知識，關於「{query}」的相關信息可能需要查閱更多QSI內部文檔。"
```

#### 2. 簡體中文引擎 (`DynamicSimplifiedChineseRAGEngine`)

```python
def _generate_general_knowledge_answer(self, question: str) -> str:
    """当找不到相关文档时，基于常识提供简体中文回答"""
    general_prompt = """你是一个IT领域的专家助手。虽然在QSI内部文档中找不到相关资料，但请基于一般IT常识来回答以下问题。

问题：{question}

请注意：
1. 使用简体中文回答
2. 基于一般IT知识提供有用的回答
3. 明确说明这是基于常识的回答，不是来自QSI内部文档
4. 如果是QSI特定的问题，建议联系相关部门

简体中文回答："""
    
    # 添加简体中文免责声明
    disclaimer = "\n\n※ 注意：以上回答基于一般IT常识，非来自QSI内部文档。如需准确信息，请联系相关部门。"

def _get_general_fallback(self, query: str) -> str:
    """获取简体中文通用回退回答"""
    return f"根据一般IT知识，关于「{query}」的相关信息可能需要查阅更多QSI内部文档。"
```

#### 3. 英文引擎 (`DynamicEnglishRAGEngine`)

```python
def _generate_general_knowledge_answer(self, question: str) -> str:
    """Provide general knowledge answer in English when no relevant documents found"""
    general_prompt = """You are an IT expert assistant. Although no relevant information was found in QSI internal documents, please provide an answer based on general IT knowledge.

Question: {question}

Please note:
1. Answer in English only
2. Provide useful answers based on general IT knowledge
3. Clearly state this is based on general knowledge, not from QSI internal documents
4. If it's QSI-specific, suggest contacting relevant departments

English answer:"""
    
    # Add English disclaimer
    disclaimer = "\n\n※ Note: The above answer is based on general IT knowledge, not from QSI internal documents. For accurate information, please contact relevant departments."

def _get_general_fallback(self, query: str) -> str:
    """Get English general fallback answer"""
    return f"Based on general IT knowledge, information about '{query}' may require consulting additional QSI internal documentation."
```

#### 4. 泰文引擎 (`DynamicThaiRAGEngine`)

```python
def _generate_general_knowledge_answer(self, question: str) -> str:
    """ให้คำตอบจากความรู้ทั่วไปเป็นภาษาไทยเมื่อไม่พบเอกสารที่เกี่ยวข้อง"""
    general_prompt = """คุณเป็นผู้ช่วยผู้เชี่ยวชาญด้าน IT แม้ว่าจะไม่พบข้อมูลที่เกี่ยวข้องในเอกสารภายใน QSI แต่กรุณาให้คำตอบโดยอิงจากความรู้ทั่วไปด้าน IT

คำถาม: {question}

กรุณาทราบ:
1. ตอบเป็นภาษาไทยเท่านั้น
2. ให้คำตอบที่เป็นประโยชน์โดยอิงจากความรู้ทั่วไปด้าน IT
3. ระบุอย่างชัดเจนว่านี่เป็นคำตอบจากความรู้ทั่วไป ไม่ใช่จากเอกสารภายใน QSI
4. หากเป็นคำถามเฉพาะของ QSI แนะนำให้ติดต่อแผนกที่เกี่ยวข้อง

คำตอบภาษาไทย:"""
    
    # เพิ่มข้อความปฏิเสธความรับผิดชอบภาษาไทย
    disclaimer = "\n\n※ หมายเหตุ: คำตอบข้างต้นอิงจากความรู้ทั่วไปด้าน IT ไม่ใช่จากเอกสารภายใน QSI หากต้องการข้อมูลที่แม่นยำ กรุณาติดต่อแผนกที่เกี่ยวข้อง"

def _get_general_fallback(self, query: str) -> str:
    """รับคำตอบทางเลือกทั่วไปภาษาไทย"""
    return f"ตามความรู้ทั่วไปด้าน IT ข้อมูลเกี่ยวกับ '{query}' อาจต้องการการปรึกษาเอกสารภายใน QSI เพิ่มเติม"
```

## 架構優勢

### 1. 單一職責原則
- 每個引擎只負責自己語言的實現
- 基礎類只提供通用功能
- 職責分離清晰

### 2. 開放封閉原則
- 對擴展開放：可以輕鬆添加新語言
- 對修改封閉：不需要修改基礎類

### 3. 可維護性
- 語言特定代碼集中在各自引擎中
- 修改某種語言不影響其他語言
- 代碼結構清晰易懂

### 4. 可測試性
- 每個引擎可以獨立測試
- 測試覆蓋更全面
- 問題定位更容易

## 文件結構

```
rag_engine/
├── dynamic_rag_base.py                    # 基礎類（通用功能）
├── dynamic_traditional_chinese_engine.py  # 繁體中文引擎
├── dynamic_simplified_chinese_engine.py   # 簡體中文引擎
├── dynamic_english_engine.py              # 英文引擎
└── dynamic_thai_engine.py                 # 泰文引擎
```

## 使用方式

### 創建引擎
```python
from rag_engine.rag_engine_factory import get_rag_engine_for_language

# 創建繁體中文動態RAG引擎
engine = get_rag_engine_for_language(
    language="Dynamic_繁體中文",
    document_indexer=None,
    ollama_model="llama3.2:3b",
    ollama_embedding_model="nomic-embed-text"
)
```

### 使用常識回答
```python
# 當找不到相關文檔時，會自動調用對應語言的常識回答
answer = engine.answer_question("什麼是人工智能？")

# 也可以直接調用
knowledge_answer = engine._generate_general_knowledge_answer("什麼是雲端運算？")
```

## 測試驗證

### 運行測試
```bash
python scripts/test_dynamic_rag_knowledge_answer.py
```

### 測試內容
- 每種語言的常識回答功能
- 免責聲明檢查
- 語言一致性驗證
- 回答質量評估

## 遷移指南

### 從舊版本遷移
1. **無需修改使用代碼**：API保持不變
2. **功能增強**：每種語言都有專門優化的實現
3. **性能提升**：減少了不必要的語言判斷邏輯

### 添加新語言
1. 創建新的引擎類繼承 `DynamicRAGEngineBase`
2. 實現 `_generate_general_knowledge_answer` 方法
3. 實現 `_get_general_fallback` 方法
4. 在工廠類中註冊新引擎

## 最佳實踐

### 1. 提示模板設計
- 明確指定目標語言
- 包含免責聲明要求
- 提供具體的回答指導

### 2. 錯誤處理
- 每個方法都有適當的異常處理
- 提供有意義的錯誤日誌
- 確保總是有回退方案

### 3. 日誌記錄
- 使用語言特定的日誌訊息
- 記錄關鍵操作和錯誤
- 便於問題診斷和性能監控

## 總結

重構後的動態RAG架構具有以下優勢：

- ✅ **清晰的職責分離**：每個引擎負責自己的語言
- ✅ **更好的可維護性**：語言特定代碼集中管理
- ✅ **符合設計原則**：遵循SOLID原則
- ✅ **易於擴展**：添加新語言更簡單
- ✅ **提高代碼質量**：結構清晰，邏輯分明

這個重構確保了動態RAG系統的長期可維護性和擴展性，同時保持了與現有代碼的兼容性。