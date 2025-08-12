"""
Dynamic RAG Engine for Traditional Chinese
"""
from .dynamic_rag_base import DynamicRAGEngineBase

class DynamicTraditionalChineseRAGEngine(DynamicRAGEngineBase):
    """動態繁體中文RAG引擎"""

    REWRITE_PROMPT_TEMPLATE = """請從以下問題中提取2-3個最重要的關鍵詞，用空格分隔。

問題: {original_query}

關鍵詞:"""

    ANSWER_PROMPT_TEMPLATE = """你是一個專業的AI文檔問答助手。請嚴格根據以下「上下文信息」來回答「用戶問題」。請使用與問題相同的語言（繁體中文）來回答。

上下文信息:
---
{context}
---

用戶問題: {question}

請提供一個準確、詳細的回答。如果上下文中沒有足夠信息，請明確說明「根據提供的文件，我找不到相關資訊」。

回答:"""

    RELEVANCE_PROMPT_TEMPLATE = """你是一個文檔相關性評估專家。請簡明扼要地解釋為什麼下面的文檔內容與用戶查詢相關。請嚴格使用繁體中文回答。

用戶查詢: {question}
文檔內容:
---
{trimmed_content}
---
相關性理由:"""

    def get_language(self) -> str:
        return "繁體中文"
