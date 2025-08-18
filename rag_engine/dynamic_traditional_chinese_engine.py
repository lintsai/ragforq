"""
Dynamic RAG Engine for Traditional Chinese
"""
import logging
from .dynamic_rag_base import DynamicRAGEngineBase

logger = logging.getLogger(__name__)

class DynamicTraditionalChineseRAGEngine(DynamicRAGEngineBase):
    """動態繁體中文RAG引擎"""

    REWRITE_PROMPT_TEMPLATE = """你是一個搜尋優化專家。請將以下問題轉換為更適合在知識庫中檢索的完整描述性語句。

要求：
1. 保持繁體中文
2. 擴展相關的專業術語和同義詞
3. 包含可能在文檔中出現的不同表達方式
4. 保留問題的核心概念和語意

原始問題: {original_query}

優化後的檢索查詢:"""

    ANSWER_PROMPT_TEMPLATE = """你是一個專業的AI文檔問答助手。請嚴格根據以下「上下文信息」來回答「用戶問題」。請使用與問題相同的語言（繁體中文）來回答。

上下文信息:
---
{context}
---

用戶問題: {question}

請提供一個準確、詳細且簡潔的回答。如果上下文中沒有足夠信息，請明確說明「根據提供的文件，我找不到相關資訊」。避免重複內容。

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
    
    def _generate_general_knowledge_answer(self, question: str) -> str:
        """當找不到相關文檔時，基於常識提供繁體中文回答"""
        try:
            if self.llm is None:
                return f"抱歉，我在文檔中找不到與「{question}」相關的具體信息。這可能是因為相關文檔不在當前的檢索範圍內，或者問題涉及的內容需要更具體的關鍵詞。建議您嘗試使用更具體的關鍵詞重新提問。"
            
            general_prompt = f"""你是一個IT領域的專家助手。雖然在QSI內部文檔中找不到相關資料，但請基於一般IT常識來回答以下問題。

問題：{question}

請注意：
1. 使用繁體中文回答
2. 基於一般IT知識提供有用的回答
3. 明確說明這是基於常識的回答，不是來自QSI內部文檔
4. 如果是QSI特定的問題，建議聯繫相關部門
5. 回答要簡潔明瞭，避免重複內容

繁體中文回答："""
            
            try:
                response = self.llm.invoke(general_prompt)
                answer = response.content.strip() if hasattr(response, 'content') else str(response).strip()
                
                # 添加繁體中文免責聲明
                disclaimer = "\n\n※ 注意：以上回答基於一般IT常識，非來自QSI內部文檔。如需準確資訊，請聯繫相關部門。"
                return answer + disclaimer
                
            except Exception as e:
                logger.error(f"繁體中文動態RAG常識回答生成失敗: {str(e)}")
                return self._get_general_fallback(question)
        
        except Exception as e:
            logger.error(f"繁體中文動態RAG常識回答處理失敗: {str(e)}")
            return self._get_general_fallback(question)
    
    def _ensure_language(self, result: str) -> str:
        """確保輸出符合繁體中文"""
        return result
    
    def get_file_count_warning(self) -> str:
        """獲取文件數量警告"""
        return getattr(self.file_retriever, '_file_count_warning', None)
    
    def _get_error_message(self) -> str:
        """獲取繁體中文錯誤消息"""
        return "處理問題時發生錯誤，請稍後再試。"
    
    def _get_general_fallback(self, query: str) -> str:
        """獲取繁體中文通用回退回答"""
        return f"根據一般IT知識，關於「{query}」的相關信息可能需要查閱更多QSI內部文檔。"
