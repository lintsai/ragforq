import os
import sys
import logging
import concurrent.futures
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path

# 添加項目根目錄到路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_engine.interfaces import RAGEngineInterface
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from utils.hf_langchain_wrapper import HuggingFaceLLM, ChatHuggingFace
from config.config import (
    OLLAMA_QUERY_OPTIMIZATION_TIMEOUT, 
    OLLAMA_ANSWER_GENERATION_TIMEOUT, 
    OLLAMA_RELEVANCE_TIMEOUT
)
# 設置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TraditionalChineseRAGEngine(RAGEngineInterface):
    """繁體中文RAG引擎實現"""
    
    def __init__(self, document_indexer, ollama_model: str = None, platform: str = None):
        super().__init__(document_indexer, ollama_model)
        
        # 如果沒有指定平台，自動檢測
        if platform is None:
            from config.config import detect_platform_from_model
            platform = detect_platform_from_model(ollama_model)
        
        # 根據傳入的平台參數初始化不同的 LLM
        if platform == "ollama":
            from langchain_ollama import OllamaLLM
            from config.config import OLLAMA_HOST
            self.llm = OllamaLLM(
                model=ollama_model,
                base_url=OLLAMA_HOST,
                temperature=0.4
            )
            logger.info(f"繁體中文RAG引擎初始化完成 (Ollama)，使用模型: {ollama_model}")
        else:
            # Hugging Face 平台
            self.llm = ChatHuggingFace(
                model_name=ollama_model,
                temperature=0.4
            )
            logger.info(f"繁體中文RAG引擎初始化完成 (Hugging Face)，使用模型: {ollama_model}")
    
    def get_language(self) -> str:
        return "繁體中文"
    
    def rewrite_query(self, original_query: str) -> str:
        """
        將繁體中文查詢優化為更精準適合向量檢索的完整描述 - 帶重試機制
        """
        from config.config import OLLAMA_MAX_RETRIES, OLLAMA_RETRY_DELAY
        
        for attempt in range(OLLAMA_MAX_RETRIES):
            try:
                rewrite_prompt = PromptTemplate(
                    template="""你是一個搜尋優化專家。請將以下問題轉換為更適合在知識庫中檢索的關鍵詞或描述性語句。請嚴格使用繁體中文輸出。

原始問題: {question}

優化後的檢索查詢:""",
                    input_variables=["question"]
                )
                
                def _invoke_rewrite():
                    # 直接調用而不使用鏈式操作
                    prompt_text = rewrite_prompt.format(question=original_query)
                    response = self.llm.invoke(prompt_text)
                    if hasattr(response, 'content'):
                        return response.content
                    else:
                        return str(response)
                
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(_invoke_rewrite)
                    try:
                        optimized_query = future.result(timeout=OLLAMA_QUERY_OPTIMIZATION_TIMEOUT)
                        logger.info(f"繁體中文查詢優化: {original_query} -> {optimized_query}")
                        return optimized_query.strip()
                    except concurrent.futures.TimeoutError:
                        if attempt < OLLAMA_MAX_RETRIES - 1:
                            logger.warning(f"查詢優化超時，第 {attempt + 1} 次重試...")
                            time.sleep(OLLAMA_RETRY_DELAY)
                            continue
                        else:
                            logger.error("查詢優化多次超時，使用原始查詢")
                            return original_query
                
            except Exception as e:
                if attempt < OLLAMA_MAX_RETRIES - 1:
                    logger.warning(f"查詢優化出錯，第 {attempt + 1} 次重試: {str(e)}")
                    time.sleep(OLLAMA_RETRY_DELAY)
                    continue
                else:
                    logger.error(f"查詢優化多次失敗: {str(e)}")
                    return original_query
        
        return original_query
    
    def answer_question(self, question: str) -> str:
        """使用繁體中文回答問題"""
        try:
            # 優化查詢
            optimized_query = self.rewrite_query(question)
            
            # 檢索文檔
            docs = self.retrieve_documents(optimized_query)
            
            if not docs:
                return self._generate_general_knowledge_answer(question)
            
            # 格式化上下文
            context = self.format_context(docs)
            
            # 生成回答
            return self._generate_answer(question, context)
            
        except Exception as e:
            logger.error(f"繁體中文問答時出錯: {str(e)}")
            return f"{self._get_error_message()}: {str(e)}"
    
    def _generate_answer(self, question: str, context: str) -> str:
        """生成繁體中文回答"""
        template = """你是一個專業的AI文檔問答助手。請嚴格根據以下「上下文信息」來回答「用戶問題」。請使用與問題相同的語言（繁體中文）來回答。

上下文信息:
---
{context}
---

用戶問題: {question}

請提供一個準確、詳細的回答。如果上下文中沒有足夠信息，請明確說明「根據提供的文件，我找不到相關資訊」。

回答:"""

        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        def _invoke():
            # 直接調用而不使用鏈式操作
            prompt_text = prompt.format(context=context, question=question)
            response = self.llm.invoke(prompt_text)
            if hasattr(response, 'content'):
                return response.content
            else:
                return str(response)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_invoke)
            try:
                answer = future.result(timeout=OLLAMA_ANSWER_GENERATION_TIMEOUT)
                
                if not answer or len(answer.strip()) < 5:
                    return self._get_general_fallback(question)
                
                # 檢測並修復重複內容
                cleaned_answer = self._clean_repetitive_content(answer.strip())
                return cleaned_answer
                
            except concurrent.futures.TimeoutError:
                logger.error("繁體中文回答生成超時")
                return self._get_timeout_message()
    
    def generate_relevance_reason(self, question: str, doc_content: str) -> str:
        """生成繁體中文相關性理由"""
        if not question or not question.strip():
            return "無法生成相關性理由：查詢為空"
        
        if not doc_content or not doc_content.strip():
            return "無法生成相關性理由：文檔內容為空"
            
        try:
            trimmed_content = doc_content[:1000].strip()
            
            relevance_prompt = PromptTemplate(
                template="""你是一個文檔相關性評估專家。請簡明扼要地解釋為什麼下面的文檔內容與用戶查詢相關。請嚴格使用繁體中文回答。

用戶查詢: {question}
文檔內容:
---
{doc_content}
---
相關性理由：""",
                input_variables=["question", "doc_content"]
            )
            
            relevance_chain = relevance_prompt | self.llm | StrOutputParser()
            
            def _invoke_relevance():
                return relevance_chain.invoke({
                    "question": question,
                    "doc_content": trimmed_content
                })
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_invoke_relevance)
                try:
                    reason = future.result(timeout=OLLAMA_RELEVANCE_TIMEOUT)
                    return reason.strip() if reason else "無法確定相關性理由"
                except concurrent.futures.TimeoutError:
                    return "生成相關性理由超時"
            
        except Exception as e:
            logger.error(f"生成繁體中文相關性理由時出錯: {str(e)}")
            return "生成相關性理由失敗"
    
    def _get_no_docs_message(self) -> str:
        return "抱歉，在QSI文檔中找不到與您問題相關的信息。"
    
    def _get_error_message(self) -> str:
        return "處理問題時發生錯誤"
    
    def _get_timeout_message(self) -> str:
        return "系統處理超時，請稍後再試。"
    
    def _generate_general_knowledge_answer(self, question: str) -> str:
        """當找不到相關文檔時，基於常識提供回答"""
        try:
            general_prompt = PromptTemplate(
                template="""你是一個IT領域的專家助手。雖然在QSI內部文檔中找不到相關資料，但請基於一般IT常識來回答以下問題。

問題：{question}

請注意：
1. 使用繁體中文回答
2. 基於一般IT知識提供有用的回答
3. 明確說明這是基於常識的回答，不是來自QSI內部文檔
4. 如果是QSI特定的問題，建議聯繫相關部門

繁體中文回答：""",
                input_variables=["question"]
            )
            
            general_chain = general_prompt | self.llm | StrOutputParser()
            
            def _invoke_general():
                return general_chain.invoke({"question": question})
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_invoke_general)
                try:
                    answer = future.result(timeout=OLLAMA_ANSWER_GENERATION_TIMEOUT)
                    # 添加免責聲明
                    disclaimer = "\n\n※ 注意：以上回答基於一般IT常識，非來自QSI內部文檔。如需準確資訊，請聯繫相關部門。"
                    return answer.strip() + disclaimer
                except concurrent.futures.TimeoutError:
                    logger.error("常識回答生成超時")
                    return self._get_no_docs_message()
        
        except Exception as e:
            logger.error(f"生成常識回答時出錯: {str(e)}")
            return self._get_no_docs_message()
    

    
    def generate_batch_relevance_reasons(self, question: str, doc_contents: list) -> list:
        """批量生成多個文檔的相關性理由，提高效能"""
        if not question or not question.strip() or not doc_contents:
            return ["無法生成相關性理由"] * len(doc_contents)
        
        try:
            # 構建批量處理的prompt
            docs_text = ""
            for i, content in enumerate(doc_contents, 1):
                if content and content.strip():
                    docs_text += f"文檔{i}: {content[:300]}...\n\n"
                else:
                    docs_text += f"文檔{i}: (空內容)\n\n"
            
            batch_prompt = PromptTemplate(
                template="""請為以下文檔分別生成與用戶查詢的相關性理由。每個理由用一句話簡潔說明。

用戶查詢: {question}

文檔內容:
{docs_text}

請按順序為每個文檔生成相關性理由，格式如下:
1. [文檔1的相關性理由]
2. [文檔2的相關性理由]
3. [文檔3的相關性理由]
...

相關性理由:""",
                input_variables=["question", "docs_text"]
            )
            
            batch_chain = batch_prompt | self.llm | StrOutputParser()
            
            def _invoke_batch():
                return batch_chain.invoke({
                    "question": question,
                    "docs_text": docs_text
                })
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_invoke_batch)
                try:
                    batch_result = future.result(timeout=OLLAMA_RELEVANCE_TIMEOUT)
                    
                    # 解析批量結果
                    reasons = []
                    lines = batch_result.strip().split('\n')
                    for line in lines:
                        line = line.strip()
                        if line and (line.startswith(('1.', '2.', '3.', '4.', '5.')) or '.' in line[:3]):
                            # 移除序號，保留理由
                            reason = line.split('.', 1)[1].strip() if '.' in line else line
                            reasons.append(reason)
                    
                    # 確保返回正確數量的理由
                    while len(reasons) < len(doc_contents):
                        reasons.append("相關文檔")
                    
                    return reasons[:len(doc_contents)]
                    
                except concurrent.futures.TimeoutError:
                    logger.error("批量生成相關性理由超時")
                    return [f"相關文檔 {i+1}" for i in range(len(doc_contents))]
        
        except Exception as e:
            logger.error(f"批量生成相關性理由時出錯: {str(e)}")
            return [f"相關文檔 {i+1}" for i in range(len(doc_contents))]
    
    def _get_general_fallback(self, query: str) -> str:
        return f"根據一般IT知識，關於「{query}」的相關信息可能需要查閱更多QSI內部文檔。"
    
    def _clean_repetitive_content(self, text: str) -> str:
        """清理重複內容"""
        if not text:
            return text
        
        # 分割成段落
        paragraphs = text.split('\n\n')
        cleaned_paragraphs = []
        seen_content = set()
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # 檢查是否是重複段落（使用前100個字符作為指紋）
            fingerprint = paragraph[:100].strip()
            if fingerprint not in seen_content:
                seen_content.add(fingerprint)
                cleaned_paragraphs.append(paragraph)
            else:
                logger.warning(f"檢測到重複段落，已移除: {fingerprint[:50]}...")
        
        # 重新組合
        cleaned_text = '\n\n'.join(cleaned_paragraphs)
        
        # 檢查是否有句子級別的重複
        sentences = cleaned_text.split('。')
        cleaned_sentences = []
        seen_sentences = set()
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # 檢查是否是重複句子（使用前50個字符作為指紋）
            sentence_fingerprint = sentence[:50].strip()
            if sentence_fingerprint not in seen_sentences:
                seen_sentences.add(sentence_fingerprint)
                cleaned_sentences.append(sentence)
            else:
                logger.warning(f"檢測到重複句子，已移除: {sentence_fingerprint[:30]}...")
        
        final_text = '。'.join(cleaned_sentences)
        if final_text and not final_text.endswith('。'):
            final_text += '。'
        
        # 如果清理後內容太短，返回原始內容的前500字符
        if len(final_text.strip()) < 50:
            logger.warning("清理後內容過短，返回原始內容的截取版本")
            return text[:500] + "..." if len(text) > 500 else text
        
        return final_text
