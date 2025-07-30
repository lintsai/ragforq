#!/usr/bin/env python
"""
帶超時機制的安全RAG引擎
解決Ollama調用hang住的問題
"""
import os
import sys
import time
import logging
import threading
import signal
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

# 添加項目根目錄到路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_engine.rag_engine import RAGEngine
from utils.read_only_vector_store import ReadOnlyDocumentIndexer
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

class TimeoutRAGEngine:
    """帶超時機制的安全RAG引擎"""
    
    def __init__(self, document_indexer, ollama_model: str, timeout: int = 30):
        """
        初始化超時RAG引擎
        
        Args:
            document_indexer: 文檔索引器
            ollama_model: Ollama模型名稱
            timeout: 超時時間（秒）
        """
        self.document_indexer = document_indexer
        self.ollama_model = ollama_model
        self.timeout = timeout
        self._rag_engine = None
        self._init_lock = threading.Lock()
        
        logger.info(f"初始化超時RAG引擎，超時時間: {timeout}秒")
    
    def _init_rag_engine(self):
        """延遲初始化RAG引擎"""
        if self._rag_engine is None:
            with self._init_lock:
                if self._rag_engine is None:
                    logger.info("正在初始化內部RAG引擎...")
                    self._rag_engine = RAGEngine(self.document_indexer, self.ollama_model)
                    logger.info("內部RAG引擎初始化完成")
        return self._rag_engine
    
    def _safe_call_with_timeout(self, func, *args, **kwargs):
        """
        安全調用函數，帶超時機制
        
        Args:
            func: 要調用的函數
            *args: 函數參數
            **kwargs: 函數關鍵字參數
            
        Returns:
            函數執行結果
        """
        with ThreadPoolExecutor(max_workers=1) as executor:
            try:
                future = executor.submit(func, *args, **kwargs)
                result = future.result(timeout=self.timeout)
                return result
            except FutureTimeoutError:
                logger.error(f"函數調用超時 ({self.timeout}秒): {func.__name__}")
                raise TimeoutError(f"操作超時 ({self.timeout}秒)")
            except Exception as e:
                logger.error(f"函數調用失敗: {func.__name__} - {str(e)}")
                raise
    
    def answer_question(self, question: str, language: str = "繁體中文") -> str:
        """
        安全的問答功能，帶超時機制
        
        Args:
            question: 用戶問題
            language: 回答語言
            
        Returns:
            回答文本
        """
        try:
            logger.info(f"開始處理問題: {question[:50]}...")
            
            # 初始化RAG引擎
            rag_engine = self._init_rag_engine()
            
            # 使用超時機制調用問答
            def _answer():
                return rag_engine.answer_question(question, language)
            
            answer = self._safe_call_with_timeout(_answer)
            
            logger.info(f"問答完成，回答長度: {len(answer)} 字符")
            return answer
            
        except TimeoutError as e:
            error_msg = f"問答超時: {str(e)}"
            logger.error(error_msg)
            return f"抱歉，系統響應超時。請稍後再試。({str(e)})"
        except Exception as e:
            error_msg = f"問答過程中發生錯誤: {str(e)}"
            logger.error(error_msg)
            return f"抱歉，處理您的問題時發生錯誤。請稍後再試。"
    
    def get_answer_with_sources(self, question: str, language: str = "繁體中文") -> Tuple[str, str, List[Document]]:
        """
        安全的帶來源問答功能
        
        Args:
            question: 用戶問題
            language: 回答語言
            
        Returns:
            (回答, 來源列表字符串, 相關文檔) 的元組
        """
        try:
            logger.info(f"開始處理帶來源問題: {question[:50]}...")
            
            # 初始化RAG引擎
            rag_engine = self._init_rag_engine()
            
            # 使用超時機制調用問答
            def _answer_with_sources():
                return rag_engine.get_answer_with_sources(question, language)
            
            result = self._safe_call_with_timeout(_answer_with_sources)
            
            logger.info("帶來源問答完成")
            return result
            
        except TimeoutError as e:
            error_msg = f"問答超時: {str(e)}"
            logger.error(error_msg)
            timeout_answer = f"抱歉，系統響應超時。請稍後再試。({str(e)})"
            return timeout_answer, "", []
        except Exception as e:
            error_msg = f"問答過程中發生錯誤: {str(e)}"
            logger.error(error_msg)
            error_answer = f"抱歉，處理您的問題時發生錯誤。請稍後再試。"
            return error_answer, "", []
    
    def get_answer_with_query_rewrite(self, original_query: str, language: str = "繁體中文") -> Tuple[str, str, List[Document], str]:
        """
        安全的查詢改寫問答功能
        
        Args:
            original_query: 原始查詢
            language: 回答語言
            
        Returns:
            (回答, 來源列表字符串, 相關文檔, 改寫後的查詢) 的元組
        """
        try:
            logger.info(f"開始處理查詢改寫問題: {original_query[:50]}...")
            
            # 初始化RAG引擎
            rag_engine = self._init_rag_engine()
            
            # 使用超時機制調用問答
            def _answer_with_rewrite():
                return rag_engine.get_answer_with_query_rewrite(original_query, language)
            
            result = self._safe_call_with_timeout(_answer_with_rewrite)
            
            logger.info("查詢改寫問答完成")
            return result
            
        except TimeoutError as e:
            error_msg = f"問答超時: {str(e)}"
            logger.error(error_msg)
            timeout_answer = f"抱歉，系統響應超時。請稍後再試。({str(e)})"
            return timeout_answer, "", [], original_query
        except Exception as e:
            error_msg = f"問答過程中發生錯誤: {str(e)}"
            logger.error(error_msg)
            error_answer = f"抱歉，處理您的問題時發生錯誤。請稍後再試。"
            return error_answer, "", [], original_query
    
    def generate_relevance_reason(self, question: str, doc_content: str) -> str:
        """
        安全的相關性理由生成
        
        Args:
            question: 用戶查詢
            doc_content: 文檔內容
            
        Returns:
            相關性理由描述
        """
        try:
            # 初始化RAG引擎
            rag_engine = self._init_rag_engine()
            
            # 使用較短的超時時間
            def _generate_reason():
                return rag_engine.generate_relevance_reason(question, doc_content)
            
            # 相關性理由生成使用較短超時時間
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_generate_reason)
                try:
                    result = future.result(timeout=10)  # 10秒超時
                    return result
                except FutureTimeoutError:
                    logger.warning("相關性理由生成超時")
                    return "相關性分析超時"
                    
        except Exception as e:
            logger.error(f"生成相關性理由失敗: {str(e)}")
            return "無法生成相關性理由"

def create_safe_rag_engine(vector_db_path: str, ollama_model: str, ollama_embedding_model: str, timeout: int = 30) -> TimeoutRAGEngine:
    """
    創建安全的RAG引擎
    
    Args:
        vector_db_path: 向量數據庫路徑
        ollama_model: Ollama語言模型
        ollama_embedding_model: Ollama嵌入模型
        timeout: 超時時間（秒）
        
    Returns:
        超時RAG引擎實例
    """
    try:
        logger.info(f"創建安全RAG引擎: {vector_db_path}")
        
        # 創建只讀文檔索引器
        document_indexer = ReadOnlyDocumentIndexer(vector_db_path, ollama_embedding_model)
        
        # 創建超時RAG引擎
        timeout_engine = TimeoutRAGEngine(document_indexer, ollama_model, timeout)
        
        logger.info("安全RAG引擎創建完成")
        return timeout_engine
        
    except Exception as e:
        logger.error(f"創建安全RAG引擎失敗: {str(e)}")
        raise