#!/usr/bin/env python
"""
只讀向量存儲管理器
專門用於問答階段，完全避免文件鎖定問題
"""
import os
import shutil
import tempfile
import logging
from typing import Optional, List
from pathlib import Path
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

class ReadOnlyVectorStore:
    """只讀向量存儲，專門用於問答，避免文件鎖定"""
    
    def __init__(self, vector_db_path: str, embeddings: Embeddings):
        """
        初始化只讀向量存儲
        
        Args:
            vector_db_path: 向量數據庫路徑
            embeddings: 嵌入模型
        """
        self.vector_db_path = Path(vector_db_path)
        self.embeddings = embeddings
        self._vector_store = None
        self._temp_dir = None
        self._is_loaded = False
        
    def _create_readonly_copy(self) -> Optional[Path]:
        """
        創建只讀副本，確保不會鎖定原文件
        
        Returns:
            臨時副本路徑，如果失敗則返回None
        """
        try:
            # 檢查原始文件是否存在
            faiss_file = self.vector_db_path / "index.faiss"
            pkl_file = self.vector_db_path / "index.pkl"
            
            if not (faiss_file.exists() and pkl_file.exists()):
                logger.debug(f"向量數據庫文件不存在: {self.vector_db_path}")
                return None
            
            # 創建臨時目錄
            self._temp_dir = tempfile.mkdtemp(prefix="readonly_vector_")
            temp_path = Path(self._temp_dir)
            
            # 使用只讀方式複製文件
            try:
                # 先檢查文件是否可讀
                with open(faiss_file, 'rb') as f:
                    faiss_data = f.read()
                with open(pkl_file, 'rb') as f:
                    pkl_data = f.read()
                
                # 寫入臨時文件
                with open(temp_path / "index.faiss", 'wb') as f:
                    f.write(faiss_data)
                with open(temp_path / "index.pkl", 'wb') as f:
                    f.write(pkl_data)
                
                logger.debug(f"創建只讀向量數據庫副本: {temp_path}")
                return temp_path
                
            except PermissionError as e:
                logger.warning(f"文件被鎖定，無法創建副本: {str(e)}")
                return None
            except Exception as e:
                logger.error(f"複製文件時出錯: {str(e)}")
                return None
            
        except Exception as e:
            logger.error(f"創建只讀副本失敗: {str(e)}")
            self._cleanup_temp()
            return None
    
    def _cleanup_temp(self):
        """清理臨時文件"""
        if self._temp_dir and os.path.exists(self._temp_dir):
            try:
                shutil.rmtree(self._temp_dir)
                logger.debug(f"清理臨時目錄: {self._temp_dir}")
            except Exception as e:
                logger.warning(f"清理臨時目錄失敗: {str(e)}")
            finally:
                self._temp_dir = None
    
    def load_vector_store(self) -> Optional[FAISS]:
        """
        加載只讀向量存儲
        
        Returns:
            FAISS向量存儲實例，如果失敗則返回None
        """
        if self._is_loaded and self._vector_store:
            return self._vector_store
        
        try:
            # 嘗試從只讀副本加載
            temp_path = self._create_readonly_copy()
            
            if temp_path:
                logger.info(f"從只讀副本加載向量數據庫: {temp_path}")
                self._vector_store = FAISS.load_local(
                    str(temp_path), 
                    self.embeddings, 
                    allow_dangerous_deserialization=True
                )
                self._is_loaded = True
                return self._vector_store
            
            # 如果無法創建副本，返回None
            logger.warning(f"無法創建只讀副本: {self.vector_db_path}")
            return None
            
        except Exception as e:
            logger.error(f"加載只讀向量存儲失敗: {str(e)}")
            return None
    
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """
        相似度搜索
        
        Args:
            query: 查詢文本
            k: 返回結果數量
            
        Returns:
            相關文檔列表
        """
        if not self._vector_store:
            self._vector_store = self.load_vector_store()
        
        if not self._vector_store:
            logger.warning("向量存儲未加載，返回空結果")
            return []
        
        try:
            return self._vector_store.similarity_search(query, k=k)
        except Exception as e:
            logger.error(f"相似度搜索失敗: {str(e)}")
            return []
    
    def similarity_search_with_score(self, query: str, k: int = 4) -> List[tuple]:
        """
        帶分數的相似度搜索
        
        Args:
            query: 查詢文本
            k: 返回結果數量
            
        Returns:
            (文檔, 分數) 元組列表
        """
        if not self._vector_store:
            self._vector_store = self.load_vector_store()
        
        if not self._vector_store:
            logger.warning("向量存儲未加載，返回空結果")
            return []
        
        try:
            return self._vector_store.similarity_search_with_score(query, k=k)
        except Exception as e:
            logger.error(f"帶分數相似度搜索失敗: {str(e)}")
            return []
    
    def is_available(self) -> bool:
        """
        檢查向量存儲是否可用
        
        Returns:
            是否可用
        """
        if self._is_loaded:
            return self._vector_store is not None
        
        # 嘗試加載
        vector_store = self.load_vector_store()
        return vector_store is not None
    
    def __del__(self):
        """析構函數，清理臨時文件"""
        self._cleanup_temp()

class ReadOnlyDocumentIndexer:
    """只讀文檔索引器，專門用於問答"""
    
    def __init__(self, vector_db_path: str, ollama_embedding_model: str):
        """
        初始化只讀文檔索引器
        
        Args:
            vector_db_path: 向量數據庫路徑
            ollama_embedding_model: 嵌入模型名稱
        """
        from langchain_ollama import OllamaEmbeddings
        from config.config import OLLAMA_HOST
        
        self.vector_db_path = vector_db_path
        self.ollama_embedding_model = ollama_embedding_model
        
        # 創建嵌入模型
        self.embeddings = OllamaEmbeddings(
            base_url=OLLAMA_HOST,
            model=ollama_embedding_model
        )
        
        # 創建只讀向量存儲
        self.readonly_store = ReadOnlyVectorStore(vector_db_path, self.embeddings)
        
        # 模擬DocumentIndexer的接口
        self.indexed_files = {}  # 空的索引文件記錄
        
        logger.info(f"只讀文檔索引器初始化完成: {vector_db_path}")
    
    def get_vector_store(self) -> Optional[FAISS]:
        """
        獲取向量存儲實例
        
        Returns:
            FAISS向量存儲實例，如果不可用則返回None
        """
        return self.readonly_store.load_vector_store()
    
    def retrieve_documents(self, query: str, top_k: int = 5) -> List[Document]:
        """
        檢索相關文檔
        
        Args:
            query: 查詢文本
            top_k: 返回的文檔數量
            
        Returns:
            相關文檔列表
        """
        try:
            if not self.readonly_store.is_available():
                logger.warning("只讀向量存儲不可用")
                return []
            
            # 使用帶分數的搜索
            docs_with_scores = self.readonly_store.similarity_search_with_score(query, k=top_k)
            
            # 過濾相似度過低的文檔
            filtered_docs = []
            for doc, score in docs_with_scores:
                if score < 1.5:  # 距離小於1.5認為是相關的
                    doc.metadata['score'] = score
                    filtered_docs.append(doc)
            
            logger.info(f"檢索到 {len(docs_with_scores)} 個文檔，過濾後保留 {len(filtered_docs)} 個相關文檔")
            return filtered_docs
            
        except Exception as e:
            logger.error(f"檢索文檔時出錯: {str(e)}")
            return []
    
    def list_indexed_files(self) -> List[dict]:
        """
        列出已索引的文件（只讀模式下返回空列表）
        
        Returns:
            空列表
        """
        logger.info("只讀模式下無法列出索引文件")
        return []