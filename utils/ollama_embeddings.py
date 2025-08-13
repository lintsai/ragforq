"""
Ollama 嵌入模型包裝器
"""

import logging
import requests
import numpy as np
from typing import List, Union
from config.config import OLLAMA_HOST, OLLAMA_EMBEDDING_TIMEOUT

logger = logging.getLogger(__name__)

class OllamaEmbeddings:
    """Ollama 嵌入模型包裝器"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.host = OLLAMA_HOST.rstrip('/')
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """嵌入文檔列表"""
        try:
            embeddings = []
            for text in texts:
                embedding = self._get_embedding(text)
                embeddings.append(embedding)
            return embeddings
        except Exception as e:
            logger.error(f"Ollama 文檔嵌入失敗: {str(e)}")
            # 返回零向量作為回退
            return [[0.0] * 384 for _ in texts]
    
    def embed_query(self, text: str) -> List[float]:
        """嵌入查詢文本"""
        try:
            return self._get_embedding(text)
        except Exception as e:
            logger.error(f"Ollama 查詢嵌入失敗: {str(e)}")
            # 返回零向量作為回退
            return [0.0] * 384
    
    def _get_embedding(self, text: str) -> List[float]:
        """從 Ollama 獲取嵌入向量"""
        try:
            response = requests.post(
                f"{self.host}/api/embeddings",
                json={
                    "model": self.model_name,
                    "prompt": text
                },
                timeout=OLLAMA_EMBEDDING_TIMEOUT
            )
            response.raise_for_status()
            
            result = response.json()
            embedding = result.get("embedding", [])
            
            if not embedding:
                logger.warning(f"Ollama 返回空嵌入向量: {text[:50]}...")
                return [0.0] * 384
            
            return embedding
            
        except Exception as e:
            logger.error(f"Ollama 嵌入請求失敗: {str(e)}")
            raise