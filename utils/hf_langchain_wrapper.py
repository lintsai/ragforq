"""
Hugging Face LangChain 包裝器
提供與 LangChain 兼容的 Hugging Face 模型接口
"""

import logging
from typing import List, Optional, Any, Dict
import numpy as np
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun

from utils.model_manager import get_model_manager

logger = logging.getLogger(__name__)

class HuggingFaceEmbeddings(Embeddings):
    """Hugging Face 嵌入模型的 LangChain 包裝器"""
    
    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name
        self.model_manager = get_model_manager()
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """嵌入文檔列表"""
        try:
            embeddings = self.model_manager.encode_texts(texts, self.model_name)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"文檔嵌入失敗: {str(e)}")
            # 返回零向量作為回退
            return [[0.0] * 384 for _ in texts]
    
    def embed_query(self, text: str) -> List[float]:
        """嵌入查詢文本"""
        try:
            embedding = self.model_manager.encode_texts(text, self.model_name)
            return embedding[0].tolist()
        except Exception as e:
            logger.error(f"查詢嵌入失敗: {str(e)}")
            # 返回零向量作為回退
            return [0.0] * 384

class HuggingFaceLLM(LLM):
    """Hugging Face 語言模型的 LangChain 包裝器"""
    
    def __init__(self, model_name: Optional[str] = None, temperature: float = 0.1, 
                 max_length: int = 4096):
        super().__init__()
        self.model_name = model_name
        self.temperature = temperature
        self.max_length = max_length
        self.model_manager = get_model_manager()
    
    @property
    def _llm_type(self) -> str:
        return "huggingface"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """調用模型生成文本"""
        try:
            # 合併參數
            generation_kwargs = {
                "temperature": kwargs.get("temperature", self.temperature),
                "max_length": kwargs.get("max_length", self.max_length),
            }
            
            response = self.model_manager.generate_text(
                prompt, 
                self.model_name, 
                **generation_kwargs
            )
            
            # 處理停止詞
            if stop:
                for stop_word in stop:
                    if stop_word in response:
                        response = response.split(stop_word)[0]
                        break
            
            return response
            
        except Exception as e:
            logger.error(f"LLM 調用失敗: {str(e)}")
            return "抱歉，處理您的請求時發生錯誤。"
    
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """返回識別參數"""
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_length": self.max_length,
        }

try:
    from langchain_core.runnables import Runnable
    from langchain_core.runnables.utils import Input, Output
    LANGCHAIN_AVAILABLE = True
except ImportError:
    # 如果 LangChain 不可用，創建基本的 Runnable 類
    class Runnable:
        def __init__(self):
            pass
    Input = Any
    Output = Any
    LANGCHAIN_AVAILABLE = False

class ChatHuggingFace(Runnable):
    """聊天式 Hugging Face 模型包裝器 - 完全兼容 LangChain Runnable"""
    
    def __init__(self, model_name: Optional[str] = None, temperature: float = 0.1, max_new_tokens: int = 4096):
        super().__init__()
        self.model_name = model_name
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.model_manager = get_model_manager()
    
    def invoke(self, input_data, config=None, **kwargs) -> "AIMessage":
        """調用模型並返回 AIMessage 格式的響應 - 完全兼容 LangChain"""
        try:
            # 處理不同類型的輸入
            if isinstance(input_data, dict):
                prompt = input_data.get("question", str(input_data))
            else:
                prompt = str(input_data)
            
            response = self.model_manager.generate_text(
                prompt,
                self.model_name,
                temperature=kwargs.get("temperature", self.temperature),
                max_length=kwargs.get("max_length", self.max_new_tokens)
            )
            
            return AIMessage(content=response)
            
        except Exception as e:
            logger.error(f"聊天模型調用失敗: {str(e)}")
            return AIMessage(content="抱歉，處理您的請求時發生錯誤。")
    
    @property
    def InputType(self):
        return str
    
    @property
    def OutputType(self):
        return AIMessage

class AIMessage:
    """AI 消息類，模擬 LangChain 的 AIMessage"""
    
    def __init__(self, content: str):
        self.content = content
    
    def __str__(self):
        return self.content