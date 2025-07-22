#!/usr/bin/env python
"""
Ollama API 工具函數
用於獲取可用的模型列表和管理模型相關操作
"""
import requests
import logging
from typing import List, Dict, Any, Optional
from config.config import OLLAMA_HOST

logger = logging.getLogger(__name__)

class OllamaUtils:
    """Ollama API 工具類"""
    
    def __init__(self, host: str = OLLAMA_HOST):
        """
        初始化 Ollama 工具
        
        Args:
            host: Ollama 服務地址
        """
        self.host = host.rstrip('/')
        
    def get_available_models(self) -> List[Dict[str, Any]]:
        """
        獲取可用的模型列表
        
        Returns:
            模型列表，每個模型包含 name, size, modified 等信息
        """
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=10)
            response.raise_for_status()
            
            data = response.json()
            models = data.get('models', [])
            
            # 格式化模型信息
            formatted_models = []
            for model in models:
                formatted_models.append({
                    'name': model.get('name', ''),
                    'size': model.get('size', 0),
                    'modified': model.get('modified_at', ''),
                    'digest': model.get('digest', ''),
                    'details': model.get('details', {})
                })
            
            logger.info(f"獲取到 {len(formatted_models)} 個可用模型")
            return formatted_models
            
        except requests.exceptions.RequestException as e:
            logger.error(f"獲取 Ollama 模型列表失敗: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"處理 Ollama 模型列表時出錯: {str(e)}")
            return []
    
    def get_model_names(self) -> List[str]:
        """
        獲取模型名稱列表
        
        Returns:
            模型名稱列表
        """
        models = self.get_available_models()
        return [model['name'] for model in models]
    
    def is_model_available(self, model_name: str) -> bool:
        """
        檢查指定模型是否可用
        
        Args:
            model_name: 模型名稱
            
        Returns:
            模型是否可用
        """
        available_models = self.get_model_names()
        return model_name in available_models
    
    def check_ollama_connection(self) -> bool:
        """
        檢查 Ollama 服務連接狀態
        
        Returns:
            連接是否正常
        """
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False

# 全局實例
ollama_utils = OllamaUtils()