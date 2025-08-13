"""
平台管理器 - 統一管理 Hugging Face 和 Ollama 平台
"""

import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import requests

from config.config import (
    INFERENCE_ENGINE, DEFAULT_LLM_MODEL, DEFAULT_EMBEDDING_MODEL,
    ENVIRONMENT
)
from utils.huggingface_utils import huggingface_utils
from utils.ollama_utils import ollama_utils
from utils.model_manager import get_model_manager
from utils.vllm_manager import get_vllm_manager

logger = logging.getLogger(__name__)

class PlatformType(Enum):
    """平台類型枚舉"""
    HUGGINGFACE = "huggingface"
    OLLAMA = "ollama"

class ModelType(Enum):
    """模型類型枚舉"""
    LANGUAGE_MODEL = "language_model"
    EMBEDDING_MODEL = "embedding_model"

class PlatformManager:
    """平台管理器 - 統一管理不同AI平台"""
    
    def __init__(self):
        self.current_platform = PlatformType.HUGGINGFACE  # 默認使用 Hugging Face
        self.available_platforms = [PlatformType.HUGGINGFACE, PlatformType.OLLAMA]
        self.model_cache = {}
        
    def get_available_platforms(self) -> List[Dict[str, Any]]:
        """獲取可用平台列表"""
        platforms = []
        
        # Hugging Face 平台
        hf_status = self._check_huggingface_status()
        platforms.append({
            "type": PlatformType.HUGGINGFACE.value,
            "name": "Hugging Face",
            "description": "開源模型平台，支援 vLLM 高性能推理",
            "status": "available" if hf_status else "unavailable",
            "features": [
                "豐富的模型選擇",
                "vLLM 高性能推理",
                "生產級部署",
                "官方模型支援"
            ],
            "recommended": True
        })
        
        # Ollama 平台
        ollama_status = self._check_ollama_status()
        platforms.append({
            "type": PlatformType.OLLAMA.value,
            "name": "Ollama",
            "description": "本地模型運行平台，易於部署和管理",
            "status": "available" if ollama_status else "unavailable",
            "features": [
                "本地部署",
                "易於管理",
                "資源效率高",
                "快速啟動"
            ],
            "recommended": False
        })
        
        return platforms
    
    def set_platform(self, platform_type: str) -> bool:
        """設置當前使用的平台"""
        try:
            self.current_platform = PlatformType(platform_type)
            logger.info(f"切換到平台: {self.current_platform.value}")
            return True
        except ValueError:
            logger.error(f"不支援的平台類型: {platform_type}")
            return False
    
    def get_available_models(self, model_type: ModelType = None) -> Dict[str, List[Dict[str, Any]]]:
        """獲取當前平台的可用模型"""
        if self.current_platform == PlatformType.HUGGINGFACE:
            return self._get_huggingface_models(model_type)
        elif self.current_platform == PlatformType.OLLAMA:
            return self._get_ollama_models(model_type)
        else:
            return {"language_models": [], "embedding_models": []}
    
    def _check_huggingface_status(self) -> bool:
        """檢查 Hugging Face 平台狀態"""
        try:
            # 檢查是否能導入必要的庫
            import transformers
            import torch
            
            # 檢查 vLLM（如果配置使用）
            if INFERENCE_ENGINE == "vllm":
                try:
                    import vllm
                    return True
                except ImportError:
                    logger.warning("vLLM 未安裝，但 Hugging Face 基本功能可用")
                    return True
            
            return True
        except ImportError as e:
            logger.error(f"Hugging Face 依賴缺失: {str(e)}")
            return False
    
    def _check_ollama_status(self) -> bool:
        """檢查 Ollama 平台狀態"""
        return ollama_utils.check_ollama_connection()
    
    def _get_huggingface_models(self, model_type: ModelType = None) -> Dict[str, List[Dict[str, Any]]]:
        """獲取 Hugging Face 本地可用模型列表"""
        models = {
            "language_models": [],
            "embedding_models": []
        }
        
        try:
            # 獲取本地模型
            local_models = huggingface_utils.get_local_models()
            
            # 分類模型
            for model in local_models:
                model_info = {
                    "id": model['name'],
                    "name": self._format_model_display_name(model['name']),
                    "description": f"本地模型 ({model['size_formatted']})",
                    "size": model['size_formatted'],
                    "path": model['path'],
                    "available": True,
                    "local": True
                }
                
                if model['type'] == 'language':
                    if model_type is None or model_type == ModelType.LANGUAGE_MODEL:
                        models["language_models"].append(model_info)
                elif model['type'] == 'embedding':
                    if model_type is None or model_type == ModelType.EMBEDDING_MODEL:
                        models["embedding_models"].append(model_info)
            
        except Exception as e:
            logger.error(f"獲取 Hugging Face 模型列表時出錯: {str(e)}")
        
        return models
    
    def _format_model_display_name(self, model_name: str) -> str:
        """格式化模型顯示名稱"""
        # 移除組織名稱前綴，讓名稱更簡潔
        if '/' in model_name:
            return model_name.split('/')[-1]
        return model_name
    

    
    def _get_ollama_models(self, model_type: ModelType = None) -> Dict[str, List[Dict[str, Any]]]:
        """獲取 Ollama 模型列表"""
        models = {
            "language_models": [],
            "embedding_models": []
        }
        
        try:
            available_models = ollama_utils.get_available_models()
            
            for model in available_models:
                model_name = model['name'].lower()
                model_info = {
                    "id": model['name'],
                    "name": model['name'],
                    "description": f"Ollama 本地模型 ({model['size']} bytes)",
                    "size": self._format_size(model['size']),
                    "recommended": False,
                    "modified": model['modified']
                }
                
                # 根據模型名稱分類
                if any(embed_keyword in model_name for embed_keyword in ['embed', 'embedding', 'nomic']):
                    if model_type is None or model_type == ModelType.EMBEDDING_MODEL:
                        models["embedding_models"].append(model_info)
                else:
                    if model_type is None or model_type == ModelType.LANGUAGE_MODEL:
                        models["language_models"].append(model_info)
            
        except Exception as e:
            logger.error(f"獲取 Ollama 模型失敗: {str(e)}")
        
        return models
    
    def _format_size(self, size_bytes: int) -> str:
        """格式化文件大小"""
        if size_bytes < 1024**3:
            return f"{size_bytes / 1024**2:.1f}MB"
        else:
            return f"{size_bytes / 1024**3:.1f}GB"
    
    def validate_model_selection(self, language_model: str, embedding_model: str) -> Tuple[bool, str]:
        """驗證模型選擇的兼容性"""
        if self.current_platform == PlatformType.HUGGINGFACE:
            return self._validate_huggingface_models(language_model, embedding_model)
        elif self.current_platform == PlatformType.OLLAMA:
            return self._validate_ollama_models(language_model, embedding_model)
        else:
            return False, "未知平台"
    
    def _validate_huggingface_models(self, language_model: str, embedding_model: str) -> Tuple[bool, str]:
        """驗證 Hugging Face 模型選擇"""
        # 檢查大型模型的硬體需求
        large_models = ["openai/gpt-oss-20b", "EleutherAI/gpt-neox-20b"]
        
        if language_model in large_models:
            if INFERENCE_ENGINE != "vllm":
                return False, "大型模型建議使用 vLLM 推理引擎以獲得最佳性能"
        
        return True, "模型選擇有效"
    
    def _validate_ollama_models(self, language_model: str, embedding_model: str) -> Tuple[bool, str]:
        """驗證 Ollama 模型選擇"""
        available_models = ollama_utils.get_model_names()
        
        if language_model not in available_models:
            return False, f"語言模型 {language_model} 在 Ollama 中不可用"
        
        if embedding_model not in available_models:
            return False, f"嵌入模型 {embedding_model} 在 Ollama 中不可用"
        
        return True, "模型選擇有效"
    
    def get_recommended_config(self) -> Dict[str, Any]:
        """獲取推薦配置"""
        if self.current_platform == PlatformType.HUGGINGFACE:
            if ENVIRONMENT == "production":
                return {
                    "platform": PlatformType.HUGGINGFACE.value,
                    "language_model": "openai/gpt-oss-20b",
                    "embedding_model": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
                    "inference_engine": "vllm",
                    "reason": "生產環境推薦使用 gpt-oss-20b + vLLM 獲得最佳性能"
                }
            else:
                return {
                    "platform": PlatformType.HUGGINGFACE.value,
                    "language_model": "Qwen/Qwen2-0.5B-Instruct",
                    "embedding_model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                    "inference_engine": "transformers",
                    "reason": "開發環境推薦使用多語言友善模型（Qwen2 0.5B）"
                }
        else:
            # Ollama 推薦配置
            available_models = ollama_utils.get_model_names()
            
            # 尋找推薦的語言模型
            language_model = None
            for model in available_models:
                if 'embed' not in model.lower():
                    language_model = model
                    break
            
            # 尋找推薦的嵌入模型
            embedding_model = None
            for model in available_models:
                if 'embed' in model.lower():
                    embedding_model = model
                    break
            
            return {
                "platform": PlatformType.OLLAMA.value,
                "language_model": language_model,
                "embedding_model": embedding_model,
                "inference_engine": "ollama",
                "reason": "Ollama 本地部署，易於管理"
            }

# 全局平台管理器實例
platform_manager = PlatformManager()

def get_platform_manager() -> PlatformManager:
    """獲取全局平台管理器實例"""
    return platform_manager
