"""
設置流程管理器 - 管理用戶配置流程
"""

import os
import logging
from typing import Dict, Any, Optional, List
from enum import Enum
import json
from pathlib import Path

from utils.platform_manager import get_platform_manager, PlatformType, ModelType
from utils.ollama_utils import ollama_utils
from utils.vector_db_manager import vector_db_manager

logger = logging.getLogger(__name__)

class SetupStep(Enum):
    """設置步驟枚舉"""
    PLATFORM_SELECTION = "platform_selection"
    MODEL_SELECTION = "model_selection"
    RAG_MODE_SELECTION = "rag_mode_selection"
    CONFIGURATION_REVIEW = "configuration_review"
    READY = "ready"

class RAGMode(Enum):
    """RAG 模式枚舉"""
    TRADITIONAL = "traditional"
    DYNAMIC = "dynamic"

class SetupFlowManager:
    """設置流程管理器"""
    
    def __init__(self):
        self.platform_manager = get_platform_manager()
        self.config_file = Path("config/user_setup.json")
        self.current_config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """加載用戶配置"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"加載配置失敗: {str(e)}")
        
        # 返回默認配置
        return {
            "setup_completed": False,
            "current_step": SetupStep.PLATFORM_SELECTION.value,
            "platform": PlatformType.HUGGINGFACE.value,
            "language_model": None,
            "embedding_model": None,
            "rag_mode": RAGMode.TRADITIONAL.value,
            "use_dynamic_rag": False,
            "inference_engine": "transformers",
            "language": "繁體中文"
        }
    
    def _save_config(self):
        """保存用戶配置"""
        try:
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.current_config, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存配置失敗: {str(e)}")
    
    def get_current_step(self) -> str:
        """獲取當前設置步驟"""
        return self.current_config.get("current_step", SetupStep.PLATFORM_SELECTION.value)
    
    def is_setup_completed(self) -> bool:
        """檢查設置是否完成"""
        return self.current_config.get("setup_completed", False)
    
    def get_setup_progress(self) -> Dict[str, Any]:
        """獲取設置進度"""
        steps = [
            SetupStep.PLATFORM_SELECTION,
            SetupStep.MODEL_SELECTION,
            SetupStep.RAG_MODE_SELECTION,
            SetupStep.CONFIGURATION_REVIEW,
            SetupStep.READY
        ]
        
        current_step = self.get_current_step()
        current_index = 0
        
        for i, step in enumerate(steps):
            if step.value == current_step:
                current_index = i
                break
        
        return {
            "current_step": current_step,
            "current_index": current_index,
            "total_steps": len(steps),
            "progress_percentage": (current_index / (len(steps) - 1)) * 100,
            "completed": self.is_setup_completed()
        }
    
    def get_platform_selection_data(self) -> Dict[str, Any]:
        """獲取平台選擇數據"""
        platforms = self.platform_manager.get_available_platforms()
        
        return {
            "step": SetupStep.PLATFORM_SELECTION.value,
            "title": "選擇 AI 平台",
            "description": "請選擇您要使用的 AI 模型平台",
            "platforms": platforms,
            "current_selection": self.current_config.get("platform"),
            "can_proceed": bool(self.current_config.get("platform"))
        }
    
    def set_platform(self, platform_type: str) -> Dict[str, Any]:
        """設置平台類型"""
        success = self.platform_manager.set_platform(platform_type)
        
        if success:
            self.current_config["platform"] = platform_type
            self.current_config["current_step"] = SetupStep.MODEL_SELECTION.value
            # 清除之前的模型選擇
            self.current_config["language_model"] = None
            self.current_config["embedding_model"] = None
            self._save_config()
            
            return {
                "success": True,
                "message": f"已選擇平台: {platform_type}",
                "next_step": SetupStep.MODEL_SELECTION.value
            }
        else:
            return {
                "success": False,
                "message": f"不支援的平台: {platform_type}"
            }
    
    def get_model_selection_data(self) -> Dict[str, Any]:
        """獲取模型選擇數據"""
        if not self.current_config.get("platform"):
            return {"error": "請先選擇平台"}
        
        # 設置當前平台
        self.platform_manager.set_platform(self.current_config["platform"])
        
        # 獲取可用模型
        models = self.platform_manager.get_available_models()
        
        # 獲取推薦配置
        recommended = self.platform_manager.get_recommended_config()
        
        return {
            "step": SetupStep.MODEL_SELECTION.value,
            "title": "選擇 AI 模型",
            "description": "請選擇語言模型和嵌入模型",
            "platform": self.current_config["platform"],
            "models": models,
            "recommended": recommended,
            "current_selection": {
                "language_model": self.current_config.get("language_model"),
                "embedding_model": self.current_config.get("embedding_model")
            },
            "can_proceed": bool(
                self.current_config.get("language_model") and 
                self.current_config.get("embedding_model")
            )
        }
    
    def set_models(self, language_model: str, embedding_model: str, inference_engine: str = "transformers") -> Dict[str, Any]:
        """設置模型選擇"""
        # 驗證模型選擇
        is_valid, message = self.platform_manager.validate_model_selection(
            language_model, embedding_model
        )
        
        # 驗證推理引擎
        if inference_engine not in ["transformers", "vllm"]:
            return {
                "success": False,
                "message": f"不支援的推理引擎: {inference_engine}"
            }
        
        if is_valid:
            self.current_config["language_model"] = language_model
            self.current_config["embedding_model"] = embedding_model
            self.current_config["inference_engine"] = inference_engine
            self.current_config["current_step"] = SetupStep.RAG_MODE_SELECTION.value
            self._save_config()
            
            return {
                "success": True,
                "message": "模型選擇已保存",
                "next_step": SetupStep.RAG_MODE_SELECTION.value
            }
        else:
            return {
                "success": False,
                "message": message
            }
    
    def get_rag_mode_selection_data(self) -> Dict[str, Any]:
        """獲取 RAG 模式選擇數據"""
        if not self.current_config.get("language_model"):
            return {"error": "請先選擇模型"}
        
        rag_modes = [
            {
                "type": RAGMode.TRADITIONAL.value,
                "name": "傳統 RAG",
                "description": "使用預建向量資料庫，快速響應，適合穩定的文檔集合",
                "features": [
                    "快速查詢響應",
                    "穩定的性能",
                    "適合大量文檔",
                    "需要預先索引"
                ],
                "recommended": True,
                "requirements": "需要預先建立向量資料庫"
            },
            {
                "type": RAGMode.DYNAMIC.value,
                "name": "動態 RAG",
                "description": "即時檢索和處理文件，無需預建資料庫，適合動態文檔環境",
                "features": [
                    "無需預先索引",
                    "即時文檔處理",
                    "適合動態內容",
                    "智能文件檢索"
                ],
                "recommended": False,
                "requirements": "需要更多計算資源"
            }
        ]
        
        # 檢查是否有可用的向量資料庫
        has_vector_db = False
        if self.current_config["platform"] == PlatformType.OLLAMA.value:
            try:
                usable_models = vector_db_manager.get_usable_models()
                has_vector_db = len(usable_models) > 0
            except:
                has_vector_db = False
        
        return {
            "step": SetupStep.RAG_MODE_SELECTION.value,
            "title": "選擇 RAG 模式",
            "description": "請選擇檢索增強生成 (RAG) 的工作模式",
            "rag_modes": rag_modes,
            "has_vector_db": has_vector_db,
            "current_selection": self.current_config.get("rag_mode"),
            "can_proceed": bool(self.current_config.get("rag_mode"))
        }
    
    def set_rag_mode(self, rag_mode: str) -> Dict[str, Any]:
        """設置 RAG 模式"""
        try:
            RAGMode(rag_mode)  # 驗證模式有效性
            
            self.current_config["rag_mode"] = rag_mode
            self.current_config["use_dynamic_rag"] = (rag_mode == RAGMode.DYNAMIC.value)
            self.current_config["current_step"] = SetupStep.CONFIGURATION_REVIEW.value
            self._save_config()
            
            return {
                "success": True,
                "message": f"已選擇 RAG 模式: {rag_mode}",
                "next_step": SetupStep.CONFIGURATION_REVIEW.value
            }
        except ValueError:
            return {
                "success": False,
                "message": f"不支援的 RAG 模式: {rag_mode}"
            }
    
    def get_configuration_review_data(self) -> Dict[str, Any]:
        """獲取配置審查數據"""
        config = self.current_config.copy()
        
        # 獲取平台信息
        platforms = self.platform_manager.get_available_platforms()
        platform_info = next(
            (p for p in platforms if p["type"] == config["platform"]), 
            {"name": config["platform"]}
        )
        
        # 獲取模型信息
        self.platform_manager.set_platform(config["platform"])
        models = self.platform_manager.get_available_models()
        
        language_model_info = None
        embedding_model_info = None
        
        for model in models["language_models"]:
            if model["id"] == config["language_model"]:
                language_model_info = model
                break
        
        for model in models["embedding_models"]:
            if model["id"] == config["embedding_model"]:
                embedding_model_info = model
                break
        
        # 估算資源需求
        resource_requirements = self._estimate_resource_requirements(config)
        
        return {
            "step": SetupStep.CONFIGURATION_REVIEW.value,
            "title": "配置確認",
            "description": "請確認您的配置選擇",
            "configuration": {
                "platform": {
                    "type": config["platform"],
                    "name": platform_info["name"],
                    "description": platform_info.get("description", "")
                },
                "language_model": language_model_info,
                "embedding_model": embedding_model_info,
                "rag_mode": {
                    "type": config["rag_mode"],
                    "name": "傳統 RAG" if config["rag_mode"] == RAGMode.TRADITIONAL.value else "動態 RAG",
                    "use_dynamic": config["use_dynamic_rag"]
                }
            },
            "resource_requirements": resource_requirements,
            "can_proceed": True
        }
    
    def _estimate_resource_requirements(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """估算資源需求"""
        requirements = {
            "gpu_memory": "4GB+",
            "system_memory": "8GB+",
            "storage": "10GB+",
            "network": "穩定網路連接"
        }
        
        # 根據語言模型調整需求
        language_model = config.get("language_model", "")
        
        if "20b" in language_model.lower():
            requirements.update({
                "gpu_memory": "40GB+",
                "system_memory": "64GB+",
                "storage": "100GB+",
                "special_notes": "建議使用 A100 或更高級 GPU"
            })
        elif "6b" in language_model.lower() or "7b" in language_model.lower():
            requirements.update({
                "gpu_memory": "12GB+",
                "system_memory": "32GB+",
                "storage": "50GB+"
            })
        elif "xl" in language_model.lower():
            requirements.update({
                "gpu_memory": "6GB+",
                "system_memory": "16GB+",
                "storage": "20GB+"
            })
        
        return requirements
    
    def complete_setup(self) -> Dict[str, Any]:
        """完成設置"""
        try:
            self.current_config["setup_completed"] = True
            self.current_config["current_step"] = SetupStep.READY.value
            self._save_config()
            
            # 應用配置到系統
            self._apply_configuration()
            
            return {
                "success": True,
                "message": "設置完成！系統已準備就緒。",
                "configuration": self.current_config
            }
        except Exception as e:
            logger.error(f"完成設置失敗: {str(e)}")
            return {
                "success": False,
                "message": f"設置失敗: {str(e)}"
            }
    
    def _apply_configuration(self):
        """應用配置到系統"""
        # 這裡可以將配置寫入到環境變數或配置文件
        # 暫時只記錄日誌
        logger.info(f"應用配置: {self.current_config}")
    
    def reset_setup(self) -> Dict[str, Any]:
        """重置設置"""
        self.current_config = {
            "setup_completed": False,
            "current_step": SetupStep.PLATFORM_SELECTION.value,
            "platform": PlatformType.HUGGINGFACE.value,
            "language_model": None,
            "embedding_model": None,
            "rag_mode": RAGMode.TRADITIONAL.value,
            "use_dynamic_rag": False,
            "inference_engine": "transformers",
            "language": "繁體中文"
        }
        self._save_config()
        
        return {
            "success": True,
            "message": "設置已重置"
        }
    
    def get_current_configuration(self) -> Dict[str, Any]:
        """獲取當前配置"""
        return self.current_config.copy()

# 全局設置流程管理器實例
setup_flow_manager = SetupFlowManager()

def get_setup_flow_manager() -> SetupFlowManager:
    """獲取全局設置流程管理器實例"""
    return setup_flow_manager