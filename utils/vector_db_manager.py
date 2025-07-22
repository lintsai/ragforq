#!/usr/bin/env python
"""
向量數據庫管理工具
管理不同模型的向量數據庫文件夾和相關操作
"""
import os
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from config.config import VECTOR_DB_PATH

logger = logging.getLogger(__name__)

class VectorDBManager:
    """向量數據庫管理器"""
    
    def __init__(self, base_path: str = VECTOR_DB_PATH):
        """
        初始化向量數據庫管理器
        
        Args:
            base_path: 向量數據庫基礎路徑
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def get_model_folder_name(self, ollama_model: str, ollama_embedding_model: str, version: str = None) -> str:
        """
        根據模型名稱生成文件夾名稱
        使用 '@' 作為主要分隔符，'#' 作為版本分隔符
        
        Args:
            ollama_model: Ollama 語言模型名稱
            ollama_embedding_model: Ollama 嵌入模型名稱
            version: 版本標識（可選），如日期 "20250722"
            
        Returns:
            文件夾名稱
        """
        # 清理模型名稱中的特殊字符，使用簡潔的替換
        clean_model = ollama_model.replace(':', '_').replace('/', '_').replace('\\', '_')
        clean_embedding = ollama_embedding_model.replace(':', '_').replace('/', '_').replace('\\', '_')
        
        # 基礎名稱
        base_name = f"ollama@{clean_model}@{clean_embedding}"
        
        # 如果有版本標識，添加版本後綴
        if version:
            return f"{base_name}#{version}"
        else:
            return base_name
    
    def parse_folder_name(self, folder_name: str) -> Optional[Tuple[str, str, str]]:
        """
        從資料夾名稱解析出模型名稱和版本
        注意：由於 ':' 被替換為 '_'，解析時無法完全恢復原始名稱
        因此建議依賴 .model 文件獲取準確的模型信息
        
        Args:
            folder_name: 資料夾名稱
            
        Returns:
            (ollama_model, ollama_embedding_model, version) 或 None
        """
        if folder_name.startswith('ollama@'):
            # 格式: ollama@model@embedding 或 ollama@model@embedding#version
            # 先分離版本
            if '#' in folder_name:
                base_name, version = folder_name.rsplit('#', 1)
            else:
                base_name, version = folder_name, "current"  # 無版本視為當前版本
            
            parts = base_name.split('@')
            if len(parts) == 3:
                # 注意：這裡無法準確恢復 ':' 字符，因為 '_' 可能是原始名稱的一部分
                # 建議使用 .model 文件獲取準確信息
                model = parts[1]
                embedding = parts[2]
                return (model, embedding, version)
        
        return None
    
    def get_model_path(self, ollama_model: str, ollama_embedding_model: str) -> Path:
        """
        獲取指定模型的向量數據庫路徑
        
        Args:
            ollama_model: Ollama 語言模型名稱
            ollama_embedding_model: Ollama 嵌入模型名稱
            
        Returns:
            向量數據庫路徑
        """
        folder_name = self.get_model_folder_name(ollama_model, ollama_embedding_model)
        return self.base_path / folder_name
    
    def create_model_folder(self, ollama_model: str, ollama_embedding_model: str) -> Path:
        """
        創建模型文件夾
        
        Args:
            ollama_model: Ollama 語言模型名稱
            ollama_embedding_model: Ollama 嵌入模型名稱
            
        Returns:
            創建的文件夾路徑
        """
        model_path = self.get_model_path(ollama_model, ollama_embedding_model)
        model_path.mkdir(parents=True, exist_ok=True)
        
        # 創建 .model 文件記錄模型信息
        import datetime
        model_info = {
            "OLLAMA_MODEL": ollama_model,
            "OLLAMA_EMBEDDING_MODEL": ollama_embedding_model,
            "created_at": datetime.datetime.now().isoformat()
        }
        
        model_file_path = model_path / ".model"
        with open(model_file_path, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, ensure_ascii=False, indent=2)
        
        logger.info(f"創建模型文件夾: {model_path}")
        return model_path
    
    def get_model_info(self, model_path: Path) -> Optional[Dict[str, str]]:
        """
        從 .model 文件讀取模型信息
        
        Args:
            model_path: 模型文件夾路徑
            
        Returns:
            模型信息字典，如果文件不存在則返回 None
        """
        model_file_path = model_path / ".model"
        
        if not model_file_path.exists():
            return None
        
        try:
            with open(model_file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"讀取模型信息文件失敗 {model_file_path}: {str(e)}")
            return None
    
    def list_available_models(self) -> List[Dict[str, Any]]:
        """
        列出所有可用的向量數據庫模型
        
        Returns:
            模型列表，每個模型包含文件夾名稱、模型信息、是否正在訓練等
        """
        models = []
        
        if not self.base_path.exists():
            return models
        
        for folder in self.base_path.iterdir():
            if folder.is_dir() and folder.name.startswith('ollama@'):
                model_info = self.get_model_info(folder)
                is_training = self.is_training(folder)
                has_data = self.has_vector_data(folder)
                
                # 解析版本信息
                parsed = self.parse_folder_name(folder.name)
                version = parsed[2] if parsed else None
                
                models.append({
                    'folder_name': folder.name,
                    'folder_path': str(folder),
                    'version': version,
                    'model_info': model_info,
                    'is_training': is_training,
                    'has_data': has_data,
                    'display_name': self._get_display_name(folder.name, model_info)
                })
        
        return models
    
    def _get_display_name(self, folder_name: str, model_info: Optional[Dict[str, str]]) -> str:
        """
        生成顯示名稱
        
        Args:
            folder_name: 文件夾名稱
            model_info: 模型信息
            
        Returns:
            顯示名稱
        """
        # 優先使用 .model 文件中的信息
        if model_info:
            ollama_model = model_info.get('OLLAMA_MODEL', '')
            ollama_embedding = model_info.get('OLLAMA_EMBEDDING_MODEL', '')
            version = model_info.get('version')
            base_name = f"{ollama_model} + {ollama_embedding}"
            if version:
                return f"{base_name} ({version})"
            return base_name
        
        # 嘗試從資料夾名稱解析
        parsed = self.parse_folder_name(folder_name)
        if parsed:
            ollama_model, ollama_embedding, version = parsed
            base_name = f"{ollama_model} + {ollama_embedding}"
            if version and version != "current":
                return f"{base_name} ({version})"
            elif version == "current":
                return f"{base_name} (當前版本)"
            return base_name
        
        # 最後使用資料夾名稱
        return folder_name
    
    def is_training(self, model_path: Path) -> bool:
        """
        檢查模型是否正在訓練（是否存在 .lock 文件）
        
        Args:
            model_path: 模型文件夾路徑
            
        Returns:
            是否正在訓練
        """
        lock_file = model_path / ".lock"
        return lock_file.exists()
    
    def has_vector_data(self, model_path: Path) -> bool:
        """
        檢查模型是否有向量數據
        
        Args:
            model_path: 模型文件夾路徑
            
        Returns:
            是否有向量數據
        """
        index_faiss = model_path / "index.faiss"
        index_pkl = model_path / "index.pkl"
        return index_faiss.exists() and index_pkl.exists()
    
    def create_lock_file(self, model_path: Path) -> None:
        """
        創建訓練鎖定文件
        
        Args:
            model_path: 模型文件夾路徑
        """
        lock_file = model_path / ".lock"
        lock_file.touch()
        logger.info(f"創建訓練鎖定文件: {lock_file}")
    
    def remove_lock_file(self, model_path: Path) -> None:
        """
        移除訓練鎖定文件
        
        Args:
            model_path: 模型文件夾路徑
        """
        lock_file = model_path / ".lock"
        if lock_file.exists():
            lock_file.unlink()
            logger.info(f"移除訓練鎖定文件: {lock_file}")
    
    def get_usable_models(self) -> List[Dict[str, Any]]:
        """
        獲取可用於問答的模型列表（有數據且未在訓練中）
        對於同一模型組合的多個版本，優先返回最新的可用版本
        
        Returns:
            可用模型列表
        """
        all_models = self.list_available_models()
        usable_models = []
        model_groups = {}  # 按模型組合分組
        
        # 按模型組合分組
        for model in all_models:
            if model['has_data'] and not model['is_training']:
                model_info = model.get('model_info')
                if model_info:
                    # 使用模型組合作為鍵
                    key = f"{model_info.get('OLLAMA_MODEL', '')}@{model_info.get('OLLAMA_EMBEDDING_MODEL', '')}"
                    if key not in model_groups:
                        model_groups[key] = []
                    model_groups[key].append(model)
        
        # 為每個模型組合選擇最佳版本
        for group_models in model_groups.values():
            if len(group_models) == 1:
                usable_models.append(group_models[0])
            else:
                # 按版本排序，優先選擇最新版本
                # 版本排序規則：有具體版本號的優先於 "current"，版本號按字典序倒序
                def version_sort_key(model):
                    version = model.get('version', 'current')
                    if version == 'current':
                        return ('0', '')  # current 版本排在最後
                    else:
                        return ('1', version)  # 具體版本按版本號倒序
                
                sorted_models = sorted(group_models, key=version_sort_key, reverse=True)
                usable_models.append(sorted_models[0])  # 選擇最新版本
        
        return usable_models
    
    def create_versioned_model_folder(self, ollama_model: str, ollama_embedding_model: str, version: str) -> Path:
        """
        創建帶版本的模型文件夾
        
        Args:
            ollama_model: Ollama 語言模型名稱
            ollama_embedding_model: Ollama 嵌入模型名稱
            version: 版本標識，如日期 "20250722"
            
        Returns:
            創建的文件夾路徑
        """
        folder_name = self.get_model_folder_name(ollama_model, ollama_embedding_model, version)
        model_path = self.base_path / folder_name
        model_path.mkdir(parents=True, exist_ok=True)
        
        # 創建 .model 文件記錄模型信息
        import datetime
        model_info = {
            "OLLAMA_MODEL": ollama_model,
            "OLLAMA_EMBEDDING_MODEL": ollama_embedding_model,
            "version": version,
            "created_at": datetime.datetime.now().isoformat()
        }
        
        model_file_path = model_path / ".model"
        with open(model_file_path, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, ensure_ascii=False, indent=2)
        
        logger.info(f"創建版本化模型文件夾: {model_path}")
        return model_path
    
    def get_model_versions(self, ollama_model: str, ollama_embedding_model: str) -> List[Dict[str, Any]]:
        """
        獲取指定模型組合的所有版本
        
        Args:
            ollama_model: Ollama 語言模型名稱
            ollama_embedding_model: Ollama 嵌入模型名稱
            
        Returns:
            版本列表，每個版本包含版本信息和狀態
        """
        base_folder_name = self.get_model_folder_name(ollama_model, ollama_embedding_model)
        versions = []
        
        if not self.base_path.exists():
            return versions
        
        for folder in self.base_path.iterdir():
            if folder.is_dir():
                # 檢查是否是同一模型組合的版本
                if folder.name == base_folder_name or folder.name.startswith(f"{base_folder_name}#"):
                    model_info = self.get_model_info(folder)
                    is_training = self.is_training(folder)
                    has_data = self.has_vector_data(folder)
                    
                    # 解析版本信息
                    parsed = self.parse_folder_name(folder.name)
                    version = parsed[2] if parsed else None
                    
                    versions.append({
                        'folder_name': folder.name,
                        'folder_path': str(folder),
                        'version': version,
                        'model_info': model_info,
                        'is_training': is_training,
                        'has_data': has_data,
                        'display_name': self._get_display_name(folder.name, model_info)
                    })
        
        # 按版本排序（最新的在前）
        versions.sort(key=lambda x: x['version'] or '', reverse=True)
        return versions
    
    def get_latest_model_version(self, ollama_model: str, ollama_embedding_model: str) -> Optional[Dict[str, Any]]:
        """
        獲取指定模型組合的最新版本
        
        Args:
            ollama_model: Ollama 語言模型名稱
            ollama_embedding_model: Ollama 嵌入模型名稱
            
        Returns:
            最新版本的模型信息，如果不存在則返回 None
        """
        versions = self.get_model_versions(ollama_model, ollama_embedding_model)
        
        # 優先返回有數據且未在訓練的版本
        for version in versions:
            if version['has_data'] and not version['is_training']:
                return version
        
        # 如果沒有可用版本，返回最新的版本
        return versions[0] if versions else None

# 全局實例
vector_db_manager = VectorDBManager()