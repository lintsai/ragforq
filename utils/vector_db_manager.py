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
    
    def create_lock_file(self, model_path: Path, process_info: dict = None) -> None:
        """
        創建訓練鎖定文件
        
        Args:
            model_path: 模型文件夾路徑
            process_info: 進程信息（可選）
        """
        from utils.training_lock_manager import training_lock_manager
        training_lock_manager.create_lock(model_path, process_info)
    
    def remove_lock_file(self, model_path: Path) -> None:
        """
        移除訓練鎖定文件
        
        Args:
            model_path: 模型文件夾路徑
        """
        from utils.training_lock_manager import training_lock_manager
        training_lock_manager.remove_lock(model_path)
    
    def is_lock_valid(self, model_path: Path) -> tuple:
        """
        檢查鎖定是否有效
        
        Args:
            model_path: 模型文件夾路徑
            
        Returns:
            (是否有效, 原因描述)
        """
        from utils.training_lock_manager import training_lock_manager
        return training_lock_manager.is_lock_valid(model_path)
    
    def get_lock_info(self, model_path: Path) -> dict:
        """
        獲取鎖定詳細信息
        
        Args:
            model_path: 模型文件夾路徑
            
        Returns:
            鎖定信息字典
        """
        from utils.training_lock_manager import training_lock_manager
        return training_lock_manager.get_lock_info(model_path)
    
    def force_unlock_model(self, model_path: Path, reason: str = "管理員手動解鎖") -> bool:
        """
        強制解鎖模型
        
        Args:
            model_path: 模型文件夾路徑
            reason: 解鎖原因
            
        Returns:
            是否成功解鎖
        """
        from utils.training_lock_manager import training_lock_manager
        return training_lock_manager.force_unlock(model_path, reason)
    
    def get_usable_models(self) -> List[Dict[str, Any]]:
        """
        獲取可用於問答的模型列表（有數據且未在訓練中，或訓練鎖定無效）
        按時間降冪排序，最新訓練出來的為預設
        
        Returns:
            可用模型列表，按創建時間降冪排序
        """
        all_models = self.list_available_models()
        usable_models = []
        
        # 篩選可用模型
        for model in all_models:
            if model['has_data']:
                # 檢查是否真的不可用
                can_use = True
                if model['is_training']:
                    # 檢查鎖定是否有效
                    model_path = Path(model['folder_path'])
                    is_valid, reason = self.is_lock_valid(model_path)
                    if is_valid:
                        can_use = False
                        logger.debug(f"模型 {model['display_name']} 正在有效訓練中: {reason}")
                    else:
                        logger.info(f"模型 {model['display_name']} 鎖定無效，自動清理: {reason}")
                        # 自動清理無效鎖定
                        try:
                            self.remove_lock_file(model_path)
                            model['is_training'] = False  # 更新狀態
                            logger.info(f"已清理模型 {model['display_name']} 的無效鎖定")
                        except Exception as e:
                            logger.error(f"清理無效鎖定失敗: {str(e)}")
                            can_use = False  # 如果清理失敗，仍然不可用
                
                if can_use:
                    usable_models.append(model)
        
        # 按時間降冪排序，當前版本優先，然後是最新的
        def get_sort_key(model):
            """獲取排序鍵，當前版本優先，然後按時間降冪排序"""
            model_info = model.get('model_info', {})
            version = model.get('version')
            
            # 1. 當前版本 (current) 優先級最高，使用一個很大的時間戳
            if version == 'current':
                import time
                return time.time() + 999999999  # 確保當前版本排在最前面
            
            # 2. 優先使用 .model 文件中的 created_at 時間
            if 'created_at' in model_info:
                try:
                    from datetime import datetime
                    created_time = datetime.fromisoformat(model_info['created_at'])
                    return created_time.timestamp()
                except:
                    pass
            
            # 3. 其次使用版本號（如果是日期格式）
            if version and version != 'current':
                try:
                    # 嘗試將版本號解析為日期（格式如 20250131）
                    if len(version) == 8 and version.isdigit():
                        from datetime import datetime
                        version_date = datetime.strptime(version, '%Y%m%d')
                        return version_date.timestamp()
                except:
                    pass
            
            # 4. 最後使用文件夾的修改時間
            try:
                folder_path = Path(model['folder_path'])
                if folder_path.exists():
                    return folder_path.stat().st_mtime
            except:
                pass
            
            # 5. 如果都失敗，返回 0（最舊）
            return 0
        
        # 按時間降冪排序（最新的在前）
        usable_models.sort(key=get_sort_key, reverse=True)
        
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