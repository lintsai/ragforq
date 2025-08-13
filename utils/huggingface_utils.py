#!/usr/bin/env python
"""
Hugging Face 本地模型工具函數
用於檢測本地已下載的模型和管理模型相關操作
"""
import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from config.config import HF_MODEL_CACHE_DIR

logger = logging.getLogger(__name__)

class HuggingFaceUtils:
    """Hugging Face 本地模型工具類"""
    
    def __init__(self, cache_dir: str = HF_MODEL_CACHE_DIR):
        """
        初始化 Hugging Face 工具
        
        Args:
            cache_dir: 模型緩存目錄
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def get_local_models(self) -> List[Dict[str, Any]]:
        """
        獲取本地已下載的模型列表
        
        Returns:
            模型列表，每個模型包含 name, path, size, type 等信息
        """
        models = []
        
        try:
            # 檢查 Hugging Face 緩存目錄結構
            # 通常結構為: models--organization--model-name/snapshots/hash/
            if not self.cache_dir.exists():
                logger.warning(f"緩存目錄不存在: {self.cache_dir}")
                return models
            
            # 遍歷緩存目錄
            for model_dir in self.cache_dir.iterdir():
                if model_dir.is_dir() and model_dir.name.startswith('models--'):
                    model_info = self._parse_model_directory(model_dir)
                    if model_info:
                        models.append(model_info)
            
            # 也檢查直接存放的模型（如果有的話）
            for item in self.cache_dir.iterdir():
                if item.is_dir() and not item.name.startswith('models--'):
                    # 檢查是否是直接存放的模型目錄
                    if self._is_model_directory(item):
                        model_info = self._parse_direct_model_directory(item)
                        if model_info:
                            models.append(model_info)
            
            logger.info(f"找到 {len(models)} 個本地模型")
            return models
            
        except Exception as e:
            logger.error(f"掃描本地模型時出錯: {str(e)}")
            return []
    
    def _parse_model_directory(self, model_dir: Path) -> Optional[Dict[str, Any]]:
        """解析 Hugging Face 緩存目錄結構的模型"""
        try:
            # 從目錄名解析模型名稱
            # models--organization--model-name -> organization/model-name
            dir_name = model_dir.name
            if not dir_name.startswith('models--'):
                return None
            
            parts = dir_name.replace('models--', '').split('--')
            if len(parts) < 2:
                return None
            
            model_name = '/'.join(parts)
            
            # 查找 snapshots 目錄
            snapshots_dir = model_dir / 'snapshots'
            if not snapshots_dir.exists():
                return None
            
            # 找到最新的快照
            snapshot_dirs = [d for d in snapshots_dir.iterdir() if d.is_dir()]
            if not snapshot_dirs:
                return None
            
            # 使用最新修改的快照
            latest_snapshot = max(snapshot_dirs, key=lambda x: x.stat().st_mtime)
            
            # 計算模型大小
            total_size = self._calculate_directory_size(latest_snapshot)
            
            # 檢測模型類型
            model_type = self._detect_model_type(latest_snapshot, model_name)
            
            return {
                'name': model_name,
                'path': str(latest_snapshot),
                'cache_path': str(model_dir),
                'size': total_size,
                'size_formatted': self._format_size(total_size),
                'type': model_type,
                'modified': latest_snapshot.stat().st_mtime,
                'available': True
            }
            
        except Exception as e:
            logger.error(f"解析模型目錄 {model_dir} 時出錯: {str(e)}")
            return None
    
    def _parse_direct_model_directory(self, model_dir: Path) -> Optional[Dict[str, Any]]:
        """解析直接存放的模型目錄"""
        try:
            model_name = model_dir.name
            total_size = self._calculate_directory_size(model_dir)
            model_type = self._detect_model_type(model_dir, model_name)
            
            return {
                'name': model_name,
                'path': str(model_dir),
                'cache_path': str(model_dir),
                'size': total_size,
                'size_formatted': self._format_size(total_size),
                'type': model_type,
                'modified': model_dir.stat().st_mtime,
                'available': True
            }
            
        except Exception as e:
            logger.error(f"解析直接模型目錄 {model_dir} 時出錯: {str(e)}")
            return None
    
    def _is_model_directory(self, path: Path) -> bool:
        """檢查目錄是否是模型目錄"""
        # 檢查是否包含模型文件
        model_files = [
            'config.json', 'pytorch_model.bin', 'model.safetensors',
            'tokenizer.json', 'tokenizer_config.json'
        ]
        
        for file_name in model_files:
            if (path / file_name).exists():
                return True
        
        return False
    
    def _detect_model_type(self, model_path: Path, model_name: str) -> str:
        """檢測模型類型"""
        model_name_lower = model_name.lower()
        
        # 檢查是否是嵌入模型
        embedding_indicators = [
            'sentence-transformers', 'embedding', 'embed', 'retrieval',
            'mpnet', 'minilm', 'distilbert', 'roberta'
        ]
        
        if any(indicator in model_name_lower for indicator in embedding_indicators):
            return 'embedding'
        
        # 檢查配置文件來確定模型類型
        config_file = model_path / 'config.json'
        if config_file.exists():
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                # 根據架構判斷
                architectures = config.get('architectures', [])
                if architectures:
                    arch = architectures[0].lower()
                    if 'embedding' in arch or 'sentence' in arch:
                        return 'embedding'
                    elif any(lang_arch in arch for lang_arch in ['gpt', 'llama', 'qwen', 'bert']):
                        return 'language'
                
            except Exception as e:
                logger.debug(f"讀取配置文件失敗: {e}")
        
        # 默認判斷為語言模型
        return 'language'
    
    def _calculate_directory_size(self, path: Path) -> int:
        """計算目錄大小"""
        total_size = 0
        try:
            for file_path in path.rglob('*'):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
        except Exception as e:
            logger.debug(f"計算目錄大小時出錯: {e}")
        
        return total_size
    
    def _format_size(self, size_bytes: int) -> str:
        """格式化文件大小"""
        if size_bytes == 0:
            return "0B"
        
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f}{unit}"
            size_bytes /= 1024.0
        
        return f"{size_bytes:.1f}PB"
    
    def get_language_models(self) -> List[Dict[str, Any]]:
        """獲取本地語言模型列表"""
        all_models = self.get_local_models()
        return [model for model in all_models if model['type'] == 'language']
    
    def get_embedding_models(self) -> List[Dict[str, Any]]:
        """獲取本地嵌入模型列表"""
        all_models = self.get_local_models()
        return [model for model in all_models if model['type'] == 'embedding']
    
    def get_model_names(self) -> List[str]:
        """獲取所有本地模型名稱列表"""
        models = self.get_local_models()
        return [model['name'] for model in models]
    
    def is_model_available(self, model_name: str) -> bool:
        """檢查指定模型是否在本地可用"""
        available_models = self.get_model_names()
        return model_name in available_models
    
    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """獲取特定模型的詳細信息"""
        models = self.get_local_models()
        for model in models:
            if model['name'] == model_name:
                return model
        return None

# 全局實例
huggingface_utils = HuggingFaceUtils()