#!/usr/bin/env python
"""
訓練鎖定管理器
管理訓練過程中的鎖定狀態，支持檢測無效鎖定和手動解鎖
"""
import os
import json
import time
import psutil
import logging
from typing import Dict, Optional, Tuple
from pathlib import Path
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class TrainingLockManager:
    """訓練鎖定管理器"""
    
    def __init__(self):
        self.lock_timeout = 24 * 60 * 60  # 24小時超時
    
    def create_lock(self, model_path: Path, process_info: Optional[Dict] = None) -> None:
        """
        創建訓練鎖定文件，包含進程信息
        
        Args:
            model_path: 模型文件夾路徑
            process_info: 進程信息（可選）
        """
        lock_file = model_path / ".lock"
        lock_data = {
            "created_at": datetime.now().isoformat(),
            "pid": os.getpid(),
            "process_name": psutil.Process().name(),
            "hostname": os.uname().nodename if hasattr(os, 'uname') else 'unknown'
        }
        
        if process_info:
            lock_data.update(process_info)
        
        with open(lock_file, 'w', encoding='utf-8') as f:
            json.dump(lock_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"創建訓練鎖定: {lock_file}")
    
    def remove_lock(self, model_path: Path) -> bool:
        """
        移除訓練鎖定文件
        
        Args:
            model_path: 模型文件夾路徑
            
        Returns:
            是否成功移除
        """
        lock_file = model_path / ".lock"
        try:
            if lock_file.exists():
                lock_file.unlink()
                logger.info(f"移除訓練鎖定: {lock_file}")
                return True
            return False
        except Exception as e:
            logger.error(f"移除鎖定文件失敗 {lock_file}: {str(e)}")
            return False
    
    def is_locked(self, model_path: Path) -> bool:
        """
        檢查模型是否被鎖定
        
        Args:
            model_path: 模型文件夾路徑
            
        Returns:
            是否被鎖定
        """
        lock_file = model_path / ".lock"
        return lock_file.exists()
    
    def get_lock_info(self, model_path: Path) -> Optional[Dict]:
        """
        獲取鎖定信息
        
        Args:
            model_path: 模型文件夾路徑
            
        Returns:
            鎖定信息字典，如果沒有鎖定則返回 None
        """
        lock_file = model_path / ".lock"
        
        if not lock_file.exists():
            return None
        
        try:
            with open(lock_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"讀取鎖定信息失敗 {lock_file}: {str(e)}")
            return {"error": str(e)}
    
    def is_process_running(self, pid: int) -> bool:
        """
        檢查進程是否還在運行
        
        Args:
            pid: 進程ID
            
        Returns:
            進程是否在運行
        """
        try:
            return psutil.pid_exists(pid)
        except Exception:
            return False
    
    def is_lock_valid(self, model_path: Path) -> Tuple[bool, str]:
        """
        檢查鎖定是否有效
        
        Args:
            model_path: 模型文件夾路徑
            
        Returns:
            (是否有效, 原因描述)
        """
        lock_info = self.get_lock_info(model_path)
        
        if not lock_info:
            return True, "沒有鎖定"
        
        if "error" in lock_info:
            return False, f"鎖定文件損壞: {lock_info['error']}"
        
        # 檢查進程是否還在運行
        pid = lock_info.get('pid')
        if pid and not self.is_process_running(pid):
            return False, f"訓練進程 (PID: {pid}) 已停止"
        
        # 檢查鎖定時間是否超時
        created_at_str = lock_info.get('created_at')
        if created_at_str:
            try:
                created_at = datetime.fromisoformat(created_at_str)
                if datetime.now() - created_at > timedelta(seconds=self.lock_timeout):
                    return False, f"鎖定已超時 (超過 {self.lock_timeout/3600:.1f} 小時)"
            except Exception as e:
                logger.warning(f"解析鎖定時間失敗: {str(e)}")
        
        return True, "鎖定有效"
    
    def cleanup_invalid_locks(self, base_path: Path) -> Dict[str, str]:
        """
        清理所有無效的鎖定文件
        
        Args:
            base_path: 向量資料庫基礎路徑
            
        Returns:
            清理結果字典 {model_name: result}
        """
        results = {}
        
        if not base_path.exists():
            return results
        
        for model_folder in base_path.iterdir():
            if model_folder.is_dir() and model_folder.name.startswith('ollama@'):
                is_valid, reason = self.is_lock_valid(model_folder)
                
                if not is_valid:
                    if self.remove_lock(model_folder):
                        results[model_folder.name] = f"已清理無效鎖定: {reason}"
                    else:
                        results[model_folder.name] = f"清理失敗: {reason}"
                elif self.is_locked(model_folder):
                    results[model_folder.name] = f"鎖定有效: {reason}"
        
        return results
    
    def force_unlock(self, model_path: Path, reason: str = "管理員手動解鎖") -> bool:
        """
        強制解鎖模型
        
        Args:
            model_path: 模型文件夾路徑
            reason: 解鎖原因
            
        Returns:
            是否成功解鎖
        """
        if not self.is_locked(model_path):
            return True
        
        # 記錄解鎖操作
        lock_info = self.get_lock_info(model_path)
        logger.warning(f"強制解鎖模型 {model_path.name}: {reason}")
        logger.warning(f"原鎖定信息: {lock_info}")
        
        return self.remove_lock(model_path)

# 全局實例
training_lock_manager = TrainingLockManager()