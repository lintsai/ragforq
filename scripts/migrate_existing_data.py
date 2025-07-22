#!/usr/bin/env python
"""
遷移現有向量數據到新的模型管理結構
"""
import os
import sys
import json
import shutil
import logging
from pathlib import Path

# 添加項目根目錄到路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import VECTOR_DB_PATH, OLLAMA_MODEL, OLLAMA_EMBEDDING_MODEL
from utils.vector_db_manager import vector_db_manager

logger = logging.getLogger(__name__)

def migrate_existing_data():
    """遷移現有的向量數據到新結構"""
    vector_db_path = Path(VECTOR_DB_PATH)
    
    # 檢查是否有舊的向量數據文件
    old_index_faiss = vector_db_path / "index.faiss"
    old_index_pkl = vector_db_path / "index.pkl"
    old_indexed_files = vector_db_path / "indexed_files.pkl"
    old_progress = vector_db_path / "indexing_progress.json"
    
    if not (old_index_faiss.exists() and old_index_pkl.exists()):
        logger.info("沒有找到舊的向量數據，無需遷移")
        return
    
    logger.info("發現舊的向量數據，開始遷移...")
    
    # 創建新的模型文件夾
    new_model_path = vector_db_manager.create_model_folder(OLLAMA_MODEL, OLLAMA_EMBEDDING_MODEL)
    
    # 移動文件
    files_to_move = [
        ("index.faiss", old_index_faiss),
        ("index.pkl", old_index_pkl),
        ("indexed_files.pkl", old_indexed_files),
        ("indexing_progress.json", old_progress)
    ]
    
    moved_files = []
    
    for filename, old_path in files_to_move:
        if old_path.exists():
            new_path = new_model_path / filename
            try:
                shutil.move(str(old_path), str(new_path))
                moved_files.append(filename)
                logger.info(f"移動文件: {filename}")
            except Exception as e:
                logger.error(f"移動文件 {filename} 失敗: {str(e)}")
    
    if moved_files:
        logger.info(f"遷移完成，已移動 {len(moved_files)} 個文件到 {new_model_path}")
        logger.info("舊的向量數據已遷移到新的模型管理結構")
    else:
        logger.warning("沒有文件被移動")

def main():
    """主函數"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("開始遷移現有向量數據...")
    migrate_existing_data()
    logger.info("遷移完成")

if __name__ == "__main__":
    main()