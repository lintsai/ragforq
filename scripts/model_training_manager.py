#!/usr/bin/env python
"""
模型訓練管理器
管理不同模型的訓練過程，包括初始訓練、增量訓練和重新索引
"""
import os
import sys
import logging
import json
import argparse
from pathlib import Path

# 添加項目根目錄到路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.vector_db_manager import vector_db_manager
from utils.ollama_utils import ollama_utils
from utils.log_manager import setup_model_logger
from indexer.document_indexer import DocumentIndexer
from indexer.file_crawler import FileCrawler
from config.config import Q_DRIVE_PATH, get_supported_file_extensions

logger = logging.getLogger(__name__)

class ModelTrainingManager:
    """模型訓練管理器"""
    
    def __init__(self):
        self.vector_db_manager = vector_db_manager
        self.ollama_utils = ollama_utils
    
    def validate_models(self, ollama_model: str, ollama_embedding_model: str) -> bool:
        """
        驗證模型是否可用
        
        Args:
            ollama_model: Ollama 語言模型名稱
            ollama_embedding_model: Ollama 嵌入模型名稱
            
        Returns:
            模型是否都可用
        """
        available_models = self.ollama_utils.get_model_names()
        
        if ollama_model not in available_models:
            logger.error(f"語言模型 '{ollama_model}' 不可用")
            logger.info(f"可用模型: {available_models}")
            return False
        
        if ollama_embedding_model not in available_models:
            logger.error(f"嵌入模型 '{ollama_embedding_model}' 不可用")
            logger.info(f"可用模型: {available_models}")
            return False
        
        return True
    
    def start_initial_training(self, ollama_model: str, ollama_embedding_model: str, version: str = None) -> bool:
        """
        開始初始訓練
        
        Args:
            ollama_model: Ollama 語言模型名稱
            ollama_embedding_model: Ollama 嵌入模型名稱
            version: 版本標識（可選），如日期 "20250722"
            
        Returns:
            是否成功開始訓練
        """
        # 驗證模型
        if not self.validate_models(ollama_model, ollama_embedding_model):
            return False
        
        # 創建模型文件夾
        if version:
            model_path = self.vector_db_manager.create_versioned_model_folder(ollama_model, ollama_embedding_model, version)
        else:
            model_path = self.vector_db_manager.create_model_folder(ollama_model, ollama_embedding_model)
        
        # 檢查是否已有數據
        if self.vector_db_manager.has_vector_data(model_path):
            logger.warning(f"模型 {ollama_model}+{ollama_embedding_model} 已有向量數據，請使用增量訓練或重新索引")
            return False
        
        # 創建鎖定文件，包含進程信息
        process_info = {
            "training_type": "initial",
            "model_combination": f"{ollama_model}+{ollama_embedding_model}",
            "version": version
        }
        self.vector_db_manager.create_lock_file(model_path, process_info)
        
        try:
            logger.info(f"開始初始訓練: {ollama_model} + {ollama_embedding_model}")
            
            # 創建文檔索引器
            document_indexer = DocumentIndexer(
                vector_db_path=str(model_path),
                ollama_embedding_model=ollama_embedding_model
            )
            
            # 獲取所有文件
            file_crawler = FileCrawler(Q_DRIVE_PATH)
            all_files = list(file_crawler.crawl())
            
            logger.info(f"找到 {len(all_files)} 個文件需要索引")
            
            # 開始索引
            success_count, fail_count = document_indexer.index_files(all_files, show_progress=True)
            
            # 保存向量存儲
            document_indexer._save_vector_store()
            
            logger.info(f"初始訓練完成: 成功 {success_count} 個，失敗 {fail_count} 個")
            
            return True
            
        except Exception as e:
            logger.error(f"初始訓練失敗: {str(e)}")
            return False
        finally:
            # 移除鎖定文件
            self.vector_db_manager.remove_lock_file(model_path)
    
    def start_incremental_training(self, ollama_model: str, ollama_embedding_model: str, version: str = None) -> bool:
        """
        開始增量訓練
        
        Args:
            ollama_model: Ollama 語言模型名稱
            ollama_embedding_model: Ollama 嵌入模型名稱
            version: 版本標識（可選），如日期 "20250722"
            
        Returns:
            是否成功開始訓練
        """
        # 驗證模型
        if not self.validate_models(ollama_model, ollama_embedding_model):
            return False
        
        if version:
            folder_name = self.vector_db_manager.get_model_folder_name(ollama_model, ollama_embedding_model, version)
            model_path = self.vector_db_manager.base_path / folder_name
        else:
            model_path = self.vector_db_manager.get_model_path(ollama_model, ollama_embedding_model)
        
        # 檢查是否存在向量數據
        if not self.vector_db_manager.has_vector_data(model_path):
            logger.error(f"模型 {ollama_model}+{ollama_embedding_model} 沒有向量數據，請先進行初始訓練")
            return False
        
        # 創建鎖定文件，包含進程信息
        process_info = {
            "training_type": "incremental",
            "model_combination": f"{ollama_model}+{ollama_embedding_model}",
            "version": version
        }
        self.vector_db_manager.create_lock_file(model_path, process_info)
        
        try:
            logger.info(f"開始增量訓練: {ollama_model} + {ollama_embedding_model}")
            
            # 創建文檔索引器
            document_indexer = DocumentIndexer(
                vector_db_path=str(model_path),
                ollama_embedding_model=ollama_embedding_model
            )
            
            # 獲取所有文件
            file_crawler = FileCrawler(Q_DRIVE_PATH)
            all_files = list(file_crawler.crawl())
            
            # 檢查文件變更
            new_files, updated_files, deleted_files = file_crawler.get_file_changes(
                set(document_indexer.indexed_files.keys()) if hasattr(document_indexer, 'indexed_files') else set()
            )
            
            total_changes = len(new_files) + len(updated_files) + len(deleted_files)
            logger.info(f"檢測到 {total_changes} 個文件變更: 新增 {len(new_files)}, 更新 {len(updated_files)}, 刪除 {len(deleted_files)}")
            
            if total_changes == 0:
                logger.info("沒有文件變更，無需增量訓練")
                return True
            
            # 處理文件變更
            document_indexer.process_file_changes(new_files, updated_files, deleted_files)
            
            # 保存向量存儲
            document_indexer._save_vector_store()
            
            logger.info("增量訓練完成")
            
            return True
            
        except Exception as e:
            logger.error(f"增量訓練失敗: {str(e)}")
            return False
        finally:
            # 移除鎖定文件
            self.vector_db_manager.remove_lock_file(model_path)
    
    def start_reindex(self, ollama_model: str, ollama_embedding_model: str, version: str = None) -> bool:
        """
        開始重新索引
        
        Args:
            ollama_model: Ollama 語言模型名稱
            ollama_embedding_model: Ollama 嵌入模型名稱
            version: 版本標識（可選），如日期 "20250722"
            
        Returns:
            是否成功開始重新索引
        """
        # 驗證模型
        if not self.validate_models(ollama_model, ollama_embedding_model):
            return False
        
        model_path = self.vector_db_manager.get_model_path(ollama_model, ollama_embedding_model)
        
        # 創建鎖定文件，包含進程信息
        process_info = {
            "training_type": "reindex",
            "model_combination": f"{ollama_model}+{ollama_embedding_model}",
            "version": version
        }
        self.vector_db_manager.create_lock_file(model_path, process_info)
        
        try:
            logger.info(f"開始重新索引: {ollama_model} + {ollama_embedding_model}")
            
            # 清除現有數據
            if model_path.exists():
                import shutil
                # 保留 .model 文件
                model_file = model_path / ".model"
                model_info = None
                if model_file.exists():
                    import json
                    with open(model_file, 'r', encoding='utf-8') as f:
                        model_info = json.load(f)
                
                # 清除文件夾內容
                shutil.rmtree(model_path)
                model_path.mkdir(parents=True, exist_ok=True)
                
                # 恢復 .model 文件
                if model_info:
                    with open(model_file, 'w', encoding='utf-8') as f:
                        json.dump(model_info, f, ensure_ascii=False, indent=2)
            
            # 創建文檔索引器
            document_indexer = DocumentIndexer(
                vector_db_path=str(model_path),
                ollama_embedding_model=ollama_embedding_model
            )
            
            # 獲取所有文件
            file_crawler = FileCrawler(Q_DRIVE_PATH)
            all_files = list(file_crawler.crawl())
            
            logger.info(f"找到 {len(all_files)} 個文件需要重新索引")
            
            # 開始索引
            success_count, fail_count = document_indexer.index_files(all_files, show_progress=True)
            
            # 保存向量存儲
            document_indexer._save_vector_store()
            
            logger.info(f"重新索引完成: 成功 {success_count} 個，失敗 {fail_count} 個")
            
            return True
            
        except Exception as e:
            logger.error(f"重新索引失敗: {str(e)}")
            return False
        finally:
            # 移除鎖定文件
            self.vector_db_manager.remove_lock_file(model_path)

def main():
    """主函數"""
    # 讀取模型資訊並設置日誌
    model_folder_name = None
    try:
        with open("temp_training_info.json", "r") as f:
            training_info = json.load(f)
        ollama_model = training_info["ollama_model"]
        ollama_embedding_model = training_info["ollama_embedding_model"]
        version = training_info.get("version")
        
        model_folder_name = vector_db_manager.get_model_folder_name(ollama_model, ollama_embedding_model, version)
        setup_model_logger(model_folder_name)
        
        logger.info(f"從臨時文件讀取到模型資訊: {ollama_model} + {ollama_embedding_model}, 版本: {version}")

    except FileNotFoundError:
        # 如果沒有臨時文件，則為自動增量訓練，選擇第一個可用模型
        try:
            usable_models = vector_db_manager.get_usable_models()
            if not usable_models:
                logging.basicConfig(level=logging.INFO) # Fallback logger
                logger.error("沒有可用的模型進行增量訓練，請先進行初始訓練")
                sys.exit(1)
            
            selected_model = usable_models[0]
            model_info = selected_model['model_info']
            ollama_model = model_info['OLLAMA_MODEL']
            ollama_embedding_model = model_info['OLLAMA_EMBEDDING_MODEL']
            version = model_info.get('version')
            model_folder_name = selected_model['folder_name']

            setup_model_logger(model_folder_name)
            logger.info(f"自動選擇模型: {ollama_model} + {ollama_embedding_model}, 版本: {version}")

        except Exception as e:
            logging.basicConfig(level=logging.INFO) # Fallback logger
            logger.error(f"獲取可用模型失敗: {str(e)}")
            sys.exit(1)

    except Exception as e:
        logging.basicConfig(level=logging.INFO) # Fallback logger
        logger.error(f"讀取訓練資訊或設置日誌時出錯: {e}")
        sys.exit(1)

    logger.info("模型訓練管理器啟動")
    
    manager = ModelTrainingManager()

    action = sys.argv[1] if len(sys.argv) > 1 else 'incremental' # 默認為增量
    logger.info(f"執行動作: {action}")

    try:
        if action == 'initial':
            success = manager.start_initial_training(ollama_model, ollama_embedding_model, version)
        elif action == 'incremental':
            success = manager.start_incremental_training(ollama_model, ollama_embedding_model, version)
        elif action == 'reindex':
            success = manager.start_reindex(ollama_model, ollama_embedding_model, version)
        else:
            logger.error(f"未知的 action: {action}")
            sys.exit(1)
        
        if success:
            logger.info(f"動作 {action} 執行成功")
            sys.exit(0)
        else:
            logger.error(f"動作 {action} 執行失敗")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"執行動作 {action} 時發生未預期的錯誤: {str(e)}")
        import traceback
        logger.error(f"錯誤詳情: {traceback.format_exc()}")
        sys.exit(1)
    finally:
        # 清理暫存檔案
        try:
            if os.path.exists("temp_training_info.json"):
                os.remove("temp_training_info.json")
                logger.info("已清理暫存檔案")
        except Exception as e:
            logger.warning(f"清理暫存檔案失敗: {str(e)}")

if __name__ == "__main__":
    main()
