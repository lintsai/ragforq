#!/usr/bin/env python
"""
模型訓練管理器 - 調試版本
增加詳細的錯誤處理和日誌記錄，用於診斷增量索引問題
"""
import os
import sys
import logging
import json
import argparse
import traceback
import signal
from pathlib import Path

# 添加項目根目錄到路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.vector_db_manager import vector_db_manager
from utils.ollama_utils import ollama_utils
from indexer.document_indexer import DocumentIndexer
from indexer.file_crawler import FileCrawler
from config.config import Q_DRIVE_PATH, get_supported_file_extensions

# 設置更詳細的日誌
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/var/log/supervisor/debug_training.log', encoding='utf-8')
    ]
)

logger = logging.getLogger(__name__)

class ModelTrainingManagerDebug:
    """模型訓練管理器 - 調試版本"""
    
    def __init__(self):
        self.vector_db_manager = vector_db_manager
        self.ollama_utils = ollama_utils
        self.current_lock_path = None
        
        # 註冊信號處理器，確保異常退出時能清理資源
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """信號處理器，確保清理資源"""
        logger.warning(f"收到信號 {signum}，正在清理資源...")
        self._cleanup_resources()
        sys.exit(1)
    
    def _cleanup_resources(self):
        """清理資源"""
        try:
            if self.current_lock_path and os.path.exists(self.current_lock_path):
                os.remove(self.current_lock_path)
                logger.info(f"已清理鎖定文件: {self.current_lock_path}")
        except Exception as e:
            logger.error(f"清理資源時出錯: {str(e)}")
    
    def validate_models(self, ollama_model: str, ollama_embedding_model: str) -> bool:
        """驗證模型是否可用"""
        try:
            logger.info("開始驗證模型可用性...")
            available_models = self.ollama_utils.get_model_names()
            logger.info(f"可用模型列表: {available_models}")
            
            if ollama_model not in available_models:
                logger.error(f"語言模型 '{ollama_model}' 不可用")
                return False
            
            if ollama_embedding_model not in available_models:
                logger.error(f"嵌入模型 '{ollama_embedding_model}' 不可用")
                return False
            
            logger.info("模型驗證通過")
            return True
        except Exception as e:
            logger.error(f"驗證模型時出錯: {str(e)}")
            logger.error(f"詳細錯誤: {traceback.format_exc()}")
            return False
    
    def start_incremental_training(self, ollama_model: str, ollama_embedding_model: str, version: str = None) -> bool:
        """開始增量訓練 - 調試版本"""
        logger.info("=" * 60)
        logger.info("開始增量訓練調試版本")
        logger.info("=" * 60)
        
        try:
            # 步驟 1: 驗證模型
            logger.info("步驟 1: 驗證模型")
            if not self.validate_models(ollama_model, ollama_embedding_model):
                logger.error("模型驗證失敗")
                return False
            logger.info("✓ 模型驗證成功")
            
            # 步驟 2: 獲取模型路徑
            logger.info("步驟 2: 獲取模型路徑")
            if version:
                folder_name = self.vector_db_manager.get_model_folder_name(ollama_model, ollama_embedding_model, version)
                model_path = self.vector_db_manager.base_path / folder_name
            else:
                model_path = self.vector_db_manager.get_model_path(ollama_model, ollama_embedding_model)
            
            logger.info(f"模型路徑: {model_path}")
            logger.info(f"路徑是否存在: {model_path.exists()}")
            
            # 步驟 3: 檢查向量數據
            logger.info("步驟 3: 檢查向量數據")
            has_vector_data = self.vector_db_manager.has_vector_data(model_path)
            logger.info(f"是否有向量數據: {has_vector_data}")
            
            if not has_vector_data:
                logger.error(f"模型 {ollama_model}+{ollama_embedding_model} 沒有向量數據，請先進行初始訓練")
                return False
            logger.info("✓ 向量數據檢查通過")
            
            # 步驟 4: 創建鎖定文件
            logger.info("步驟 4: 創建鎖定文件")
            lock_file_path = model_path / ".lock"
            self.current_lock_path = str(lock_file_path)
            
            # 檢查是否已有鎖定文件
            if lock_file_path.exists():
                logger.warning(f"發現現有鎖定文件: {lock_file_path}")
                # 檢查鎖定文件的年齡
                import time
                lock_age = time.time() - lock_file_path.stat().st_mtime
                if lock_age > 3600:  # 1小時
                    logger.warning(f"鎖定文件已存在 {lock_age/60:.1f} 分鐘，可能是殘留文件，將刪除")
                    lock_file_path.unlink()
                else:
                    logger.error(f"鎖定文件太新 ({lock_age/60:.1f} 分鐘)，可能有其他任務正在運行")
                    return False
            
            self.vector_db_manager.create_lock_file(model_path)
            logger.info(f"✓ 鎖定文件已創建: {lock_file_path}")
            
            # 步驟 5: 創建文檔索引器
            logger.info("步驟 5: 創建文檔索引器")
            logger.info(f"向量數據庫路徑: {model_path}")
            logger.info(f"嵌入模型: {ollama_embedding_model}")
            
            try:
                document_indexer = DocumentIndexer(
                    vector_db_path=str(model_path),
                    ollama_embedding_model=ollama_embedding_model
                )
                logger.info("✓ 文檔索引器創建成功")
            except Exception as e:
                logger.error(f"創建文檔索引器失敗: {str(e)}")
                logger.error(f"詳細錯誤: {traceback.format_exc()}")
                return False
            
            # 步驟 6: 檢查索引器狀態
            logger.info("步驟 6: 檢查索引器狀態")
            logger.info(f"已索引文件數量: {len(document_indexer.indexed_files)}")
            logger.info(f"索引進度: {document_indexer.indexing_progress}")
            
            # 步驟 7: 獲取文件列表
            logger.info("步驟 7: 獲取文件列表")
            try:
                file_crawler = FileCrawler(Q_DRIVE_PATH)
                all_files = list(file_crawler.crawl())
                logger.info(f"找到 {len(all_files)} 個文件")
            except Exception as e:
                logger.error(f"獲取文件列表失敗: {str(e)}")
                logger.error(f"詳細錯誤: {traceback.format_exc()}")
                return False
            
            # 步驟 8: 檢查文件變更
            logger.info("步驟 8: 檢查文件變更")
            try:
                indexed_files_set = set(document_indexer.indexed_files.keys()) if hasattr(document_indexer, 'indexed_files') else set()
                logger.info(f"當前已索引文件數量: {len(indexed_files_set)}")
                
                new_files, updated_files, deleted_files = file_crawler.get_file_changes(indexed_files_set)
                
                total_changes = len(new_files) + len(updated_files) + len(deleted_files)
                logger.info(f"檢測到 {total_changes} 個文件變更:")
                logger.info(f"  - 新增: {len(new_files)} 個")
                logger.info(f"  - 更新: {len(updated_files)} 個")
                logger.info(f"  - 刪除: {len(deleted_files)} 個")
                
                if total_changes == 0:
                    logger.info("沒有文件變更，增量訓練完成")
                    return True
                
            except Exception as e:
                logger.error(f"檢查文件變更失敗: {str(e)}")
                logger.error(f"詳細錯誤: {traceback.format_exc()}")
                return False
            
            # 步驟 9: 處理文件變更
            logger.info("步驟 9: 處理文件變更")
            try:
                document_indexer.process_file_changes(new_files, updated_files, deleted_files)
                logger.info("✓ 文件變更處理完成")
            except Exception as e:
                logger.error(f"處理文件變更失敗: {str(e)}")
                logger.error(f"詳細錯誤: {traceback.format_exc()}")
                return False
            
            # 步驟 10: 保存向量存儲
            logger.info("步驟 10: 保存向量存儲")
            try:
                document_indexer._save_vector_store()
                logger.info("✓ 向量存儲保存完成")
            except Exception as e:
                logger.error(f"保存向量存儲失敗: {str(e)}")
                logger.error(f"詳細錯誤: {traceback.format_exc()}")
                return False
            
            logger.info("=" * 60)
            logger.info("增量訓練成功完成")
            logger.info("=" * 60)
            return True
            
        except Exception as e:
            logger.error(f"增量訓練過程中發生未預期的錯誤: {str(e)}")
            logger.error(f"詳細錯誤堆疊: {traceback.format_exc()}")
            return False
        finally:
            # 步驟 11: 清理資源
            logger.info("步驟 11: 清理資源")
            try:
                if self.current_lock_path and os.path.exists(self.current_lock_path):
                    os.remove(self.current_lock_path)
                    logger.info(f"✓ 鎖定文件已清理: {self.current_lock_path}")
                else:
                    logger.warning(f"鎖定文件不存在或已被清理: {self.current_lock_path}")
            except Exception as e:
                logger.error(f"清理鎖定文件失敗: {str(e)}")

def main():
    """主函數"""
    logger.info("模型訓練管理器調試版本啟動")
    
    # 優先從暫存檔案讀取（API 調用），如果沒有則從現有模型中選擇
    ollama_model = None
    ollama_embedding_model = None
    version = None
    
    try:
        logger.info("正在讀取訓練資訊檔案...")
        with open("temp_training_info.json", "r") as f:
            training_info = json.load(f)
        ollama_model = training_info["ollama_model"]
        ollama_embedding_model = training_info["ollama_embedding_model"]
        version = training_info.get("version")
        logger.info(f"從臨時文件讀取到模型資訊: {ollama_model} + {ollama_embedding_model}, 版本: {version}")
    except FileNotFoundError:
        logger.info("未找到臨時訓練資訊檔案，嘗試從現有模型中選擇...")
        # 從可用的模型中選擇一個進行增量訓練
        try:
            usable_models = vector_db_manager.get_usable_models()
            if not usable_models:
                logger.error("沒有可用的模型進行增量訓練，請先進行初始訓練")
                logger.info("可用的解決方案:")
                logger.info("1. 通過 API 進行初始訓練")
                logger.info("2. 檢查是否有現有的向量數據庫文件夾")
                sys.exit(1)
            
            # 選擇第一個可用模型
            selected_model = usable_models[0]
            model_info = selected_model['model_info']
            ollama_model = model_info['OLLAMA_MODEL']
            ollama_embedding_model = model_info['OLLAMA_EMBEDDING_MODEL']
            version = model_info.get('version')
            
            logger.info(f"自動選擇模型: {ollama_model} + {ollama_embedding_model}, 版本: {version}")
            logger.info(f"模型路徑: {selected_model['folder_path']}")
            logger.info(f"可用模型總數: {len(usable_models)}")
            
        except Exception as e:
            logger.error(f"獲取可用模型失敗: {str(e)}")
            logger.error(f"詳細錯誤: {traceback.format_exc()}")
            sys.exit(1)
    except json.JSONDecodeError as e:
        logger.error(f"解析訓練資訊檔案失敗: {str(e)}")
        sys.exit(1)
    except KeyError as e:
        logger.error(f"訓練資訊檔案缺少必要欄位: {str(e)}")
        sys.exit(1)

    manager = ModelTrainingManagerDebug()

    action = sys.argv[1] if len(sys.argv) > 1 else None
    logger.info(f"執行動作: {action}")

    try:
        if action == 'incremental':
            success = manager.start_incremental_training(ollama_model, ollama_embedding_model, version)
        else:
            logger.error(f"調試版本只支持 incremental 動作，收到: {action}")
            sys.exit(1)
        
        if success:
            logger.info(f"動作 {action} 執行成功")
            sys.exit(0)
        else:
            logger.error(f"動作 {action} 執行失敗")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"執行動作 {action} 時發生未預期的錯誤: {str(e)}")
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
