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
import time

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
    
    def start_initial_training(self, ollama_model: str, ollama_embedding_model: str, version: str = None, model_path: Path = None) -> bool:
        """
        開始初始訓練
        """
        # 驗證模型
        if not self.validate_models(ollama_model, ollama_embedding_model):
            return False

        if self.vector_db_manager.has_vector_data(model_path):
            logger.warning(f"模型 {ollama_model}+{ollama_embedding_model} 已有向量數據，請使用增量訓練或重新索引")
            return False

        logger.info(f"開始初始訓練: {ollama_model} + {ollama_embedding_model}")
        self._update_lock_stage(model_path, stage="initial:scanning_files")

        document_indexer = DocumentIndexer(
            vector_db_path=str(model_path),
            ollama_embedding_model=ollama_embedding_model
        )

        file_crawler = FileCrawler(Q_DRIVE_PATH)
        all_files = list(file_crawler.crawl())
        logger.info(f"找到 {len(all_files)} 個文件需要索引")

        self._update_lock_stage(model_path, stage="initial:indexing")
        success_count, fail_count = document_indexer.index_files(all_files, show_progress=True)

        self._update_lock_stage(model_path, stage="initial:saving_store")
        document_indexer._save_vector_store()

        self._update_lock_stage(model_path, stage="initial:completed", meta={"success": success_count, "fail": fail_count})
        logger.info(f"初始訓練完成: 成功 {success_count} 個，失敗 {fail_count} 個")
        return True
    
    def start_incremental_training(self, ollama_model: str, ollama_embedding_model: str, version: str = None, model_path: Path = None) -> bool:
        """
        開始增量訓練
        """
        # 驗證模型
        if not self.validate_models(ollama_model, ollama_embedding_model):
            return False

        if not self.vector_db_manager.has_vector_data(model_path):
            logger.error(f"模型 {ollama_model}+{ollama_embedding_model} 沒有向量數據，請先進行初始訓練")
            return False

        logger.info(f"開始增量訓練: {ollama_model} + {ollama_embedding_model}")
        self._update_lock_stage(model_path, stage="incremental:detect_changes")

        document_indexer = DocumentIndexer(
            vector_db_path=str(model_path),
            ollama_embedding_model=ollama_embedding_model
        )

        file_crawler = FileCrawler(Q_DRIVE_PATH)
        new_files, updated_files, deleted_files = file_crawler.get_file_changes(
            set(document_indexer.indexed_files.keys()) if hasattr(document_indexer, 'indexed_files') else set()
        )

        total_changes = len(new_files) + len(updated_files) + len(deleted_files)
        logger.info(f"檢測到 {total_changes} 個文件變更: 新增 {len(new_files)}, 更新 {len(updated_files)}, 刪除 {len(deleted_files)}")

        if total_changes == 0:
            logger.info("沒有文件變更，無需增量訓練")
            self._update_lock_stage(model_path, stage="incremental:no_changes")
            return True

        self._update_lock_stage(model_path, stage="incremental:processing_changes", meta={"new": len(new_files), "updated": len(updated_files), "deleted": len(deleted_files)})
        document_indexer.process_file_changes(new_files, updated_files, deleted_files)

        self._update_lock_stage(model_path, stage="incremental:saving_store")
        document_indexer._save_vector_store()
        self._update_lock_stage(model_path, stage="incremental:completed")
        logger.info("增量訓練完成")
        return True
    
    def start_reindex(self, ollama_model: str, ollama_embedding_model: str, version: str = None, model_path: Path = None) -> bool:
        """
        開始重新索引
        """
        # 驗證模型
        if not self.validate_models(ollama_model, ollama_embedding_model):
            return False

        logger.info(f"開始重新索引: {ollama_model} + {ollama_embedding_model}")
        self._update_lock_stage(model_path, stage="reindex:clearing_old")

        if model_path.exists():
            # 為了避免刪除資料夾時把 .lock 一併移除，導致前端顯示狀態瞬間變成「未鎖定 / 可用」，
            # 我們先保存現有的 .model 與 .lock 資料，再清空資料夾後重建並恢復 lock。
            import shutil, json, tempfile
            model_file = model_path / ".model"
            lock_file = model_path / ".lock"
            model_info = None
            lock_info = None
            try:
                if model_file.exists():
                    with open(model_file, 'r', encoding='utf-8') as f:
                        model_info = json.load(f)
            except Exception as e:
                logger.warning(f"讀取 .model 失敗，將嘗試繼續: {e}")
            try:
                if lock_file.exists():
                    with open(lock_file, 'r', encoding='utf-8') as f:
                        lock_info = json.load(f)
            except Exception as e:
                logger.warning(f"讀取 .lock 失敗，將重新建立: {e}")

            # 直接刪除整個資料夾
            try:
                shutil.rmtree(model_path)
            except Exception as e:
                logger.error(f"刪除舊索引資料夾失敗: {e}")
                return False
            model_path.mkdir(parents=True, exist_ok=True)

            # 還原 .model
            if model_info:
                try:
                    with open(model_file, 'w', encoding='utf-8') as f:
                        json.dump(model_info, f, ensure_ascii=False, indent=2)
                except Exception as e:
                    logger.warning(f"寫回 .model 失敗（忽略）: {e}")

            # 還原 (或重建) .lock 以維持前端訓練中狀態
            try:
                if lock_info:
                    # 更新階段與時間戳
                    import time
                    lock_info['stage'] = 'reindex:clearing_old'
                    lock_info['stage_ts'] = int(time.time())
                    with open(lock_file, 'w', encoding='utf-8') as f:
                        json.dump(lock_info, f, ensure_ascii=False, indent=2)
                else:
                    # 若原本沒有 lock（理論上不應發生），重建一份簡化的鎖定資訊
                    from utils.training_lock_manager import training_lock_manager
                    training_lock_manager.create_lock(model_path, {"type": "reindex", "status": "restarting"})
            except Exception as e:
                logger.warning(f"恢復 .lock 失敗（忽略）: {e}")

        document_indexer = DocumentIndexer(
            vector_db_path=str(model_path),
            ollama_embedding_model=ollama_embedding_model
        )

        file_crawler = FileCrawler(Q_DRIVE_PATH)
        all_files = list(file_crawler.crawl())
        self._update_lock_stage(model_path, stage="reindex:indexing", meta={"total_files": len(all_files)})
        logger.info(f"找到 {len(all_files)} 個文件需要重新索引")

        success_count, fail_count = document_indexer.index_files(all_files, show_progress=True)

        self._update_lock_stage(model_path, stage="reindex:saving_store")
        document_indexer._save_vector_store()

        self._update_lock_stage(model_path, stage="reindex:completed", meta={"success": success_count, "fail": fail_count})
        logger.info(f"重新索引完成: 成功 {success_count} 個，失敗 {fail_count} 個")
        return True

    # ----------------- 輔助：更新鎖定文件中的階段信息 -----------------
    def _update_lock_stage(self, model_path: Path, stage: str, meta: dict = None):
        """在鎖定文件中寫入目前階段，供前端/監控輪詢顯示進度。"""
        try:
            lock_file = model_path / ".lock"
            if not lock_file.exists():
                return
            import json, time
            with open(lock_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            data['stage'] = stage
            data['stage_ts'] = int(time.time())
            if meta:
                data.setdefault('stage_meta', {}).update(meta)
            with open(lock_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.debug(f"更新鎖定階段失敗（忽略）: {e}")

    def _heartbeat(self, model_path: Path):
        """更新鎖檔心跳。"""
        try:
            lock_file = model_path / '.lock'
            if not lock_file.exists():
                return
            import json
            with open(lock_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            data['heartbeat_ts'] = int(time.time())
            with open(lock_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _check_cancel_flag(self, model_path: Path) -> bool:
        return (model_path / 'cancel.flag').exists()

def main():
    """主函數"""
    model_path = None
    try:
        # 讀取模型資訊並設置日誌
        with open("temp_training_info.json", "r") as f:
            training_info = json.load(f)
        ollama_model = training_info["ollama_model"]
        ollama_embedding_model = training_info["ollama_embedding_model"]
        version = training_info.get("version")
        
        model_folder_name = vector_db_manager.get_model_folder_name(ollama_model, ollama_embedding_model, version)
        setup_model_logger(model_folder_name)
        
        logger.info(f"從臨時文件讀取到模型資訊: {ollama_model} + {ollama_embedding_model}, 版本: {version}")

    except FileNotFoundError:
        # 自動增量訓練
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
            logging.basicConfig(level=logging.INFO)
            logger.error(f"獲取可用模型失敗: {str(e)}")
            sys.exit(1)

    except Exception as e:
        logging.basicConfig(level=logging.INFO)
        logger.error(f"讀取訓練資訊或設置日誌時出錯: {e}")
        sys.exit(1)

    logger.info("模型訓練管理器啟動")
    
    manager = ModelTrainingManager()
    action = sys.argv[1] if len(sys.argv) > 1 else 'incremental'
    logger.info(f"執行動作: {action}")

    # 獲取模型路徑並立即創建鎖定文件
    folder_name = vector_db_manager.get_model_folder_name(ollama_model, ollama_embedding_model, version)
    model_path = vector_db_manager.base_path / folder_name
    model_path.mkdir(parents=True, exist_ok=True)

    # 確保存在 .model 檔（部分舊流程僅建立資料夾，未寫入 .model，導致前端無法辨識/選取）
    try:
        model_file = model_path / '.model'
        if not model_file.exists():
            import datetime
            model_info = {
                "OLLAMA_MODEL": ollama_model,
                "OLLAMA_EMBEDDING_MODEL": ollama_embedding_model,
                "created_at": datetime.datetime.now().isoformat()
            }
            if version:
                model_info["version"] = version
            with open(model_file, 'w', encoding='utf-8') as f:
                json.dump(model_info, f, ensure_ascii=False, indent=2)
            logger.info(f"已自動建立缺失的 .model 檔: {model_file}")
        else:
            # 可選：驗證必要欄位
            try:
                with open(model_file, 'r', encoding='utf-8') as f:
                    info = json.load(f)
                changed = False
                if 'OLLAMA_MODEL' not in info:
                    info['OLLAMA_MODEL'] = ollama_model; changed = True
                if 'OLLAMA_EMBEDDING_MODEL' not in info:
                    info['OLLAMA_EMBEDDING_MODEL'] = ollama_embedding_model; changed = True
                if version and info.get('version') != version:
                    info['version'] = version; changed = True
                if 'created_at' not in info:
                    import datetime as _dt
                    info['created_at'] = _dt.datetime.now().isoformat(); changed = True
                if changed:
                    with open(model_file, 'w', encoding='utf-8') as f:
                        json.dump(info, f, ensure_ascii=False, indent=2)
                    logger.info(f"已修正不完整的 .model 檔: {model_file}")
            except Exception as _e:
                logger.warning(f"讀取/驗證現有 .model 檔失敗，將覆寫: {_e}")
                try:
                    with open(model_file, 'w', encoding='utf-8') as f:
                        json.dump({
                            "OLLAMA_MODEL": ollama_model,
                            "OLLAMA_EMBEDDING_MODEL": ollama_embedding_model,
                            **({"version": version} if version else {}),
                            "created_at": __import__('datetime').datetime.now().isoformat()
                        }, f, ensure_ascii=False, indent=2)
                except Exception:
                    pass
    except Exception as e:
        logger.warning(f"建立或驗證 .model 檔時發生例外（忽略不致命）: {e}")

    process_info = {
        "training_type": action,
        "model_combination": f"{ollama_model}+{ollama_embedding_model}",
        "version": version
    }
    manager.vector_db_manager.create_lock_file(model_path, process_info)
    logger.info(f"已為模型 {folder_name} 創建鎖定文件")
    manager._update_lock_stage(model_path, stage=f"{action}:init")

    cancelled_gracefully = False
    try:
        if action == 'initial':
            success = manager.start_initial_training(ollama_model, ollama_embedding_model, version, model_path)
        elif action == 'incremental':
            success = manager.start_incremental_training(ollama_model, ollama_embedding_model, version, model_path)
        elif action == 'reindex':
            success = manager.start_reindex(ollama_model, ollama_embedding_model, version, model_path)
        else:
            logger.error(f"未知的 action: {action}")
            sys.exit(1)
        
        # 檢查是否有取消旗標
        if manager._check_cancel_flag(model_path):
            cancelled_gracefully = True
            success = False
            manager._update_lock_stage(model_path, stage=f"{action}:cancelled", meta={"graceful": True})
            logger.info("訓練被優雅取消。")

        if success:
            manager._update_lock_stage(model_path, stage=f"{action}:success")
            logger.info(f"動作 {action} 執行成功")
            sys.exit(0)
        else:
            if not cancelled_gracefully:
                manager._update_lock_stage(model_path, stage=f"{action}:failed")
                logger.error(f"動作 {action} 執行失敗")
                sys.exit(1)
            else:
                sys.exit(0)
            
    except Exception as e:
        logger.error(f"執行動作 {action} 時發生未預期的錯誤: {str(e)}")
        import traceback
        logger.error(f"錯誤詳情: {traceback.format_exc()}")
        try:
            manager._update_lock_stage(model_path, stage=f"{action}:exception", meta={"error": str(e)})
        except Exception:
            pass
        sys.exit(1)
    finally:
        # 確保鎖定文件和臨時文件被清理
        if model_path:
            if cancelled_gracefully:
                # 優雅取消：仍移除鎖，前端顯示狀態可透過 .stage_meta 判斷
                pass
            manager.vector_db_manager.remove_lock_file(model_path)
            logger.info(f"已為模型 {folder_name} 移除鎖定文件 (cancelled={cancelled_gracefully})")
            # 清理 cancel.flag
            try:
                cf = model_path / 'cancel.flag'
                if cf.exists():
                    cf.unlink()
            except Exception:
                pass
        try:
            if os.path.exists("temp_training_info.json"):
                os.remove("temp_training_info.json")
                logger.info("已清理暫存檔案")
        except Exception as e:
            logger.warning(f"清理暫存檔案失敗: {str(e)}")

if __name__ == "__main__":
    main()
