#!/usr/bin/env python
"""
穩定恢復索引腳本 - 針對大規模文件索引優化
解決內存不足、COM錯誤、網絡中斷等問題
"""

import os
import sys
import logging
import time
import gc
import signal
import json
from pathlib import Path
from typing import List, Set

# 添加項目根目錄到路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from indexer.file_crawler import FileCrawler
from indexer.document_indexer import DocumentIndexer
from config.config import Q_DRIVE_PATH, VECTOR_DB_PATH

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/indexing.log', mode='a', encoding='utf-8')
    ]
)

logger = logging.getLogger(__name__)

class StableIndexer:
    def __init__(self):
        self.indexer = None
        self.should_stop = False
        self.processed_count = 0
        self.failed_files = []
        
        # 註冊信號處理器
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """處理中斷信號"""
        logger.info(f"收到信號 {signum}，正在安全停止...")
        self.should_stop = True
    
    def _check_system_health(self) -> bool:
        """檢查系統健康狀態"""
        try:
            # 檢查內存使用
            import psutil
            memory = psutil.virtual_memory()
            if memory.percent > 90:
                logger.warning(f"內存使用率過高: {memory.percent}%")
                return False
            
            # 檢查磁盤空間
            disk = psutil.disk_usage('/')
            if disk.percent > 95:
                logger.warning(f"磁盤空間不足: {disk.percent}%")
                return False
            
            return True
        except ImportError:
            logger.warning("未安裝 psutil，跳過系統健康檢查")
            return True
        except Exception as e:
            logger.error(f"系統健康檢查失敗: {str(e)}")
            return True
    
    def _check_network_connectivity(self) -> bool:
        """檢查網絡連接"""
        try:
            if not os.path.exists(Q_DRIVE_PATH):
                logger.error(f"無法訪問 Q 槽: {Q_DRIVE_PATH}")
                return False
            
            # 嘗試讀取一個文件來測試連接
            test_path = os.path.join(Q_DRIVE_PATH, ".")
            os.listdir(test_path)
            return True
        except Exception as e:
            logger.error(f"網絡連接檢查失敗: {str(e)}")
            return False
    
    def _check_ollama_service(self) -> bool:
        """檢查 Ollama 服務"""
        try:
            import requests
            from config.config import OLLAMA_HOST
            
            response = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Ollama 服務檢查失敗: {str(e)}")
            return False
    
    def _get_pending_files(self) -> List[str]:
        """獲取待處理文件列表"""
        logger.info("分析待處理文件...")
        
        # 初始化索引器
        if not self.indexer:
            self.indexer = DocumentIndexer()
        
        # 檢查進度記錄
        progress = self.indexer.indexing_progress
        
        if progress.get('pending_files'):
            logger.info(f"從進度記錄中找到 {len(progress['pending_files'])} 個待處理文件")
            return progress['pending_files']
        
        # 重新爬取文件
        logger.info("重新爬取文件列表...")
        crawler = FileCrawler(Q_DRIVE_PATH)
        all_files = list(crawler.crawl())
        
        # 過濾需要處理的文件
        pending_files = []
        for file_path in all_files:
            try:
                if not os.path.exists(file_path):
                    continue
                
                file_mtime = os.path.getmtime(file_path)
                if (file_path not in self.indexer.indexed_files or 
                    self.indexer.indexed_files[file_path] < file_mtime):
                    pending_files.append(file_path)
            except Exception as e:
                logger.warning(f"檢查文件 {file_path} 時出錯: {str(e)}")
                continue
        
        logger.info(f"找到 {len(pending_files)} 個需要處理的文件")
        return pending_files
    
    def _process_batch_safely(self, batch: List[str]) -> tuple:
        """安全處理一批文件"""
        success_count = 0
        fail_count = 0
        
        for file_path in batch:
            if self.should_stop:
                logger.info("收到停止信號，中斷處理")
                break
            
            try:
                # 檢查系統健康狀態
                if not self._check_system_health():
                    logger.warning("系統資源不足，暫停處理")
                    time.sleep(10)
                    gc.collect()
                    continue
                
                # 處理單個文件
                if self.indexer.index_file(file_path):
                    success_count += 1
                    self.processed_count += 1
                else:
                    fail_count += 1
                    self.failed_files.append(file_path)
                
                # 每處理10個文件進行一次清理
                if (success_count + fail_count) % 10 == 0:
                    gc.collect()
                
            except Exception as e:
                logger.error(f"處理文件 {file_path} 時出錯: {str(e)}")
                fail_count += 1
                self.failed_files.append(file_path)
                continue
        
        return success_count, fail_count
    
    def run(self):
        """執行穩定索引恢復"""
        try:
            logger.info("=" * 60)
            logger.info("開始穩定索引恢復過程")
            logger.info("=" * 60)
            
            # 系統檢查
            logger.info("1. 系統健康檢查...")
            if not self._check_system_health():
                logger.error("系統資源不足，建議稍後重試")
                return False
            
            logger.info("2. 網絡連接檢查...")
            if not self._check_network_connectivity():
                logger.error("網絡連接異常，請檢查 Q 槽掛載")
                return False
            
            logger.info("3. Ollama 服務檢查...")
            if not self._check_ollama_service():
                logger.error("Ollama 服務異常，請檢查服務狀態")
                return False
            
            # 獲取待處理文件
            logger.info("4. 獲取待處理文件...")
            pending_files = self._get_pending_files()
            
            if not pending_files:
                logger.info("沒有需要處理的文件")
                return True
            
            # 使用小批次處理，提高穩定性
            batch_size = 5  # 減小批次大小
            total_files = len(pending_files)
            total_success = 0
            total_fail = 0
            
            logger.info(f"5. 開始處理 {total_files} 個文件，批次大小: {batch_size}")
            
            for i in range(0, total_files, batch_size):
                if self.should_stop:
                    break
                
                batch_end = min(i + batch_size, total_files)
                batch = pending_files[i:batch_end]
                
                logger.info(f"處理批次 {i//batch_size + 1}/{(total_files + batch_size - 1)//batch_size}")
                logger.info(f"  文件範圍: {i+1}-{batch_end}/{total_files}")
                
                # 處理批次
                success, fail = self._process_batch_safely(batch)
                total_success += success
                total_fail += fail
                
                logger.info(f"  批次結果: {success} 成功, {fail} 失敗")
                
                # 更新進度
                remaining_files = pending_files[batch_end:]
                self.indexer._save_indexing_progress(
                    pending_files=remaining_files,
                    completed_batches=i//batch_size + 1,
                    total_batches=(total_files + batch_size - 1)//batch_size
                )
                
                # 定期保存
                if (i//batch_size + 1) % 10 == 0:
                    logger.info("定期保存向量數據庫...")
                    self.indexer._save_vector_store()
                    self.indexer._save_indexed_files()
                
                # 短暫休息，避免系統過載
                time.sleep(1)
            
            # 最終保存
            logger.info("6. 保存最終結果...")
            self.indexer._save_vector_store()
            self.indexer._save_indexed_files()
            
            # 標記完成
            self.indexer._save_indexing_progress(
                pending_files=[],
                in_progress=False
            )
            
            # 統計結果
            logger.info("=" * 60)
            logger.info("索引恢復完成")
            logger.info(f"總處理文件: {total_success + total_fail}")
            logger.info(f"成功: {total_success}")
            logger.info(f"失敗: {total_fail}")
            
            if self.failed_files:
                logger.info(f"失敗文件數: {len(self.failed_files)}")
                # 保存失敗文件列表
                failed_file_path = "logs/failed_files.json"
                with open(failed_file_path, 'w', encoding='utf-8') as f:
                    json.dump(self.failed_files, f, ensure_ascii=False, indent=2)
                logger.info(f"失敗文件列表已保存到: {failed_file_path}")
            
            logger.info("=" * 60)
            return True
            
        except Exception as e:
            logger.error(f"索引恢復過程失敗: {str(e)}")
            import traceback
            logger.error(f"詳細錯誤: {traceback.format_exc()}")
            return False

def main():
    indexer = StableIndexer()
    success = indexer.run()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()