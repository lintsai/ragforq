#!/usr/bin/env python
"""
恢復索引腳本 - 從中斷點繼續索引
"""

import os
import sys
import logging
import time
import gc
import psutil
from pathlib import Path

# 添加項目根目錄到路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from indexer.file_crawler import FileCrawler
from indexer.document_indexer import DocumentIndexer
from config.config import Q_DRIVE_PATH, VECTOR_DB_PATH, FILE_BATCH_SIZE

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/indexing.log', mode='a', encoding='utf-8')  # 使用追加模式
    ]
)

logger = logging.getLogger(__name__)

def check_system_resources():
    """檢查系統資源狀況"""
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    logger.info(f"系統資源狀況:")
    logger.info(f"  內存使用: {memory.percent}% ({memory.used/1024/1024/1024:.1f}GB/{memory.total/1024/1024/1024:.1f}GB)")
    logger.info(f"  磁盤使用: {disk.percent}% ({disk.used/1024/1024/1024:.1f}GB/{disk.total/1024/1024/1024:.1f}GB)")
    
    # 檢查資源是否充足
    if memory.percent > 85:
        logger.warning("內存使用率過高，建議釋放內存後再繼續")
        return False
    
    if disk.percent > 90:
        logger.warning("磁盤空間不足，建議清理磁盤後再繼續")
        return False
    
    return True

def check_network_mount():
    """檢查網絡掛載狀態"""
    try:
        if os.path.exists(Q_DRIVE_PATH):
            # 嘗試列出目錄內容來測試連接
            test_files = list(os.listdir(Q_DRIVE_PATH))[:5]
            logger.info(f"網絡掛載正常，可訪問 {len(test_files)} 個項目")
            return True
        else:
            logger.error(f"無法訪問 Q 槽路徑: {Q_DRIVE_PATH}")
            return False
    except Exception as e:
        logger.error(f"檢查網絡掛載時出錯: {str(e)}")
        return False

def check_ollama_service():
    """檢查 Ollama 服務狀態"""
    try:
        import requests
        from config.config import OLLAMA_HOST
        
        response = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=10)
        if response.status_code == 200:
            logger.info("Ollama 服務正常運行")
            return True
        else:
            logger.error(f"Ollama 服務響應異常: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"檢查 Ollama 服務時出錯: {str(e)}")
        return False

def main():
    try:
        logger.info("=" * 60)
        logger.info("開始恢復索引過程")
        logger.info("=" * 60)
        
        # 系統檢查
        logger.info("1. 檢查系統資源...")
        if not check_system_resources():
            logger.error("系統資源不足，請檢查後重試")
            return
        
        logger.info("2. 檢查網絡掛載...")
        if not check_network_mount():
            logger.error("網絡掛載異常，請檢查後重試")
            return
        
        logger.info("3. 檢查 Ollama 服務...")
        if not check_ollama_service():
            logger.error("Ollama 服務異常，請檢查後重試")
            return
        
        # 確保向量數據庫目錄存在
        vector_db_dir = Path(VECTOR_DB_PATH)
        vector_db_dir.mkdir(parents=True, exist_ok=True)
        
        # 創建文件索引器
        logger.info("4. 初始化文件索引器...")
        indexer = DocumentIndexer()
        
        # 檢查索引進度
        progress = indexer.indexing_progress
        logger.info(f"當前索引進度:")
        logger.info(f"  已完成批次: {progress.get('completed_batches', 0)}")
        logger.info(f"  總批次數: {progress.get('total_batches', 0)}")
        logger.info(f"  待處理文件: {len(progress.get('pending_files', []))}")
        logger.info(f"  索引進行中: {progress.get('in_progress', False)}")
        
        # 如果有未完成的任務，繼續執行
        if progress.get('in_progress', False) or progress.get('pending_files', []):
            logger.info("5. 發現未完成的索引任務，繼續執行...")
            
            # 獲取所有需要處理的文件
            if not progress.get('pending_files', []):
                logger.info("重新爬取文件列表...")
                crawler = FileCrawler(Q_DRIVE_PATH)
                all_files = list(crawler.crawl())
                
                # 過濾已索引的文件
                pending_files = []
                for file_path in all_files:
                    file_mtime = os.path.getmtime(file_path)
                    if file_path not in indexer.indexed_files or indexer.indexed_files[file_path] < file_mtime:
                        pending_files.append(file_path)
                
                logger.info(f"找到 {len(pending_files)} 個需要處理的文件")
            else:
                pending_files = progress['pending_files']
                logger.info(f"繼續處理 {len(pending_files)} 個待處理文件")
            
            if pending_files:
                # 使用較小的批次大小以提高穩定性
                batch_size = min(FILE_BATCH_SIZE, 10)  # 限制最大批次大小
                success_count, fail_count = indexer.index_files(pending_files, show_progress=True)
                
                logger.info(f"索引完成: {success_count} 成功, {fail_count} 失敗")
                
                # 強制垃圾回收
                gc.collect()
            else:
                logger.info("沒有需要處理的文件")
        else:
            logger.info("5. 沒有發現未完成的索引任務")
            
            # 檢查是否需要重新索引
            logger.info("檢查是否有新文件需要索引...")
            crawler = FileCrawler(Q_DRIVE_PATH)
            all_files = list(crawler.crawl())
            
            new_files = []
            for file_path in all_files:
                if not os.path.exists(file_path):
                    continue
                    
                file_mtime = os.path.getmtime(file_path)
                if file_path not in indexer.indexed_files or indexer.indexed_files[file_path] < file_mtime:
                    new_files.append(file_path)
            
            if new_files:
                logger.info(f"發現 {len(new_files)} 個新文件或更新文件")
                success_count, fail_count = indexer.index_files(new_files, show_progress=True)
                logger.info(f"索引完成: {success_count} 成功, {fail_count} 失敗")
            else:
                logger.info("所有文件都已是最新索引")
        
        # 顯示最終統計
        indexed_files = indexer.list_indexed_files()
        logger.info(f"索引統計: 共 {len(indexed_files)} 個文件已建立索引")
        
        logger.info("=" * 60)
        logger.info("索引恢復過程完成")
        logger.info("=" * 60)
        
    except KeyboardInterrupt:
        logger.info("用戶中斷操作，進度已保存")
    except Exception as e:
        logger.error(f"恢復索引過程失敗: {str(e)}")
        import traceback
        logger.error(f"詳細錯誤信息: {traceback.format_exc()}")

if __name__ == "__main__":
    main()