#!/usr/bin/env python
"""
初始化索引腳本 - 爬取Q槽文件並建立索引
"""

import os
import sys
import logging
import time
from pathlib import Path
import gc
import json

# 添加項目根目錄到路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from indexer.file_crawler import FileCrawler
from indexer.document_indexer import DocumentIndexer
from config.config import Q_DRIVE_PATH, VECTOR_DB_PATH, FILE_BATCH_SIZE, MAX_WORKERS

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/indexing.log', mode='w', encoding='utf-8')
    ]
)

logger = logging.getLogger(__name__)

PROGRESS_FILE = "logs/indexing_progress.json"

def save_progress(processed_files):
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(list(processed_files), f, ensure_ascii=False, indent=2)

def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            return set(json.load(f))
    return set()

def main():
    try:
        logger.info("開始執行初始化索引過程")
        
        # 確保向量數據庫目錄存在
        vector_db_dir = Path(VECTOR_DB_PATH)
        vector_db_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"向量數據庫目錄: {vector_db_dir}")
        
        # 創建文件爬蟲
        logger.info("創建文件爬蟲...")
        crawler = FileCrawler(Q_DRIVE_PATH)
        
        # 爬取文件
        logger.info("開始爬取文件...")
        start_time = time.time()
        files = list(crawler.crawl())  # 將生成器轉換為列表
        crawl_time = time.time() - start_time
        logger.info(f"文件爬取完成，共發現 {len(files)} 個文件，耗時 {crawl_time:.2f} 秒")
        
        if not files:
            logger.warning("未找到任何文件，請檢查 Q_DRIVE_PATH 設置")
            return
        
        # 創建文件索引器
        logger.info("創建文件索引器...")
        indexer = DocumentIndexer()
        
        # --- 加入進度續跑 ---
        processed_files = load_progress()
        files_to_process = [f for f in files if f not in processed_files]
        logger.info(f"剩餘待處理檔案數: {len(files_to_process)}")
        # --- 分批處理 ---
        batch_size = 1000
        total_files = len(files_to_process)
        for i in range(0, total_files, batch_size):
            batch = files_to_process[i:i+batch_size]
            logger.info(f"處理第 {i//batch_size+1} 批，共 {len(batch)} 個檔案")
            try:
                success_count, fail_count = indexer.index_files(batch, show_progress=True)
                logger.info(f"本批完成: {success_count} 成功, {fail_count} 失敗")
                processed_files.update(batch)
                save_progress(processed_files)
                gc.collect()
            except Exception as batch_e:
                logger.error(f"本批處理失敗: {str(batch_e)}")
                import traceback
                logger.error(traceback.format_exc())
        logger.info("全部批次處理完成")
        
        # 列出已索引的文件
        logger.info("正在獲取已索引文件列表...")
        indexed_files = indexer.list_indexed_files()
        if indexed_files:
            logger.info(f"\n已成功索引的文件（共 {len(indexed_files)} 個）:")
            for file_info in indexed_files[:10]:  # 只顯示前10個文件
                logger.info(f"文件: {file_info['file_path']}")
                logger.info(f"  - 類型: {file_info['file_type']}")
                logger.info(f"  - 大小: {file_info['file_size']} 字節")
                logger.info(f"  - 最後修改: {file_info['last_modified']}")
            
            if len(indexed_files) > 10:
                logger.info(f"... 以及其他 {len(indexed_files) - 10} 個文件")
        else:
            logger.warning("沒有找到任何已索引的文件")
        
        logger.info(f"初始化索引過程完成")
        
    except Exception as e:
        logger.error(f"初始化索引過程失敗: {str(e)}")
        import traceback
        logger.error(f"詳細錯誤信息: {traceback.format_exc()}")
        # sys.exit(1)  # 建議移除，讓外部監控自動重啟

if __name__ == "__main__":
    main()