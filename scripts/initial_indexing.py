#!/usr/bin/env python
"""
初始化索引腳本 - 爬取Q槽文件並建立索引
"""

import os
import sys
import logging
import time
from pathlib import Path

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
        
        # 開始索引文件 (使用並行處理)
        total_files = len(files)
        logger.info(f"開始並行索引 {total_files} 個文件...")
        
        # 使用索引器的並行處理功能
        start_index_time = time.time()
        success_count, fail_count = indexer.index_files(files, show_progress=True)
        index_time = time.time() - start_index_time
        
        # 計算處理速度
        files_per_second = total_files / index_time if index_time > 0 else 0
        logger.info(f"索引完成，總耗時 {index_time:.2f} 秒，處理速度 {files_per_second:.2f} 文件/秒")
        
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
        
        logger.info(f"初始化索引過程完成: {success_count} 個成功, {fail_count} 個失敗")
        
    except Exception as e:
        logger.error(f"初始化索引過程失敗: {str(e)}")
        import traceback
        logger.error(f"詳細錯誤信息: {traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    main()