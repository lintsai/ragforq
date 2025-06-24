#!/usr/bin/env python
"""
測試索引腳本 - 僅測試一小部分文件
"""

import os
import sys
import logging
import time
from pathlib import Path

# 添加項目根目錄到路徑
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

from indexer.file_crawler import FileCrawler
from indexer.document_indexer import DocumentIndexer
from config.config import Q_DRIVE_PATH, is_q_drive_accessible

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(project_root, 'logs', 'test_indexing.log'), mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    """主函數 - 執行測試索引"""
    
    # 確保logs目錄存在
    logs_dir = os.path.join(project_root, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    start_time = time.time()
    logger.info("開始執行測試索引過程")
    
    # 檢查Q槽是否可訪問
    if not is_q_drive_accessible():
        logger.error(f"無法訪問Q槽: {Q_DRIVE_PATH}")
        print(f"錯誤: 無法訪問Q槽: {Q_DRIVE_PATH}")
        print("請確保Q槽已掛載並且當前用戶有訪問權限。")
        return
    
    try:
        # 創建文件爬蟲
        logger.info("創建文件爬蟲...")
        crawler = FileCrawler(root_path=Q_DRIVE_PATH)
        
        # 爬取文件，但只處理前 3 個文件作為測試
        logger.info("開始爬取少量文件進行測試...")
        all_files = list(crawler.crawl())
        file_paths = all_files[:3]  # 只處理前 3 個文件
        
        if not file_paths:
            logger.warning(f"在Q槽未找到任何支持的文件。")
            print("警告: 未找到任何支持的文件。")
            return
        
        # 創建文件索引器
        logger.info("創建文件索引器...")
        indexer = DocumentIndexer()
        
        # 對每個文件單獨測試，以便於診斷
        for file_path in file_paths:
            try:
                logger.info(f"測試索引文件: {file_path}")
                success = indexer.index_file(file_path)
                if success:
                    logger.info(f"成功索引文件: {file_path}")
                else:
                    logger.error(f"索引文件失敗: {file_path}")
            except Exception as e:
                logger.error(f"處理文件時出錯: {file_path}, 錯誤: {str(e)}")
                import traceback
                logger.error(f"詳細錯誤: {traceback.format_exc()}")
        
        # 計算總耗時
        elapsed_time = time.time() - start_time
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        logger.info(f"測試索引完成! 耗時: {int(hours)}小時 {int(minutes)}分鐘 {int(seconds)}秒")
        
        print("\n測試索引過程完成!")
        print(f"測試了 {len(file_paths)} 個文件 (從總共 {len(all_files)} 個)")
        print(f"總耗時: {int(hours)}小時 {int(minutes)}分鐘 {int(seconds)}秒")
        
    except Exception as e:
        logger.error(f"測試索引過程中發生錯誤: {str(e)}")
        import traceback
        logger.error(f"詳細錯誤: {traceback.format_exc()}")
        print(f"錯誤: 測試索引過程中發生錯誤: {str(e)}")


if __name__ == "__main__":
    main() 