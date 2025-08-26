#!/usr/bin/env python
"""
重新索引腳本 - 清除現有索引並重新建立
"""

import os
import sys
import logging
import time
import argparse
import shutil
from pathlib import Path

# 添加項目根目錄到路徑
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

from indexer.file_crawler import FileCrawler
from indexer.document_indexer import DocumentIndexer
from config.config import Q_DRIVE_PATH, VECTOR_DB_PATH, is_q_drive_accessible
from utils.state_manager import record_full_index

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(project_root, 'logs', 'reindex.log'), mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def clear_vector_db():
    """清除現有向量數據庫"""
    try:
        if os.path.exists(VECTOR_DB_PATH):
            logger.info(f"清除現有向量數據庫: {VECTOR_DB_PATH}")
            shutil.rmtree(VECTOR_DB_PATH)
            os.makedirs(VECTOR_DB_PATH, exist_ok=True)
            logger.info("向量數據庫目錄已重新創建")
            return True
        else:
            logger.info(f"向量數據庫目錄不存在，將創建: {VECTOR_DB_PATH}")
            os.makedirs(VECTOR_DB_PATH, exist_ok=True)
            return True
    except Exception as e:
        logger.error(f"清除向量數據庫時出錯: {str(e)}")
        return False

def main():
    """主函數 - 執行重新索引"""
    
    # 解析命令行參數
    parser = argparse.ArgumentParser(description='清除現有索引並重新建立')
    parser.add_argument('--skip-clear', action='store_true', 
                        help='跳過清除現有索引步驟')
    parser.add_argument('--exclude-dirs', type=str, nargs='*', default=[],
                        help='排除的目錄列表')
    args = parser.parse_args()
    
    # 確保logs目錄存在
    logs_dir = os.path.join(project_root, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    start_time = time.time()
    logger.info("開始執行重新索引過程")
    
    # 檢查Q槽是否可訪問
    if not is_q_drive_accessible():
        logger.error(f"無法訪問Q槽: {Q_DRIVE_PATH}")
        print(f"錯誤: 無法訪問Q槽: {Q_DRIVE_PATH}")
        print("請確保Q槽已掛載並且當前用戶有訪問權限。")
        return
    
    try:
        # 清除現有索引
        if not args.skip_clear:
            if not clear_vector_db():
                print("錯誤: 無法清除現有索引。請檢查日誌獲取詳細信息。")
                return
            print("已清除現有索引")
        else:
            print("跳過清除索引步驟")
        
        # 創建文件爬蟲
        logger.info(f"創建文件爬蟲，排除目錄: {args.exclude_dirs}...")
        crawler = FileCrawler(root_path=Q_DRIVE_PATH, exclude_dirs=args.exclude_dirs)
        
        # 爬取文件
        logger.info("開始爬取文件...")
        file_paths = list(crawler.crawl())
        
        if not file_paths:
            logger.warning(f"在Q槽未找到任何支持的文件。")
            print("警告: 未找到任何支持的文件。")
            return
        
        # 創建文件索引器
        logger.info("創建文件索引器...")
        indexer = DocumentIndexer()
        
        # 索引文件
        logger.info(f"開始索引 {len(file_paths)} 個文件...")
        success_count, fail_count = indexer.index_files(file_paths)
        try:
            record_full_index(success_count)
            logger.info(f"State updated with full reindex count={success_count}")
        except Exception as e:
            logger.warning(f"Failed to record full index state: {e}")
        
        # 計算總耗時
        elapsed_time = time.time() - start_time
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        logger.info(f"重新索引完成! 成功: {success_count}, 失敗: {fail_count}, 耗時: {int(hours)}小時 {int(minutes)}分鐘 {int(seconds)}秒")
        
        print("\n重新索引過程完成!")
        print(f"總共發現: {len(file_paths)} 個文件")
        print(f"成功索引: {success_count} 個文件")
        print(f"索引失敗: {fail_count} 個文件")
        print(f"總耗時: {int(hours)}小時 {int(minutes)}分鐘 {int(seconds)}秒")
        
    except Exception as e:
        logger.error(f"重新索引過程中發生錯誤: {str(e)}")
        print(f"錯誤: 重新索引過程中發生錯誤: {str(e)}")


if __name__ == "__main__":
    main() 