#!/usr/bin/env python
"""
文件監控腳本 - 持續監控Q槽文件變化並更新索引
"""

import os
import sys
import logging
import time
import argparse
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
        logging.FileHandler(os.path.join(project_root, 'logs', 'monitor.log'), mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def file_changes_callback(new_files, updated_files, deleted_files):
    """
    文件變更回調函數
    
    Args:
        new_files: 新文件集合
        updated_files: 更新文件集合
        deleted_files: 刪除文件集合
    """
    if not (new_files or updated_files or deleted_files):
        logger.info("未檢測到文件變更")
        return
    
    logger.info(f"檢測到文件變更: {len(new_files)} 個新文件, {len(updated_files)} 個更新文件, {len(deleted_files)} 個刪除文件")
    
    try:
        # 創建文件索引器
        indexer = DocumentIndexer()
        
        # 處理文件變更
        indexer.process_file_changes(new_files, updated_files, deleted_files)
        
        logger.info("文件索引已更新")
    
    except Exception as e:
        logger.error(f"處理文件變更時出錯: {str(e)}")


def main():
    """主函數 - 執行文件監控"""
    
    # 解析命令行參數
    parser = argparse.ArgumentParser(description='監控Q槽文件變化並更新索引')
    parser.add_argument('--interval', type=int, default=3600, 
                        help='檢查間隔（秒），默認為3600秒（1小時）')
    args = parser.parse_args()
    
    # 確保logs目錄存在
    logs_dir = os.path.join(project_root, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    logger.info(f"開始監控Q槽文件變化，檢查間隔: {args.interval} 秒")
    print(f"開始監控Q槽文件變化，檢查間隔: {args.interval} 秒")
    print("按 Ctrl+C 停止監控")
    
    # 檢查Q槽是否可訪問
    if not is_q_drive_accessible():
        logger.error(f"無法訪問Q槽: {Q_DRIVE_PATH}")
        print(f"錯誤: 無法訪問Q槽: {Q_DRIVE_PATH}")
        print("請確保Q槽已掛載並且當前用戶有訪問權限。")
        return
    
    try:
        # 創建文件爬蟲
        crawler = FileCrawler(root_path=Q_DRIVE_PATH)
        
        # 啟動監控
        crawler.monitor_changes(callback=file_changes_callback, interval_seconds=args.interval)
        
    except KeyboardInterrupt:
        logger.info("監控已被用戶終止")
        print("\n監控已停止")
    
    except Exception as e:
        logger.error(f"監控過程中發生錯誤: {str(e)}")
        print(f"錯誤: 監控過程中發生錯誤: {str(e)}")


if __name__ == "__main__":
    main() 