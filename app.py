#!/usr/bin/env python
"""
主應用入口點 - 啟動API服務並初始化必要組件
"""

import os
import sys
import logging
import argparse
import uvicorn
from pathlib import Path

# 設置項目根目錄
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from config.config import APP_HOST, APP_PORT, is_q_drive_accessible
from indexer.file_crawler import FileCrawler
from indexer.document_indexer import DocumentIndexer

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(project_root, 'logs', 'app.log'), mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def check_environment():
    """檢查運行環境"""
    
    # 確保logs目錄存在
    logs_dir = os.path.join(project_root, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    # 檢查Q槽是否可訪問
    logger.info("檢查Q槽訪問權限...")
    if not is_q_drive_accessible():
        logger.warning("無法訪問Q槽。API將正常啟動，但可能無法檢索文件。")
        print("警告: 無法訪問Q槽。API將正常啟動，但可能無法檢索文件。")
    else:
        logger.info("Q槽訪問正常")

def main():
    """主函數 - 啟動應用"""
    
    # 解析命令行參數
    parser = argparse.ArgumentParser(description='啟動Q槽文件智能助手')
    parser.add_argument('--host', type=str, default=APP_HOST, 
                        help=f'API主機地址，默認為 {APP_HOST}')
    parser.add_argument('--port', type=int, default=APP_PORT, 
                        help=f'API端口，默認為 {APP_PORT}')
    parser.add_argument('--index-first', action='store_true',
                        help='啟動前先執行初始索引（如果沒有現有索引）')
    args = parser.parse_args()
    
    # 檢查環境
    check_environment()
    
    # 如果指定了先索引選項，且沒有現有索引，則執行初始索引
    if args.index_first:
        from config.config import VECTOR_DB_PATH
        faiss_index_path = os.path.join(VECTOR_DB_PATH, "faiss_index")
        
        if not os.path.exists(faiss_index_path):
            logger.info("未找到現有索引，正在執行初始索引...")
            print("未找到現有索引，正在執行初始索引...")
            
            try:
                # 導入並執行初始索引腳本
                from scripts.initial_indexing import main as run_indexing
                run_indexing()
            except Exception as e:
                logger.error(f"初始索引失敗: {str(e)}")
                print(f"警告: 初始索引失敗: {str(e)}")
                print("系統將繼續啟動，但搜索功能可能不完整。")
    
    # 啟動API服務
    logger.info(f"啟動API服務，地址: {args.host}:{args.port}")
    print(f"\n啟動Q槽文件智能助手 API 服務...")
    print(f"API 地址: http://{args.host}:{args.port}")
    print(f"API 文檔: http://{args.host}:{args.port}/docs")
    print("\n按 Ctrl+C 停止服務")
    
    # 啟動uvicorn服務器
    uvicorn.run("api.main:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main() 