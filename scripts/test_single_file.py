#!/usr/bin/env python
"""
測試單文件索引腳本 - 僅測試一個指定的文件
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

from indexer.document_indexer import DocumentIndexer
from utils.file_parsers import FileParser

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(project_root, 'logs', 'test_single_file.log'), mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    """主函數 - 執行單文件測試索引"""
    
    parser = argparse.ArgumentParser(description='測試單個文件的索引功能')
    parser.add_argument('file_path', help='要測試索引的文件路徑')
    args = parser.parse_args()
    
    file_path = args.file_path
    
    # 確保logs目錄存在
    logs_dir = os.path.join(project_root, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    if not os.path.exists(file_path):
        print(f"錯誤: 文件不存在: {file_path}")
        return
    
    start_time = time.time()
    print(f"開始測試索引單個文件: {file_path}")
    
    try:
        # 首先測試文件解析
        print("\n===== 測試文件解析 =====")
        parser = FileParser.get_parser_for_file(file_path)
        if not parser:
            print(f"錯誤: 不支持的文件類型: {file_path}")
            return
        
        print(f"使用解析器: {parser.__class__.__name__}")
        try:
            content_blocks = parser.parse(file_path)
            print(f"解析結果: 獲取了 {len(content_blocks)} 個內容塊")
            
            # 顯示第一個塊的一部分內容作為預覽
            if content_blocks:
                text, metadata = content_blocks[0]
                preview = text[:100] + "..." if len(text) > 100 else text
                print(f"內容預覽: {preview}")
                print(f"元數據: {metadata}")
            else:
                print("警告: 未提取到內容")
        except Exception as e:
            print(f"解析過程中出錯: {str(e)}")
            import traceback
            print(f"詳細錯誤:\n{traceback.format_exc()}")
        
        # 然後測試索引
        print("\n===== 測試文件索引 =====")
        indexer = DocumentIndexer()
        
        # 嘗試索引文件
        try:
            success = indexer.index_file(file_path)
            if success:
                print(f"成功索引文件: {file_path}")
            else:
                print(f"索引文件失敗: {file_path}")
        except Exception as e:
            print(f"索引過程中出錯: {str(e)}")
            import traceback
            print(f"詳細錯誤:\n{traceback.format_exc()}")
        
        # 計算耗時
        elapsed_time = time.time() - start_time
        minutes, seconds = divmod(elapsed_time, 60)
        
        print(f"\n測試完成! 總耗時: {int(minutes)}分鐘 {int(seconds)}秒")
        
    except Exception as e:
        print(f"測試過程中出錯: {str(e)}")
        import traceback
        print(f"詳細錯誤:\n{traceback.format_exc()}")


if __name__ == "__main__":
    main() 