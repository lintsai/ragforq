import os
import logging
import time
from typing import List, Generator, Set, Tuple
from pathlib import Path
import sys

# 添加項目根目錄到路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import get_supported_file_extensions, Q_DRIVE_PATH

# 設置日誌
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 添加文件處理器，將日誌寫入到 indexing.log 文件
log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
os.makedirs(log_dir, exist_ok=True)
file_handler = logging.FileHandler(os.path.join(log_dir, "indexing.log"), encoding="utf-8")

# 設置日誌格式
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# 添加到 logger
logger.addHandler(file_handler)

class FileCrawler:
    """Q槽文件爬蟲，用於發現和跟蹤文件變化"""
    
    def __init__(self, root_path: str = Q_DRIVE_PATH, exclude_dirs: List[str] = None):
        """
        初始化文件爬蟲
        
        Args:
            root_path: 根目錄路徑，默認為Q槽路徑
            exclude_dirs: 要排除的目錄列表
        """
        self.root_path = root_path
        self.exclude_dirs = exclude_dirs or []
        self.supported_extensions = get_supported_file_extensions()
        
        # 檢查根目錄是否存在
        if not os.path.exists(self.root_path):
            logger.error(f"指定的根目錄不存在: {self.root_path}")
            raise FileNotFoundError(f"根目錄不存在: {self.root_path}")
    
    def crawl(self) -> Generator[str, None, None]:
        """
        爬取所有支持的文件
        
        Returns:
            生成器，逐個返回文件路徑
        """
        logger.info(f"開始爬取文件，根目錄: {self.root_path}")
        start_time = time.time()
        count = 0
        
        for root, dirs, files in os.walk(self.root_path):
            # 排除指定目錄
            dirs[:] = [d for d in dirs if d not in self.exclude_dirs and not d.startswith('.')]
            
            for file in files:
                # 排除臨時文件（以 ~$ 開頭的文件）
                if file.startswith('~$'):
                    continue
                    
                file_path = os.path.join(root, file)
                _, ext = os.path.splitext(file.lower())
                
                # 檢查文件擴展名是否支持
                if ext in self.supported_extensions:
                    count += 1
                    yield file_path
        
        elapsed_time = time.time() - start_time
        logger.info(f"文件爬取完成，共發現 {count} 個文件，耗時 {elapsed_time:.2f} 秒")
    
    def get_file_changes(self, previous_files: Set[str]) -> Tuple[Set[str], Set[str], Set[str]]:
        """
        檢測文件變更情況
        
        Args:
            previous_files: 之前的文件集合
            
        Returns:
            (新文件, 更新文件, 刪除文件) 的集合元組
        """
        current_files = set(self.crawl())
        
        # 獲取文件修改時間
        file_mtimes = {f: os.path.getmtime(f) for f in current_files if os.path.exists(f)}
        prev_file_mtimes = {f: os.path.getmtime(f) for f in previous_files if os.path.exists(f)}
        
        # 新文件: 之前不存在的文件
        new_files = current_files - previous_files
        
        # 更新文件: 修改時間改變的文件
        updated_files = {f for f in current_files & previous_files 
                        if f in file_mtimes and f in prev_file_mtimes 
                        and file_mtimes[f] > prev_file_mtimes[f]}
        
        # 刪除文件: 不再存在的文件
        deleted_files = previous_files - current_files
        
        logger.info(f"文件變化: {len(new_files)} 個新文件, {len(updated_files)} 個更新文件, {len(deleted_files)} 個刪除文件")
        
        return new_files, updated_files, deleted_files
    
    def monitor_changes(self, callback=None, interval_seconds: int = 3600):
        """
        持續監控文件變化
        
        Args:
            callback: 當檢測到文件變化時調用的回調函數，參數為 (新文件, 更新文件, 刪除文件)
            interval_seconds: 檢查間隔，默認為1小時
        """
        previous_files = set(self.crawl())
        logger.info(f"初始化文件監控，共 {len(previous_files)} 個文件")
        
        try:
            while True:
                time.sleep(interval_seconds)
                changes = self.get_file_changes(previous_files)
                
                if any(changes) and callback:
                    callback(*changes)
                
                # 更新文件列表
                previous_files = set(self.crawl())
                
        except KeyboardInterrupt:
            logger.info("文件監控已停止")


if __name__ == "__main__":
    # 測試代碼
    crawler = FileCrawler()
    for file_path in crawler.crawl():
        print(file_path) 