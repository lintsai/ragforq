#!/usr/bin/env python
"""
日誌管理工具
"""
import os
import logging
from pathlib import Path
from datetime import datetime
import pytz
from config.config import LOGS_DIR

class TimezoneFormatter(logging.Formatter):
    """自定義格式化工具，以在特定時區中顯示日誌時間。"""
    def __init__(self, fmt=None, datefmt=None, tz='Asia/Taipei'):
        super().__init__(fmt, datefmt)
        self.tz = pytz.timezone(tz)

    def converter(self, timestamp):
        return datetime.fromtimestamp(timestamp, self.tz)

    def formatTime(self, record, datefmt=None):
        dt = self.converter(record.created)
        if datefmt:
            s = dt.strftime(datefmt)
        else:
            try:
                s = dt.isoformat(timespec='milliseconds')
            except TypeError:
                s = dt.isoformat()
        return s

def setup_model_logger(model_folder_name: str):
    """
    為指定的模型設置一個專用的日誌記錄器。

    這會將根日誌記錄器的輸出重定向到一個特定於模型的文件，
    位於 logs/{model_folder_name}.log。

    Args:
        model_folder_name (str): 模型的資料夾名稱，用於命名日誌文件。
    """
    log_directory = Path(LOGS_DIR)
    log_directory.mkdir(exist_ok=True)

    log_file_path = log_directory / f"{model_folder_name}.log"

    # 獲取根日誌記錄器
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # 移除所有現有的處理程序，以避免日誌重複或輸出到舊文件
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # 創建一個文件處理程序，將日誌寫入模型特定的文件
    file_handler = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
    
    # 創建一個時區感知的日誌格式器
    formatter = TimezoneFormatter(
        '%(asctime)s - %(levelname)s - %(module)s - %(message)s'
    )
    file_handler.setFormatter(formatter)

    # 將文件處理程序添加到根日誌記錄器
    root_logger.addHandler(file_handler)

    # 同時，為了方便在 supervisorctl logs 中查看，也添加一個控制台輸出
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    logging.info(f"日誌記錄器已設置，將輸出到 {log_file_path}")