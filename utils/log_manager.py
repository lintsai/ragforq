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

_INIT_FLAG_ATTR = "_rag_logging_initialized"

def init_logging(level: int = logging.INFO, log_file: str | None = None, console: bool = True, force: bool = False, tz: str = 'Asia/Taipei'):
    """Idempotent root logger initializer to avoid duplicate logs.

    Args:
        level: logging level for root logger
        log_file: optional path to log file; if relative, placed under LOGS_DIR
        console: whether to add a console stream handler
        force: if True, remove existing handlers even if already initialized
        tz: timezone for timestamps
    """
    root = logging.getLogger()
    if getattr(root, _INIT_FLAG_ATTR, False) and not force:
        # Already initialized; still allow elevating level if caller requests higher verbosity
        if level < root.level:
            root.setLevel(level)
        return

    # Remove handlers when forcing or when we detect multiple stream handlers (cleanup scenario)
    if force or len(root.handlers) > 1:
        for h in root.handlers[:]:
            root.removeHandler(h)

    root.setLevel(level)
    formatter = TimezoneFormatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s', tz=tz)

    # File handler
    if log_file:
        try:
            # If not absolute, resolve under LOGS_DIR
            if not os.path.isabs(log_file):
                from config.config import LOGS_DIR  # lazy import to avoid circular at module load
                os.makedirs(LOGS_DIR, exist_ok=True)
                log_file = os.path.join(LOGS_DIR, log_file)
            else:
                os.makedirs(os.path.dirname(log_file), exist_ok=True)
            # Avoid adding duplicate file handler pointing to same file
            already = False
            for h in root.handlers:
                if isinstance(h, logging.FileHandler) and getattr(h, 'baseFilename', None) == os.path.abspath(log_file):
                    already = True
                    break
            if not already:
                fh = logging.FileHandler(log_file, mode='a', encoding='utf-8')
                fh.setFormatter(formatter)
                root.addHandler(fh)
        except Exception as e:
            # Fallback to console only if file handler fails
            logging.getLogger(__name__).warning(f"初始化文件日誌失敗: {e}")

    if console:
        # Avoid duplicate consoles (keep only one StreamHandler)
        if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
            sh = logging.StreamHandler()
            sh.setFormatter(formatter)
            root.addHandler(sh)

    setattr(root, _INIT_FLAG_ATTR, True)

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