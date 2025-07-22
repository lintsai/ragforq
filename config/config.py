import os
import logging
from dotenv import load_dotenv
from pathlib import Path
from typing import List, Optional

# 加載環境變量
load_dotenv()

# 設置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Q槽路徑設置
Q_DRIVE_PATH = os.getenv("Q_DRIVE_PATH", "Q:")
Q_DRIVE_NETWORK_PATH = os.getenv("Q_DRIVE_NETWORK_PATH", "\\\\server\\Q")

# 向量數據庫設置
VECTOR_DB_PATH = os.path.abspath(os.getenv("VECTOR_DB_PATH", "./vector_db").split("#")[0].strip().strip('"'))

# 文件類型設置
SUPPORTED_FILE_TYPES = os.getenv("SUPPORTED_FILE_TYPES", ".pdf,.docx,.xlsx,.txt,.md,.pptx,.csv").split(",")

# 應用設置
APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
APP_PORT = int(os.getenv("APP_PORT", "8000"))
STREAMLIT_PORT = int(os.getenv("STREAMLIT_PORT", "8501"))

# 其他設置
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
MAX_TOKENS_CHUNK = int(os.getenv("MAX_TOKENS_CHUNK", "500").split("#")[0].strip())
SIMILARITY_TOP_K = int(os.getenv("SIMILARITY_TOP_K", "3").split("#")[0].strip())

# Ollama設置
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
# 注意：OLLAMA_MODEL 和 OLLAMA_EMBEDDING_MODEL 已移除，現在通過管理介面動態選擇
# 如果需要默認值，可以在相應的類中設定

# 建立向量數據庫目錄（如果不存在）
vector_db_path = Path(VECTOR_DB_PATH)
vector_db_path.mkdir(parents=True, exist_ok=True)

# API 基礎 URL
API_BASE_URL = f"http://{APP_HOST}:{APP_PORT}"

# 文件處理配置
MAX_TOKENS_CHUNK = int(os.getenv("MAX_TOKENS_CHUNK", "2000").split("#")[0].strip())  # 增加每個塊的大小
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200").split("#")[0].strip())  # 提高重疊性確保上下文連貫
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "0").split("#")[0].strip())  # 0表示自動決定

# 嵌入批處理大小 - 降低以減少內存使用
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "8").split("#")[0].strip())

# 文件索引批處理大小 - 降低以提高穩定性
FILE_BATCH_SIZE = int(os.getenv("FILE_BATCH_SIZE", "20").split("#")[0].strip())

# 檢查Q槽是否可訪問
def is_q_drive_accessible() -> bool:
    """檢查Q槽是否可訪問"""
    try:
        return os.path.exists(Q_DRIVE_PATH)
    except Exception as e:
        logger.error(f"檢查 Q 槽可訪問性時出錯: {str(e)}")
        return False

# 獲取支持的文件類型列表
def get_supported_file_extensions() -> List[str]:
    """獲取所有支持的文件擴展名"""
    return SUPPORTED_FILE_TYPES

DISPLAY_DRIVE_NAME = os.getenv("DISPLAY_DRIVE_NAME", "Q槽")
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "ragadmin123")
