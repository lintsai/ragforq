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

# 向量數據庫與路徑設置
VECTOR_DB_PATH = os.path.abspath(os.getenv("VECTOR_DB_PATH", "./vector_db").split("#")[0].strip().strip('"'))
LOGS_DIR = os.path.abspath(os.getenv("LOGS_DIR", "./logs").split("#")[0].strip().strip('"'))
BACKUPS_DIR = os.path.abspath(os.getenv("BACKUPS_DIR", "./backups").split("#")[0].strip().strip('"'))

# 文件類型設置
SUPPORTED_FILE_TYPES = os.getenv("SUPPORTED_FILE_TYPES", ".pdf,.docx,.xlsx,.txt,.md,.pptx,.csv").split(",")

# 應用設置
APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
APP_PORT = int(os.getenv("APP_PORT", "8000"))
STREAMLIT_PORT = int(os.getenv("STREAMLIT_PORT", "8501"))

# 其他設置
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
MAX_TOKENS_CHUNK = int(os.getenv("MAX_TOKENS_CHUNK", "500").split("#")[0].strip())
SIMILARITY_TOP_K = int(os.getenv("SIMILARITY_TOP_K", "10").split("#")[0].strip())

# Hugging Face 設置
HF_MODEL_CACHE_DIR = os.getenv("HF_MODEL_CACHE_DIR", "./models/cache")
HF_USE_GPU = os.getenv("HF_USE_GPU", "true").lower() == "true"

# Ollama 設置
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# 環境配置
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")  # development, production

# 用戶配置管理 - 平台和模型選擇通過前端設置流程管理
def load_user_config():
    """載入用戶配置"""
    config_file = Path("config/user_setup.json")
    if config_file.exists():
        try:
            import json
            with open(config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            pass
    return {}

def get_selected_platform():
    """獲取用戶選擇的平台"""
    user_config = load_user_config()
    return user_config.get("platform", "huggingface")  # 默認 Hugging Face

def detect_platform_from_model(model_name: str) -> str:
    """根據模型名稱自動檢測平台"""
    if model_name and ":" in model_name:
        # Ollama 模型格式 (例如: llama3.2:3b, qwen2:0.5b-instruct)
        return "ollama"
    else:
        # Hugging Face 模型格式 (例如: Qwen/Qwen2-0.5B-Instruct)
        return "huggingface"

def get_selected_models():
    """獲取用戶選擇的模型"""
    user_config = load_user_config()
    return {
        "language_model": user_config.get("language_model"),
        "embedding_model": user_config.get("embedding_model")
    }

def is_setup_completed():
    """檢查是否完成初始設置 - Web 應用模式總是返回 True"""
    user_config = load_user_config()
    return user_config.get("web_app_mode", True)  # Web 應用模式默認已完成設置

# 動態獲取當前配置
user_config = load_user_config()
SELECTED_PLATFORM = get_selected_platform()

# 模型配置（向後兼容）
DEFAULT_LLM_MODEL = os.getenv("DEFAULT_LLM_MODEL", "Qwen/Qwen2-0.5B-Instruct")
DEFAULT_EMBEDDING_MODEL = os.getenv("DEFAULT_EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# PyTorch 設置
TORCH_DEVICE = os.getenv("TORCH_DEVICE", "auto")  # auto, cpu, cuda
TORCH_DTYPE = os.getenv("TORCH_DTYPE", "float16")  # float16, float32

# TensorFlow 設置
TF_MEMORY_GROWTH = os.getenv("TF_MEMORY_GROWTH", "true").lower() == "true"
TF_MIXED_PRECISION = os.getenv("TF_MIXED_PRECISION", "true").lower() == "true"

# vLLM 設置
def get_inference_engine():
    """獲取用戶選擇的推理引擎"""
    user_config = load_user_config()
    return user_config.get("inference_engine", os.getenv("INFERENCE_ENGINE", "transformers"))

INFERENCE_ENGINE = get_inference_engine()
VLLM_GPU_MEMORY_UTILIZATION = float(os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.9"))
VLLM_MAX_MODEL_LEN = int(os.getenv("VLLM_MAX_MODEL_LEN", "4096"))
VLLM_TENSOR_PARALLEL_SIZE = int(os.getenv("VLLM_TENSOR_PARALLEL_SIZE", "1"))
VLLM_DTYPE = os.getenv("VLLM_DTYPE", "float16")  # float16, bfloat16, float32

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

# Ollama 超時設定
OLLAMA_REQUEST_TIMEOUT = int(os.getenv("OLLAMA_REQUEST_TIMEOUT", "300"))  # 請求超時（秒）- 增加到5分鐘
OLLAMA_EMBEDDING_TIMEOUT = int(os.getenv("OLLAMA_EMBEDDING_TIMEOUT", "180"))  # 嵌入超時（秒）- 增加到3分鐘
OLLAMA_QUERY_OPTIMIZATION_TIMEOUT = int(os.getenv("OLLAMA_QUERY_OPTIMIZATION_TIMEOUT", "90"))  # 查詢優化超時（秒）
OLLAMA_ANSWER_GENERATION_TIMEOUT = int(os.getenv("OLLAMA_ANSWER_GENERATION_TIMEOUT", "300"))  # 回答生成超時（秒）
OLLAMA_RELEVANCE_TIMEOUT = int(os.getenv("OLLAMA_RELEVANCE_TIMEOUT", "120"))  # 相關性分析超時（秒）
OLLAMA_CONNECTION_TIMEOUT = int(os.getenv("OLLAMA_CONNECTION_TIMEOUT", "60"))  # 連接超時（秒）

# 添加重試機制配置
OLLAMA_MAX_RETRIES = int(os.getenv("OLLAMA_MAX_RETRIES", "3"))  # 最大重試次數
OLLAMA_RETRY_DELAY = int(os.getenv("OLLAMA_RETRY_DELAY", "5"))  # 重試延遲（秒）

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
