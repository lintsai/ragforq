"""
RAG系統優化配置
解決傳統RAG和動態RAG的常見問題
"""

import os
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class RAGOptimizationConfig:
    """RAG系統優化配置類"""
    
    def __init__(self):
        self.config = self._load_optimization_config()
    
    def _load_optimization_config(self) -> Dict[str, Any]:
        """載入優化配置"""
        return {
            # 索引優化配置
            "indexing": {
                "batch_size": int(os.getenv("OPTIMIZED_BATCH_SIZE", "10")),  # 減少批次大小避免超時
                "max_file_size_mb": int(os.getenv("MAX_FILE_SIZE_MB", "50")),  # 最大文件大小限制
                "encoding_detection": True,  # 啟用編碼檢測
                "skip_corrupted_files": True,  # 跳過損壞文件
                "resume_on_failure": True,  # 支持斷點續傳
            },
            
            # 超時優化配置
            "timeout": {
                "connection_timeout": 60,  # 連接超時
                "request_timeout": 300,    # 請求超時（5分鐘）
                "embedding_timeout": 180,  # 嵌入超時（3分鐘）
                "max_retries": 3,          # 最大重試次數
                "retry_delay": 5,          # 重試延遲（秒）
            },
            
            # 相關性優化配置
            "relevance": {
                "dynamic_threshold": True,     # 使用動態閾值
                "file_deduplication": True,    # 文件去重
                "min_similarity_score": 0.3,  # 最小相似度分數
                "max_documents": 8,           # 最大文檔數量
                "context_window": 4000,       # 上下文窗口大小
            },
            
            # 動態RAG優化配置
            "dynamic_rag": {
                "max_scan_files": 5000,       # 最大掃描文件數
                "scan_depth": 8,              # 掃描深度
                "cache_duration": 300,        # 緩存持續時間（秒）
                "priority_directories": [     # 優先掃描目錄關鍵詞
                    "文件", "資料", "檔案", "document", "data", "file", "共用", "share"
                ],
                "enhanced_query_rewrite": True,  # 增強查詢重寫
            },
            
            # 編碼處理配置
            "encoding": {
                "auto_detect": True,          # 自動檢測編碼
                "fallback_encodings": [       # 備用編碼列表
                    "utf-8", "big5", "gb18030", "gb2312", "latin-1", "cp1252", "utf-16"
                ],
                "use_chardet": True,          # 使用chardet庫
                "clean_garbled_text": True,   # 清理亂碼
            },
            
            # 性能優化配置
            "performance": {
                "parallel_processing": True,   # 並行處理
                "memory_optimization": True,   # 內存優化
                "progress_tracking": True,     # 進度追蹤
                "error_recovery": True,        # 錯誤恢復
            }
        }
    
    def get_config(self, section: str = None) -> Dict[str, Any]:
        """
        獲取配置
        
        Args:
            section: 配置節名稱，如果為None則返回全部配置
            
        Returns:
            配置字典
        """
        if section:
            return self.config.get(section, {})
        return self.config
    
    def update_config(self, section: str, key: str, value: Any):
        """
        更新配置
        
        Args:
            section: 配置節名稱
            key: 配置鍵
            value: 配置值
        """
        if section not in self.config:
            self.config[section] = {}
        self.config[section][key] = value
        logger.info(f"更新配置: {section}.{key} = {value}")
    
    def apply_optimizations(self):
        """應用優化配置到環境變量"""
        try:
            # 應用超時配置
            timeout_config = self.get_config("timeout")
            os.environ["OLLAMA_CONNECTION_TIMEOUT"] = str(timeout_config["connection_timeout"])
            os.environ["OLLAMA_REQUEST_TIMEOUT"] = str(timeout_config["request_timeout"])
            os.environ["OLLAMA_EMBEDDING_TIMEOUT"] = str(timeout_config["embedding_timeout"])
            os.environ["OLLAMA_MAX_RETRIES"] = str(timeout_config["max_retries"])
            os.environ["OLLAMA_RETRY_DELAY"] = str(timeout_config["retry_delay"])
            
            # 應用索引配置
            indexing_config = self.get_config("indexing")
            os.environ["FILE_BATCH_SIZE"] = str(indexing_config["batch_size"])
            os.environ["MAX_FILE_SIZE_MB"] = str(indexing_config["max_file_size_mb"])
            
            # 應用動態RAG配置
            dynamic_config = self.get_config("dynamic_rag")
            os.environ["DYNAMIC_MAX_SCAN_FILES"] = str(dynamic_config["max_scan_files"])
            os.environ["DYNAMIC_SCAN_DEPTH"] = str(dynamic_config["scan_depth"])
            
            logger.info("RAG優化配置已應用到環境變量")
            
        except Exception as e:
            logger.error(f"應用優化配置時出錯: {str(e)}")

# 創建全局優化配置實例
rag_optimization = RAGOptimizationConfig()

def get_optimization_config(section: str = None) -> Dict[str, Any]:
    """
    便捷函數：獲取優化配置
    
    Args:
        section: 配置節名稱
        
    Returns:
        配置字典
    """
    return rag_optimization.get_config(section)

def apply_rag_optimizations():
    """
    便捷函數：應用RAG優化配置
    """
    rag_optimization.apply_optimizations()

# 自動應用優化配置
if __name__ != "__main__":
    apply_rag_optimizations()