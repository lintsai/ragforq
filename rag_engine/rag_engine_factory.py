import os
import sys
import logging
from typing import Dict, Optional
from pathlib import Path

# 添加項目根目錄到路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_engine.interfaces import RAGEngineInterface
from rag_engine.traditional_chinese_engine import TraditionalChineseRAGEngine
from rag_engine.simplified_chinese_engine import SimplifiedChineseRAGEngine
from rag_engine.english_engine import EnglishRAGEngine
from rag_engine.thai_engine import ThaiRAGEngine
from rag_engine.dynamic_traditional_chinese_engine import DynamicTraditionalChineseRAGEngine
from rag_engine.dynamic_simplified_chinese_engine import DynamicSimplifiedChineseRAGEngine
from rag_engine.dynamic_english_engine import DynamicEnglishRAGEngine
from rag_engine.dynamic_thai_engine import DynamicThaiRAGEngine

# 設置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGEngineFactory:
    """RAG引擎工廠類，負責創建和管理不同語言的RAG引擎"""
    
    # 支持的語言映射
    LANGUAGE_MAPPING = {
        "繁體中文": "traditional_chinese",
        "中文": "traditional_chinese",
        "简体中文": "simplified_chinese",
        "簡體中文": "simplified_chinese",
        "English": "english",
        "english": "english",
        "泰文": "thai",
        "ไทย": "thai",
        "thai": "thai",
        "Thai": "thai",
        # 動態鍵保留以兼容舊用法，但不再依賴於此固定映射
        "Dynamic_繁體中文": "dynamic_traditional_chinese",
        "Dynamic_简体中文": "dynamic_simplified_chinese",
        "Dynamic_English": "dynamic_english",
        "Dynamic_ไทย": "dynamic_thai",
    }
    
    # 引擎類映射
    ENGINE_CLASSES = {
        "traditional_chinese": TraditionalChineseRAGEngine,
        "simplified_chinese": SimplifiedChineseRAGEngine,
        "english": EnglishRAGEngine,
        "thai": ThaiRAGEngine,
        "dynamic_traditional_chinese": DynamicTraditionalChineseRAGEngine,
        "dynamic_simplified_chinese": DynamicSimplifiedChineseRAGEngine,
        "dynamic_english": DynamicEnglishRAGEngine,
        "dynamic_thai": DynamicThaiRAGEngine,
    }
    
    def __init__(self):
        """初始化工廠"""
        self._engines: Dict[str, Dict[str, RAGEngineInterface]] = {}
        logger.info("RAG引擎工廠初始化完成")
    
    def get_supported_languages(self) -> list:
        """獲取支持的語言列表"""
        return list(self.LANGUAGE_MAPPING.keys())
    
    def normalize_language(self, language: str) -> str:
        """
        標準化語言名稱
        
        Args:
            language: 輸入的語言名稱
            
        Returns:
            標準化的語言標識符
        """
        # 先處理動態前綴，將基礎語言先標準化後再映射到對應動態鍵
        if isinstance(language, str) and language.startswith("Dynamic_"):
            base_lang = language[len("Dynamic_"):]
            base_normalized = self.LANGUAGE_MAPPING.get(base_lang)
            if not base_normalized:
                logger.warning(f"不支持的語言: {language}，默認使用dynamic_traditional_chinese")
                return "dynamic_traditional_chinese"
            return f"dynamic_{base_normalized}"
        
        normalized = self.LANGUAGE_MAPPING.get(language)
        if not normalized:
            logger.warning(f"不支持的語言: {language}，默認使用繁體中文")
            return "traditional_chinese"
        return normalized
    
    def create_engine(self, language: str, document_indexer, ollama_model: str, ollama_embedding_model: str = None, platform: str = None, folder_path: Optional[str] = None) -> RAGEngineInterface:
        """
        創建指定語言的RAG引擎
        
        Args:
            language: 目標語言
            document_indexer: 文檔索引器實例
            ollama_model: Ollama模型名稱
            ollama_embedding_model: Ollama嵌入模型名稱（Dynamic RAG需要）
            platform: AI平台類型 (huggingface/ollama)，如果為None則自動檢測
            
        Returns:
            對應語言的RAG引擎實例
        """
        # 如果沒有指定平台，自動檢測
        if platform is None:
            from config.config import detect_platform_from_model
            platform = detect_platform_from_model(ollama_model)
            
        normalized_lang = self.normalize_language(language)
        engine_class = self.ENGINE_CLASSES[normalized_lang]
        
        logger.info(f"創建{language}({normalized_lang})RAG引擎，使用模型: {ollama_model}，平台: {platform}")
        
        try:
            if normalized_lang.startswith("dynamic_"):
                # 動態RAG引擎，支持文件夾路徑
                engine = engine_class(ollama_model, ollama_embedding_model, platform=platform, folder_path=folder_path)
            else:
                # 傳統RAG引擎
                engine = engine_class(document_indexer, ollama_model, platform)
            
            logger.info(f"{language}RAG引擎創建成功")
            return engine
        except Exception as e:
            logger.error(f"創建{language}RAG引擎失敗: {str(e)}")
            raise
    
    def get_engine(self, language: str, document_indexer, ollama_model: str, ollama_embedding_model: str = None, platform: str = None, folder_path: Optional[str] = None) -> RAGEngineInterface:
        """
        獲取或創建指定語言的RAG引擎（帶緩存）
        
        Args:
            language: 目標語言
            document_indexer: 文檔索引器實例
            ollama_model: Ollama模型名稱
            ollama_embedding_model: Ollama嵌入模型名稱（Dynamic RAG需要）
            platform: AI平台類型 (huggingface/ollama)，如果為None則自動檢測
            
        Returns:
            對應語言的RAG引擎實例
        """
        # 如果沒有指定平台，自動檢測
        if platform is None:
            from config.config import detect_platform_from_model
            platform = detect_platform_from_model(ollama_model)
            
        normalized_lang = self.normalize_language(language)
        
        # 創建緩存鍵，只有動態 RAG 才包含文件夾路徑
        if normalized_lang.startswith("dynamic_"):
            folder_key = f"_{folder_path}" if folder_path else ""
            cache_key = f"{normalized_lang}_{ollama_model}_{ollama_embedding_model}_{platform}{folder_key}"
        else:
            # 傳統 RAG 不使用文件夾路徑，忽略 folder_path 參數
            cache_key = f"{normalized_lang}_{ollama_model}_{platform}"
        
        # 檢查緩存
        if normalized_lang not in self._engines:
            self._engines[normalized_lang] = {}
        
        if cache_key not in self._engines[normalized_lang]:
            # 創建新引擎
            self._engines[normalized_lang][cache_key] = self.create_engine(
                language, document_indexer, ollama_model, ollama_embedding_model, platform, folder_path
            )
        
        return self._engines[normalized_lang][cache_key]
    
    def clear_cache(self, language: Optional[str] = None, ollama_model: Optional[str] = None):
        """
        清理引擎緩存
        
        Args:
            language: 指定語言，如果為None則清理所有語言
            ollama_model: 指定模型，如果為None則清理所有模型
        """
        if language is None:
            # 清理所有緩存
            self._engines.clear()
            logger.info("已清理所有RAG引擎緩存")
        else:
            normalized_lang = self.normalize_language(language)
            if normalized_lang in self._engines:
                if ollama_model is None:
                    # 清理指定語言的所有模型
                    del self._engines[normalized_lang]
                    logger.info(f"已清理{language}的所有RAG引擎緩存")
                else:
                    # 清理指定語言和模型的緩存
                    cache_key = f"{normalized_lang}_{ollama_model}"
                    if cache_key in self._engines[normalized_lang]:
                        del self._engines[normalized_lang][cache_key]
                        logger.info(f"已清理{language}({ollama_model})的RAG引擎緩存")
    
    def get_cache_info(self) -> Dict[str, list]:
        """
        獲取緩存信息
        
        Returns:
            緩存信息字典
        """
        cache_info = {}
        for lang, models in self._engines.items():
            cache_info[lang] = list(models.keys())
        return cache_info
    
    def validate_language_support(self, language: str) -> bool:
        """
        驗證是否支持指定語言
        
        Args:
            language: 要驗證的語言
            
        Returns:
            是否支持該語言
        """
        return language in self.LANGUAGE_MAPPING

# 創建全局工廠實例
rag_engine_factory = RAGEngineFactory()

def get_rag_engine_for_language(language: str, document_indexer, ollama_model: str, ollama_embedding_model: str = None, platform: str = None, folder_path: Optional[str] = None) -> RAGEngineInterface:
    """
    便捷函數：獲取指定語言的RAG引擎
    
    Args:
        language: 目標語言
        document_indexer: 文檔索引器實例
        ollama_model: Ollama模型名稱
        ollama_embedding_model: Ollama嵌入模型名稱（Dynamic RAG需要）
        platform: AI平台類型 (huggingface/ollama)，如果為None則自動檢測
        
    Returns:
        對應語言的RAG引擎實例
    """
    # 如果沒有指定平台，自動檢測
    if platform is None:
        from config.config import detect_platform_from_model
        platform = detect_platform_from_model(ollama_model)
    
    return rag_engine_factory.get_engine(language, document_indexer, ollama_model, ollama_embedding_model, platform, folder_path)

def clear_rag_engine_cache(language: Optional[str] = None, ollama_model: Optional[str] = None):
    """
    便捷函數：清理RAG引擎緩存
    
    Args:
        language: 指定語言，如果為None則清理所有語言
        ollama_model: 指定模型，如果為None則清理所有模型
    """
    rag_engine_factory.clear_cache(language, ollama_model)

def get_supported_languages() -> list:
    """
    便捷函數：獲取支持的語言列表
    
    Returns:
        支持的語言列表
    """
    return rag_engine_factory.get_supported_languages()

def validate_language(language: str) -> bool:
    """
    便捷函數：驗證語言支持
    
    Args:
        language: 要驗證的語言
        
    Returns:
        是否支持該語言
    """
    return rag_engine_factory.validate_language_support(language)
