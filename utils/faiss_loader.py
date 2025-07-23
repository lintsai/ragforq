"""
FAISS 載入器包裝器
處理 FAISS AVX2 支援問題，提供穩定的 FAISS 載入機制
"""
import logging
import warnings
import os

logger = logging.getLogger(__name__)

def safe_import_faiss():
    """
    安全導入 FAISS，處理 AVX2 支援問題
    
    Returns:
        faiss 模組
    """
    try:
        # 抑制 FAISS 的 AVX2 警告
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*AVX2.*")
            warnings.filterwarnings("ignore", message=".*swigfaiss_avx2.*")
            
            # 設置環境變量來禁用 AVX2 嘗試
            os.environ['FAISS_DISABLE_AVX2'] = '1'
            
            import faiss
            
            # 檢查 FAISS 是否成功載入
            logger.info(f"FAISS 載入成功，版本: {faiss.__version__}")
            
            # 檢查編譯選項
            try:
                compile_options = faiss.get_compile_options()
                logger.info(f"FAISS 編譯選項: {compile_options}")
            except Exception as e:
                logger.debug(f"無法獲取 FAISS 編譯選項: {str(e)}")
            
            # 檢查 GPU 支援
            try:
                gpu_available = hasattr(faiss, 'StandardGpuResources')
                logger.info(f"FAISS GPU 支援: {'可用' if gpu_available else '不可用'}")
            except Exception as e:
                logger.debug(f"檢查 GPU 支援時出錯: {str(e)}")
            
            return faiss
            
    except ImportError as e:
        logger.error(f"無法導入 FAISS: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"FAISS 載入時出現未預期錯誤: {str(e)}")
        raise

def patch_faiss_loader():
    """
    修補 FAISS 載入器，抑制 AVX2 相關的錯誤訊息
    """
    try:
        import faiss.loader
        
        # 保存原始的載入函數
        original_load_library = getattr(faiss.loader, '_load_library', None)
        
        if original_load_library:
            def patched_load_library(*args, **kwargs):
                try:
                    return original_load_library(*args, **kwargs)
                except Exception as e:
                    # 如果是 AVX2 相關錯誤，記錄但不拋出異常
                    if 'swigfaiss_avx2' in str(e) or 'AVX2' in str(e):
                        logger.debug(f"AVX2 載入失敗（預期行為）: {str(e)}")
                        return None
                    else:
                        raise
            
            # 替換載入函數
            faiss.loader._load_library = patched_load_library
            logger.debug("FAISS 載入器已修補，將抑制 AVX2 錯誤")
            
    except Exception as e:
        logger.debug(f"修補 FAISS 載入器時出錯: {str(e)}")

def initialize_faiss():
    """
    初始化 FAISS，處理所有相關的載入問題
    
    Returns:
        faiss 模組
    """
    try:
        # 首先修補載入器
        patch_faiss_loader()
        
        # 然後安全導入 FAISS
        faiss = safe_import_faiss()
        
        # 測試基本功能
        try:
            # 創建一個簡單的索引來測試功能
            import numpy as np
            test_vectors = np.random.random((10, 128)).astype('float32')
            index = faiss.IndexFlatL2(128)
            index.add(test_vectors)
            
            # 測試搜索
            query = np.random.random((1, 128)).astype('float32')
            distances, indices = index.search(query, 5)
            
            logger.info("FAISS 功能測試通過")
            
        except Exception as e:
            logger.warning(f"FAISS 功能測試失敗: {str(e)}")
            # 即使測試失敗，也返回 faiss 模組，因為可能只是測試問題
        
        return faiss
        
    except Exception as e:
        logger.error(f"FAISS 初始化失敗: {str(e)}")
        raise

# 全局 FAISS 實例
_faiss_instance = None

def get_faiss():
    """
    獲取 FAISS 實例（單例模式）
    
    Returns:
        faiss 模組
    """
    global _faiss_instance
    
    if _faiss_instance is None:
        _faiss_instance = initialize_faiss()
    
    return _faiss_instance

# 在模組載入時自動初始化
try:
    _faiss_instance = initialize_faiss()
    logger.info("FAISS 自動初始化成功")
except Exception as e:
    logger.error(f"FAISS 自動初始化失敗: {str(e)}")
    _faiss_instance = None
