#!/usr/bin/env python
"""
索引診斷腳本 - 檢查索引狀態和問題
"""

import os
import sys
import json
import logging
from pathlib import Path

# 添加項目根目錄到路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from indexer.document_indexer import DocumentIndexer
from config.config import Q_DRIVE_PATH, VECTOR_DB_PATH

# 設置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_vector_db_files():
    """檢查向量數據庫文件"""
    logger.info("=" * 50)
    logger.info("檢查向量數據庫文件")
    logger.info("=" * 50)
    
    vector_db_path = Path(VECTOR_DB_PATH)
    
    if not vector_db_path.exists():
        logger.error(f"向量數據庫目錄不存在: {vector_db_path}")
        return False
    
    # 檢查關鍵文件
    files_to_check = [
        "index.faiss",
        "index.pkl", 
        "indexed_files.pkl",
        "indexing_progress.json"
    ]
    
    for filename in files_to_check:
        file_path = vector_db_path / filename
        if file_path.exists():
            size = file_path.stat().st_size
            logger.info(f"✓ {filename}: {size:,} bytes")
        else:
            logger.warning(f"✗ {filename}: 文件不存在")
    
    return True

def check_indexing_progress():
    """檢查索引進度"""
    logger.info("=" * 50)
    logger.info("檢查索引進度")
    logger.info("=" * 50)
    
    try:
        indexer = DocumentIndexer()
        progress = indexer.indexing_progress
        
        logger.info(f"索引進度信息:")
        logger.info(f"  進行中: {progress.get('in_progress', False)}")
        logger.info(f"  已完成批次: {progress.get('completed_batches', 0)}")
        logger.info(f"  總批次數: {progress.get('total_batches', 0)}")
        logger.info(f"  待處理文件: {len(progress.get('pending_files', []))}")
        logger.info(f"  當前批次文件: {len(progress.get('current_batch', []))}")
        
        # 檢查已索引文件
        indexed_count = len(indexer.indexed_files)
        logger.info(f"  已索引文件數: {indexed_count:,}")
        
        # 檢查向量數據庫
        if indexer.vector_store:
            vector_count = len(indexer.vector_store.docstore._dict)
            logger.info(f"  向量數據庫文檔數: {vector_count:,}")
        else:
            logger.warning("  向量數據庫未加載")
        
        return True
        
    except Exception as e:
        logger.error(f"檢查索引進度時出錯: {str(e)}")
        return False

def check_system_status():
    """檢查系統狀態"""
    logger.info("=" * 50)
    logger.info("檢查系統狀態")
    logger.info("=" * 50)
    
    # 檢查 Q 槽訪問
    try:
        if os.path.exists(Q_DRIVE_PATH):
            logger.info(f"✓ Q 槽可訪問: {Q_DRIVE_PATH}")
            # 嘗試列出一些文件
            items = os.listdir(Q_DRIVE_PATH)
            logger.info(f"  Q 槽包含 {len(items)} 個項目")
        else:
            logger.error(f"✗ Q 槽不可訪問: {Q_DRIVE_PATH}")
    except Exception as e:
        logger.error(f"✗ Q 槽訪問錯誤: {str(e)}")
    
    # 檢查 Ollama 服務
    try:
        import requests
        from config.config import OLLAMA_HOST
        
        response = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        if response.status_code == 200:
            logger.info(f"✓ Ollama 服務正常: {OLLAMA_HOST}")
        else:
            logger.warning(f"⚠ Ollama 服務響應異常: {response.status_code}")
    except Exception as e:
        logger.error(f"✗ Ollama 服務檢查失敗: {str(e)}")
    
    # 檢查系統資源
    try:
        import psutil
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        logger.info(f"系統資源:")
        logger.info(f"  內存: {memory.percent:.1f}% 使用 ({memory.used/1024/1024/1024:.1f}GB/{memory.total/1024/1024/1024:.1f}GB)")
        logger.info(f"  磁盤: {disk.percent:.1f}% 使用 ({disk.used/1024/1024/1024:.1f}GB/{disk.total/1024/1024/1024:.1f}GB)")
        
        if memory.percent > 85:
            logger.warning("⚠ 內存使用率較高")
        if disk.percent > 90:
            logger.warning("⚠ 磁盤空間不足")
            
    except ImportError:
        logger.info("未安裝 psutil，跳過系統資源檢查")
    except Exception as e:
        logger.error(f"系統資源檢查失敗: {str(e)}")

def analyze_log_file():
    """分析日誌文件"""
    logger.info("=" * 50)
    logger.info("分析日誌文件")
    logger.info("=" * 50)
    
    log_file = Path("logs/indexing.log")
    
    if not log_file.exists():
        logger.warning("索引日誌文件不存在")
        return
    
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        logger.info(f"日誌文件大小: {log_file.stat().st_size:,} bytes")
        logger.info(f"日誌行數: {len(lines):,}")
        
        # 分析最後幾行
        if lines:
            logger.info("最後10行日誌:")
            for line in lines[-10:]:
                logger.info(f"  {line.strip()}")
        
        # 統計錯誤和警告
        error_count = sum(1 for line in lines if " - ERROR - " in line)
        warning_count = sum(1 for line in lines if " - WARNING - " in line)
        
        logger.info(f"錯誤數量: {error_count}")
        logger.info(f"警告數量: {warning_count}")
        
        if error_count > 0:
            logger.info("最近的錯誤:")
            error_lines = [line for line in lines if " - ERROR - " in line]
            for line in error_lines[-5:]:  # 顯示最後5個錯誤
                logger.info(f"  {line.strip()}")
        
    except Exception as e:
        logger.error(f"分析日誌文件時出錯: {str(e)}")

def generate_recommendations():
    """生成建議"""
    logger.info("=" * 50)
    logger.info("建議和解決方案")
    logger.info("=" * 50)
    
    recommendations = [
        "1. 如果索引中斷，運行: python scripts/stable_resume_indexing.py",
        "2. 如果內存不足，考慮:",
        "   - 減少 FILE_BATCH_SIZE 配置",
        "   - 減少並行線程數",
        "   - 重啟系統釋放內存",
        "3. 如果網絡不穩定:",
        "   - 檢查 Q 槽掛載狀態",
        "   - 重新掛載網絡驅動器",
        "4. 如果 Ollama 服務異常:",
        "   - 重啟 Ollama 服務",
        "   - 檢查模型是否正確安裝",
        "5. 定期監控:",
        "   - 運行此診斷腳本檢查狀態",
        "   - 查看 logs/indexing.log 了解詳情"
    ]
    
    for rec in recommendations:
        logger.info(rec)

def main():
    logger.info("開始索引診斷...")
    
    check_vector_db_files()
    check_indexing_progress() 
    check_system_status()
    analyze_log_file()
    generate_recommendations()
    
    logger.info("診斷完成")

if __name__ == "__main__":
    main()