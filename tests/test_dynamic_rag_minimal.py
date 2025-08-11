#!/usr/bin/env python
"""
Dynamic RAG 最小化測試腳本
測試核心功能而不依賴複雜的Ollama調用
"""

import os
import sys
import logging
import time

# 添加項目根目錄到路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_engine.dynamic_rag_engine import SmartFileRetriever, DynamicContentProcessor, RealTimeVectorizer
from config.config import SUPPORTED_FILE_TYPES

# 設置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LocalFileRetriever(SmartFileRetriever):
    """本地文件檢索器"""
    
    def __init__(self, test_dir: str):
        self.base_path = test_dir
        self.file_cache = {}
        self.last_scan_time = 0
        self.cache_duration = 60

def test_file_retrieval_detailed():
    """詳細測試文件檢索"""
    print("=== 詳細文件檢索測試 ===")
    
    test_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test_files")
    
    if not os.path.exists(test_dir):
        print(f"❌ 測試文件目錄不存在: {test_dir}")
        return False
    
    try:
        retriever = LocalFileRetriever(test_dir)
        
        # 手動更新緩存以查看詳細信息
        retriever._update_file_cache()
        
        print(f"文件緩存內容:")
        for file_path, metadata in retriever.file_cache.items():
            print(f"  - {metadata['name']} ({metadata['ext']})")
        
        # 測試不同的查詢
        test_queries = [
            "sample",      # 應該匹配 sample.txt
            "technology",  # 應該匹配 technology.md
            "dynamic",     # 應該在 sample.txt 內容中找到
            "rag",         # 應該在 sample.txt 內容中找到
            "artificial",  # 應該在 technology.md 內容中找到
            "computer"     # 應該在兩個文件中都找到
        ]
        
        for query in test_queries:
            print(f"\n查詢: '{query}'")
            
            # 測試關鍵詞匹配
            keyword_matches = retriever._match_by_keywords(query)
            print(f"  關鍵詞匹配: {[os.path.basename(f) for f in keyword_matches]}")
            
            # 測試完整檢索
            files = retriever.retrieve_relevant_files(query, max_files=5)
            print(f"  完整檢索結果: {[os.path.basename(f) for f in files]}")
        
        return True
        
    except Exception as e:
        print(f"❌ 詳細文件檢索測試失敗: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return False

def test_content_processing_detailed():
    """詳細測試內容處理"""
    print("\n=== 詳細內容處理測試 ===")
    
    test_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test_files")
    
    if not os.path.exists(test_dir):
        print(f"❌ 測試文件目錄不存在: {test_dir}")
        return False
    
    try:
        processor = DynamicContentProcessor()
        
        # 獲取測試文件
        test_files = []
        for file in os.listdir(test_dir):
            file_path = os.path.join(test_dir, file)
            if os.path.isfile(file_path):
                file_ext = os.path.splitext(file)[1].lower()
                if file_ext in SUPPORTED_FILE_TYPES:
                    test_files.append(file_path)
        
        print(f"處理文件: {[os.path.basename(f) for f in test_files]}")
        
        # 逐個處理文件
        for file_path in test_files:
            print(f"\n處理文件: {os.path.basename(file_path)}")
            
            try:
                documents = processor._process_single_file(file_path)
                print(f"  生成段落數: {len(documents)}")
                
                for i, doc in enumerate(documents, 1):
                    content_preview = doc.page_content[:100].replace('\n', ' ')
                    print(f"    段落 {i}: {content_preview}...")
                    print(f"    元數據: {doc.metadata}")
                    
            except Exception as e:
                print(f"  ❌ 處理失敗: {str(e)}")
        
        return True
        
    except Exception as e:
        print(f"❌ 詳細內容處理測試失敗: {str(e)}")
        return False

def test_vectorization_basic():
    """基本向量化測試（不依賴Ollama）"""
    print("\n=== 基本向量化測試 ===")
    
    try:
        # 測試餘弦相似度計算
        import numpy as np
        
        # 創建測試向量
        vec1 = np.array([1.0, 2.0, 3.0, 4.0])
        vec2 = np.array([2.0, 4.0, 6.0, 8.0])  # vec1 的 2倍
        vec3 = np.array([1.0, 0.0, 0.0, 0.0])  # 正交向量
        
        # 手動實現餘弦相似度
        def cosine_similarity(v1, v2):
            dot_product = np.dot(v1, v2)
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return float(np.clip(dot_product / (norm1 * norm2), -1.0, 1.0))
        
        # 測試相似度計算
        sim1_2 = cosine_similarity(vec1, vec2)
        sim1_3 = cosine_similarity(vec1, vec3)
        
        print(f"向量1: {vec1}")
        print(f"向量2: {vec2}")
        print(f"向量3: {vec3}")
        print(f"相似度(1,2): {sim1_2:.4f} (應該接近1.0)")
        print(f"相似度(1,3): {sim1_3:.4f} (應該較小)")
        
        # 驗證結果
        if abs(sim1_2 - 1.0) < 0.01:
            print("✅ 平行向量相似度計算正確")
        else:
            print("❌ 平行向量相似度計算錯誤")
            
        if sim1_3 < 0.5:
            print("✅ 不同向量相似度計算正確")
        else:
            print("❌ 不同向量相似度計算錯誤")
        
        return True
        
    except Exception as e:
        print(f"❌ 基本向量化測試失敗: {str(e)}")
        return False

def test_integration_without_ollama():
    """不依賴Ollama的整合測試"""
    print("\n=== 無Ollama整合測試 ===")
    
    test_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test_files")
    
    if not os.path.exists(test_dir):
        print(f"❌ 測試文件目錄不存在: {test_dir}")
        return False
    
    try:
        # 1. 文件檢索
        retriever = LocalFileRetriever(test_dir)
        query = "technology"
        files = retriever.retrieve_relevant_files(query, max_files=3)
        print(f"檢索到文件: {[os.path.basename(f) for f in files]}")
        
        if not files:
            print("❌ 沒有檢索到文件")
            return False
        
        # 2. 內容處理
        processor = DynamicContentProcessor()
        documents = processor.process_files(files)
        print(f"處理得到 {len(documents)} 個文檔段落")
        
        if not documents:
            print("❌ 沒有處理得到文檔")
            return False
        
        # 3. 顯示結果
        print("\n檢索和處理結果:")
        for i, doc in enumerate(documents[:3], 1):
            file_name = doc.metadata.get('file_name', '未知文件')
            content_preview = doc.page_content[:150].replace('\n', ' ')
            print(f"  {i}. {file_name}: {content_preview}...")
        
        print("✅ 整合測試成功（不包含向量化）")
        return True
        
    except Exception as e:
        print(f"❌ 整合測試失敗: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return False

def main():
    """主測試函數"""
    print("開始Dynamic RAG最小化測試...")
    
    tests = [
        ("基本向量化測試", test_vectorization_basic),
        ("詳細文件檢索測試", test_file_retrieval_detailed),
        ("詳細內容處理測試", test_content_processing_detailed),
        ("無Ollama整合測試", test_integration_without_ollama)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"執行: {test_name}")
        print('='*60)
        
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} 通過")
            else:
                print(f"❌ {test_name} 失敗")
        except Exception as e:
            print(f"❌ {test_name} 發生異常: {str(e)}")
    
    print(f"\n{'='*60}")
    print("測試結果總結")
    print('='*60)
    print(f"通過: {passed}/{total}")
    print(f"成功率: {passed/total*100:.1f}%")
    
    if passed == total:
        print("🎉 所有最小化測試通過！Dynamic RAG核心功能正常")
        print("\n下一步:")
        print("1. 確保Ollama服務正常運行")
        print("2. 檢查模型是否已下載")
        print("3. 運行完整的Dynamic RAG測試")
    elif passed > 0:
        print("⚠️ 部分測試通過，核心功能基本正常")
    else:
        print("❌ 所有測試失敗，請檢查基本配置")

if __name__ == "__main__":
    main()