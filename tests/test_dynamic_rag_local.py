#!/usr/bin/env python
"""
Dynamic RAG 本地測試腳本
使用本地測試文件進行測試
"""

import os
import sys
import logging
import time

# 添加項目根目錄到路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_engine.dynamic_rag_engine import DynamicRAGEngine, SmartFileRetriever
from config.config import SUPPORTED_FILE_TYPES

# 設置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LocalFileRetriever(SmartFileRetriever):
    """本地文件檢索器 - 用於測試"""
    
    def __init__(self, test_dir: str):
        self.base_path = test_dir
        self.file_cache = {}
        self.last_scan_time = 0
        self.cache_duration = 60  # 1分鐘緩存

def test_local_dynamic_rag():
    """使用本地文件測試Dynamic RAG"""
    print("=== 本地Dynamic RAG測試 ===")
    
    # 獲取測試文件目錄
    test_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test_files")
    
    if not os.path.exists(test_dir):
        print(f"❌ 測試文件目錄不存在: {test_dir}")
        return False
    
    print(f"測試文件目錄: {test_dir}")
    
    # 列出測試文件
    test_files = []
    for file in os.listdir(test_dir):
        file_path = os.path.join(test_dir, file)
        if os.path.isfile(file_path):
            file_ext = os.path.splitext(file)[1].lower()
            if file_ext in SUPPORTED_FILE_TYPES:
                test_files.append(file)
    
    print(f"找到測試文件: {test_files}")
    
    if not test_files:
        print("❌ 沒有找到支持的測試文件")
        return False
    
    try:
        # 創建Dynamic RAG引擎
        print("\n創建Dynamic RAG引擎...")
        
        # 創建一個修改版的引擎，使用本地文件檢索器
        engine = DynamicRAGEngine(
            ollama_model="qwen2.5:0.5b-instruct",
            ollama_embedding_model="nomic-embed-text:latest",
            language="English"
        )
        
        # 替換文件檢索器為本地版本
        engine.file_retriever = LocalFileRetriever(test_dir)
        
        print("✅ 引擎創建成功")
        
        # 測試問題
        test_questions = [
            "What is Dynamic RAG?",
            "What is technology?",
            "How does artificial intelligence work?",
            "What are the benefits of technology?"
        ]
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n--- 測試問題 {i} ---")
            print(f"問題: {question}")
            
            start_time = time.time()
            
            try:
                # 測試基本問答
                answer = engine.answer_question(question)
                answer_time = time.time() - start_time
                
                print(f"✅ 回答生成成功，耗時: {answer_time:.2f}秒")
                print(f"回答長度: {len(answer)} 字符")
                print(f"回答: {answer}")
                
                # 測試帶來源的問答
                print("\n測試帶來源的問答...")
                start_time = time.time()
                
                answer_with_sources, sources, documents = engine.get_answer_with_sources(question)
                sources_time = time.time() - start_time
                
                print(f"✅ 帶來源回答生成成功，耗時: {sources_time:.2f}秒")
                print(f"找到文檔數量: {len(documents)}")
                
                if documents:
                    print("相關文檔:")
                    for j, doc in enumerate(documents[:3], 1):
                        file_name = doc.metadata.get('file_name', '未知文件')
                        content_preview = doc.page_content[:100].replace('\n', ' ')
                        print(f"  {j}. {file_name}: {content_preview}...")
                
                if sources:
                    print(f"來源信息: {sources}")
                
                print("-" * 50)
                
            except Exception as e:
                print(f"❌ 問答測試失敗: {str(e)}")
                import traceback
                print(f"詳細錯誤: {traceback.format_exc()}")
        
        return True
        
    except Exception as e:
        print(f"❌ 本地Dynamic RAG測試失敗: {str(e)}")
        import traceback
        print(f"詳細錯誤: {traceback.format_exc()}")
        return False

def test_file_retrieval():
    """測試文件檢索功能"""
    print("\n=== 測試文件檢索功能 ===")
    
    test_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test_files")
    
    if not os.path.exists(test_dir):
        print(f"❌ 測試文件目錄不存在: {test_dir}")
        return False
    
    try:
        retriever = LocalFileRetriever(test_dir)
        
        # 測試查詢
        test_queries = [
            "Dynamic RAG",
            "technology",
            "artificial intelligence",
            "computer"
        ]
        
        for query in test_queries:
            print(f"\n查詢: {query}")
            start_time = time.time()
            
            files = retriever.retrieve_relevant_files(query, max_files=5)
            retrieval_time = time.time() - start_time
            
            print(f"檢索時間: {retrieval_time:.2f}秒")
            print(f"找到 {len(files)} 個相關文件:")
            
            for i, file_path in enumerate(files, 1):
                file_name = os.path.basename(file_path)
                print(f"  {i}. {file_name}")
        
        return True
        
    except Exception as e:
        print(f"❌ 文件檢索測試失敗: {str(e)}")
        return False

def test_content_processing():
    """測試內容處理功能"""
    print("\n=== 測試內容處理功能 ===")
    
    test_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test_files")
    
    if not os.path.exists(test_dir):
        print(f"❌ 測試文件目錄不存在: {test_dir}")
        return False
    
    try:
        from rag_engine.dynamic_rag_engine import DynamicContentProcessor
        
        processor = DynamicContentProcessor()
        
        # 獲取測試文件
        test_files = []
        for file in os.listdir(test_dir):
            file_path = os.path.join(test_dir, file)
            if os.path.isfile(file_path):
                file_ext = os.path.splitext(file)[1].lower()
                if file_ext in SUPPORTED_FILE_TYPES:
                    test_files.append(file_path)
        
        if not test_files:
            print("❌ 沒有找到測試文件")
            return False
        
        print(f"處理 {len(test_files)} 個文件:")
        for file_path in test_files:
            print(f"  - {os.path.basename(file_path)}")
        
        start_time = time.time()
        documents = processor.process_files(test_files)
        processing_time = time.time() - start_time
        
        print(f"✅ 處理完成，耗時: {processing_time:.2f}秒")
        print(f"生成 {len(documents)} 個文檔段落")
        
        if documents:
            print("\n文檔段落預覽:")
            for i, doc in enumerate(documents[:3], 1):
                file_name = doc.metadata.get('file_name', '未知文件')
                content_preview = doc.page_content[:100].replace('\n', ' ')
                print(f"  {i}. {file_name}: {content_preview}...")
        
        return True
        
    except Exception as e:
        print(f"❌ 內容處理測試失敗: {str(e)}")
        return False

def main():
    """主測試函數"""
    print("開始本地Dynamic RAG測試...")
    
    tests = [
        ("文件檢索功能", test_file_retrieval),
        ("內容處理功能", test_content_processing),
        ("完整Dynamic RAG", test_local_dynamic_rag)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"執行: {test_name}")
        print('='*50)
        
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} 通過")
            else:
                print(f"❌ {test_name} 失敗")
        except Exception as e:
            print(f"❌ {test_name} 發生異常: {str(e)}")
    
    print(f"\n{'='*50}")
    print("測試結果總結")
    print('='*50)
    print(f"通過: {passed}/{total}")
    print(f"成功率: {passed/total*100:.1f}%")
    
    if passed == total:
        print("🎉 所有本地測試通過！Dynamic RAG功能正常")
    elif passed > 0:
        print("⚠️ 部分測試通過，Dynamic RAG基本功能正常")
    else:
        print("❌ 所有測試失敗，請檢查配置")

if __name__ == "__main__":
    main()