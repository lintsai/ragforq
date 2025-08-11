#!/usr/bin/env python
"""
Dynamic RAG 測試腳本
測試動態RAG功能的各個組件
"""

import os
import sys
import logging
import time
from pathlib import Path

# 添加項目根目錄到路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_engine.dynamic_rag_engine import DynamicRAGEngine, SmartFileRetriever, DynamicContentProcessor, RealTimeVectorizer
from config.config import Q_DRIVE_PATH

# 設置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_smart_file_retriever():
    """測試智能文件檢索器"""
    print("\n=== 測試智能文件檢索器 ===")
    
    # 檢查Q槽路徑是否存在
    if not os.path.exists(Q_DRIVE_PATH):
        print(f"警告: Q槽路徑不存在: {Q_DRIVE_PATH}")
        print("跳過文件檢索測試")
        return
    
    retriever = SmartFileRetriever()
    
    # 測試查詢
    test_queries = [
        "test",  # 使用更通用的查詢詞
        "doc",
        "file",
        "pdf",
        "txt"
    ]
    
    for query in test_queries:
        print(f"\n查詢: {query}")
        start_time = time.time()
        try:
            files = retriever.retrieve_relevant_files(query, max_files=5)
            end_time = time.time()
            
            print(f"檢索時間: {end_time - start_time:.2f}秒")
            print(f"找到 {len(files)} 個相關文件:")
            for i, file_path in enumerate(files, 1):
                print(f"  {i}. {os.path.basename(file_path)}")
                
            if files:
                break  # 找到文件就停止測試其他查詢
        except Exception as e:
            print(f"檢索失敗: {str(e)}")
    
    if not any(files for query in test_queries for files in [retriever.retrieve_relevant_files(query, max_files=1)]):
        print("未找到任何文件，可能Q槽中沒有支持的文件類型")

def test_dynamic_content_processor():
    """測試動態內容處理器"""
    print("\n=== 測試動態內容處理器 ===")
    
    if not os.path.exists(Q_DRIVE_PATH):
        print(f"警告: Q槽路徑不存在: {Q_DRIVE_PATH}")
        print("跳過內容處理測試")
        return
    
    processor = DynamicContentProcessor()
    
    # 獲取一些測試文件
    retriever = SmartFileRetriever()
    
    # 嘗試多個查詢詞來找到文件
    test_queries = ["test", "doc", "file", "txt", "pdf"]
    test_files = []
    
    for query in test_queries:
        try:
            files = retriever.retrieve_relevant_files(query, max_files=2)
            if files:
                test_files.extend(files)
                break
        except Exception as e:
            print(f"查詢 '{query}' 失敗: {str(e)}")
            continue
    
    if not test_files:
        print("沒有找到測試文件，跳過內容處理測試")
        return
    
    # 只取前3個文件避免處理時間過長
    test_files = test_files[:3]
    
    print(f"處理 {len(test_files)} 個文件:")
    for file_path in test_files:
        print(f"  - {os.path.basename(file_path)}")
    
    start_time = time.time()
    try:
        documents = processor.process_files(test_files)
        end_time = time.time()
        
        print(f"處理時間: {end_time - start_time:.2f}秒")
        print(f"生成 {len(documents)} 個文檔段落")
        
        if documents:
            print(f"第一個段落預覽: {documents[0].page_content[:100]}...")
    except Exception as e:
        print(f"處理文件時出錯: {str(e)}")

def test_real_time_vectorizer():
    """測試即時向量化引擎"""
    print("\n=== 測試即時向量化引擎 ===")
    
    try:
        # 檢查Ollama服務是否可用
        import requests
        from config.config import OLLAMA_HOST
        
        try:
            response = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
            if response.status_code != 200:
                print(f"Ollama服務不可用: {OLLAMA_HOST}")
                print("跳過向量化測試")
                return
        except Exception as e:
            print(f"無法連接到Ollama服務: {str(e)}")
            print("跳過向量化測試")
            return
        
        vectorizer = RealTimeVectorizer("nomic-embed-text")
        
        # 測試查詢向量化
        query = "test document"
        print(f"向量化查詢: {query}")
        
        start_time = time.time()
        query_vector = vectorizer.vectorize_query(query)
        end_time = time.time()
        
        print(f"向量化時間: {end_time - start_time:.2f}秒")
        print(f"向量維度: {len(query_vector)}")
        
        if len(query_vector) == 0:
            print("查詢向量化失敗，跳過後續測試")
            return
        
        # 測試文檔向量化（如果有文件的話）
        if os.path.exists(Q_DRIVE_PATH):
            processor = DynamicContentProcessor()
            retriever = SmartFileRetriever()
            
            # 嘗試找到一些文件
            test_files = []
            for query_term in ["test", "doc", "file"]:
                try:
                    files = retriever.retrieve_relevant_files(query_term, max_files=1)
                    if files:
                        test_files = files
                        break
                except:
                    continue
            
            if test_files:
                documents = processor.process_files(test_files)
                if documents:
                    print(f"\n向量化 {min(len(documents), 2)} 個文檔段落")
                    start_time = time.time()
                    doc_vectors = vectorizer.vectorize_documents(documents[:2])  # 只測試前2個
                    end_time = time.time()
                    
                    print(f"文檔向量化時間: {end_time - start_time:.2f}秒")
                    print(f"生成 {len(doc_vectors)} 個文檔向量")
                    
                    # 測試相似度計算
                    if doc_vectors:
                        similarities = vectorizer.calculate_similarities(query_vector, doc_vectors)
                        print(f"相似度計算結果: {len(similarities)} 個")
                        for i, (doc, score) in enumerate(similarities, 1):
                            print(f"  {i}. 相似度: {score:.4f}, 文件: {doc.metadata.get('file_name', '未知')}")
            else:
                print("未找到測試文件，跳過文檔向量化測試")
        else:
            print("Q槽路徑不存在，跳過文檔向量化測試")
    
    except Exception as e:
        print(f"向量化測試失敗: {str(e)}")
        import traceback
        print(f"詳細錯誤: {traceback.format_exc()}")

def test_dynamic_rag_engine():
    """測試完整的Dynamic RAG引擎"""
    print("\n=== 測試Dynamic RAG引擎 ===")
    
    try:
        # 檢查Ollama服務
        import requests
        from config.config import OLLAMA_HOST
        
        try:
            response = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
            if response.status_code != 200:
                print(f"Ollama服務不可用: {OLLAMA_HOST}")
                print("跳過Dynamic RAG引擎測試")
                return
        except Exception as e:
            print(f"無法連接到Ollama服務: {str(e)}")
            print("跳過Dynamic RAG引擎測試")
            return
        
        # 創建Dynamic RAG引擎
        engine = DynamicRAGEngine(
            ollama_model="phi3:mini",
            ollama_embedding_model="nomic-embed-text",
            language="繁體中文"
        )
        
        # 測試問題（使用更通用的問題）
        test_questions = [
            "What is a document?",
            "How to create a file?",
            "What is testing?"
        ]
        
        for question in test_questions:
            print(f"\n問題: {question}")
            start_time = time.time()
            
            try:
                answer = engine.answer_question(question)
                end_time = time.time()
                
                print(f"回答時間: {end_time - start_time:.2f}秒")
                print(f"回答: {answer[:200]}...")
                
                # 只測試一個問題以節省時間
                break
                
            except Exception as e:
                print(f"回答問題失敗: {str(e)}")
                import traceback
                print(f"詳細錯誤: {traceback.format_exc()}")
    
    except Exception as e:
        print(f"創建Dynamic RAG引擎失敗: {str(e)}")
        import traceback
        print(f"詳細錯誤: {traceback.format_exc()}")

def test_performance_comparison():
    """性能對比測試"""
    print("\n=== 性能對比測試 ===")
    
    # 這裡可以添加與傳統RAG的性能對比
    # 由於需要現有的向量資料庫，暫時跳過
    print("性能對比測試需要現有的向量資料庫，暫時跳過")

def main():
    """主測試函數"""
    print("開始Dynamic RAG測試...")
    
    # 檢查Q槽是否可訪問
    if not os.path.exists(Q_DRIVE_PATH):
        print(f"警告: Q槽路徑不存在: {Q_DRIVE_PATH}")
        print("某些測試可能會失敗")
    
    try:
        # 運行各項測試
        test_smart_file_retriever()
        test_dynamic_content_processor()
        test_real_time_vectorizer()
        test_dynamic_rag_engine()
        test_performance_comparison()
        
        print("\n=== 測試完成 ===")
        print("Dynamic RAG各組件測試完成！")
        
    except KeyboardInterrupt:
        print("\n測試被用戶中斷")
    except Exception as e:
        print(f"\n測試過程中發生錯誤: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()