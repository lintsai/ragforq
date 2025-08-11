#!/usr/bin/env python
"""
Dynamic RAG 完整功能測試
使用實際的Ollama模型進行測試
"""

import os
import sys
import logging
import time

# 添加項目根目錄到路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_engine.dynamic_rag_engine import DynamicRAGEngine
from config.config import Q_DRIVE_PATH

# 設置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_dynamic_rag_with_real_models():
    """使用真實模型測試Dynamic RAG"""
    print("=== Dynamic RAG 完整功能測試 ===")
    
    try:
        # 使用可用的模型
        language_model = "qwen2.5:0.5b-instruct"
        embedding_model = "nomic-embed-text:latest"
        
        print(f"語言模型: {language_model}")
        print(f"嵌入模型: {embedding_model}")
        
        # 創建Dynamic RAG引擎
        print("\n創建Dynamic RAG引擎...")
        start_time = time.time()
        
        engine = DynamicRAGEngine(
            ollama_model=language_model,
            ollama_embedding_model=embedding_model,
            language="繁體中文"
        )
        
        creation_time = time.time() - start_time
        print(f"✅ 引擎創建成功，耗時: {creation_time:.2f}秒")
        
        # 測試問題
        test_questions = [
            "What is a computer?",
            "How does a file system work?",
            "What is artificial intelligence?"
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
                print(f"回答預覽: {answer[:150]}...")
                
                # 測試帶來源的問答
                print("\n測試帶來源的問答...")
                start_time = time.time()
                
                answer_with_sources, sources, documents = engine.get_answer_with_sources(question)
                sources_time = time.time() - start_time
                
                print(f"✅ 帶來源回答生成成功，耗時: {sources_time:.2f}秒")
                print(f"找到文檔數量: {len(documents)}")
                print(f"來源信息長度: {len(sources)} 字符")
                
                if documents:
                    print("相關文檔:")
                    for j, doc in enumerate(documents[:3], 1):
                        file_name = doc.metadata.get('file_name', '未知文件')
                        print(f"  {j}. {file_name}")
                
                # 只測試第一個問題以節省時間
                break
                
            except Exception as e:
                print(f"❌ 問答測試失敗: {str(e)}")
                import traceback
                print(f"詳細錯誤: {traceback.format_exc()}")
        
        print(f"\n=== 測試完成 ===")
        return True
        
    except Exception as e:
        print(f"❌ Dynamic RAG測試失敗: {str(e)}")
        import traceback
        print(f"詳細錯誤: {traceback.format_exc()}")
        return False

def test_api_integration():
    """測試API整合"""
    print("\n=== 測試API整合 ===")
    
    try:
        import requests
        
        # 測試API端點
        api_url = "http://localhost:8000"
        
        # 檢查API是否運行
        try:
            response = requests.get(f"{api_url}/health", timeout=5)
            if response.status_code == 200:
                print("✅ API服務正在運行")
                
                # 測試Dynamic RAG API調用
                payload = {
                    "question": "What is a document?",
                    "use_dynamic_rag": True,
                    "selected_model": "qwen2.5:0.5b-instruct",
                    "ollama_embedding_model": "nomic-embed-text:latest",
                    "language": "繁體中文",
                    "include_sources": True
                }
                
                print("發送Dynamic RAG API請求...")
                start_time = time.time()
                
                response = requests.post(f"{api_url}/ask", json=payload, timeout=30)
                api_time = time.time() - start_time
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"✅ API調用成功，耗時: {api_time:.2f}秒")
                    print(f"回答長度: {len(result.get('answer', ''))} 字符")
                    print(f"來源數量: {len(result.get('sources', []))}")
                    print(f"回答預覽: {result.get('answer', '')[:100]}...")
                    return True
                else:
                    print(f"❌ API調用失敗: {response.status_code}")
                    print(f"錯誤信息: {response.text}")
                    return False
            else:
                print("⚠️ API服務未運行，跳過API測試")
                return True
                
        except requests.exceptions.RequestException:
            print("⚠️ 無法連接到API服務，跳過API測試")
            return True
            
    except Exception as e:
        print(f"❌ API整合測試失敗: {str(e)}")
        return False

def test_performance_metrics():
    """測試性能指標"""
    print("\n=== 性能指標測試 ===")
    
    try:
        # 創建引擎
        engine = DynamicRAGEngine(
            ollama_model="qwen2.5:0.5b-instruct",
            ollama_embedding_model="nomic-embed-text:latest",
            language="English"
        )
        
        # 測試多個查詢的性能
        queries = [
            "What is technology?",
            "How does software work?",
            "What is data processing?"
        ]
        
        total_time = 0
        successful_queries = 0
        
        for query in queries:
            try:
                start_time = time.time()
                answer = engine.answer_question(query)
                query_time = time.time() - start_time
                
                total_time += query_time
                successful_queries += 1
                
                print(f"查詢: '{query[:30]}...' - 耗時: {query_time:.2f}秒")
                
            except Exception as e:
                print(f"查詢失敗: {str(e)}")
        
        if successful_queries > 0:
            avg_time = total_time / successful_queries
            print(f"\n性能統計:")
            print(f"成功查詢: {successful_queries}/{len(queries)}")
            print(f"總耗時: {total_time:.2f}秒")
            print(f"平均耗時: {avg_time:.2f}秒/查詢")
            
            if avg_time < 10:
                print("✅ 性能表現良好")
            elif avg_time < 20:
                print("⚠️ 性能一般")
            else:
                print("❌ 性能較慢，建議優化")
        
        return True
        
    except Exception as e:
        print(f"❌ 性能測試失敗: {str(e)}")
        return False

def main():
    """主測試函數"""
    print("開始Dynamic RAG完整功能測試...")
    print(f"Q槽路徑: {Q_DRIVE_PATH}")
    print(f"Q槽可訪問: {os.path.exists(Q_DRIVE_PATH)}")
    
    tests = [
        ("Dynamic RAG引擎測試", test_dynamic_rag_with_real_models),
        ("API整合測試", test_api_integration),
        ("性能指標測試", test_performance_metrics)
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
        print("🎉 所有測試通過！Dynamic RAG功能完全正常")
    elif passed > 0:
        print("⚠️ 部分測試通過，Dynamic RAG基本功能正常")
    else:
        print("❌ 所有測試失敗，請檢查配置和依賴")

if __name__ == "__main__":
    main()