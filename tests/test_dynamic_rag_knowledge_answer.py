#!/usr/bin/env python3
"""
測試動態RAG的常識回答功能
"""

import os
import sys
import logging

# 添加項目根目錄到路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_engine.rag_engine_factory import get_rag_engine_for_language

# 設置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_dynamic_rag_knowledge_answer():
    """測試動態RAG的常識回答功能"""
    
    # 測試語言列表和對應問題
    test_cases = {
        "繁體中文": [
            "什麼是人工智能？",
            "如何學習程式設計？", 
            "雲端運算的優點是什麼？",
            "什麼是區塊鏈技術？"
        ],
        "简体中文": [
            "什么是人工智能？",
            "如何学习编程？",
            "云计算的优点是什么？",
            "什么是区块链技术？"
        ],
        "English": [
            "What is artificial intelligence?",
            "How to learn programming?",
            "What are the advantages of cloud computing?",
            "What is blockchain technology?"
        ],
        "ไทย": [
            "ปัญญาประดิษฐ์คืออะไร?",
            "จะเรียนการเขียนโปรแกรมได้อย่างไร?",
            "ข้อดีของคลาวด์คอมพิวติ้งคืออะไร?",
            "เทคโนโลยีบล็อกเชนคืออะไร?"
        ]
    }
    
    print("=== 動態RAG常識回答功能測試（重構版本）===\n")
    
    for language, questions in test_cases.items():
        print(f"測試語言: {language}")
        print("-" * 50)
        
        try:
            # 創建動態RAG引擎
            dynamic_language = f"Dynamic_{language}"
            engine = get_rag_engine_for_language(
                language=dynamic_language,
                document_indexer=None,  # 動態RAG不需要document_indexer
                ollama_model="llama3.2:3b",
                ollama_embedding_model="nomic-embed-text"
            )
            
            for question in questions:
                print(f"\n問題: {question}")
                
                try:
                    # 測試常識回答功能（現在每個引擎都有自己的實現）
                    answer = engine._generate_general_knowledge_answer(question)
                    print(f"回答: {answer[:200]}...")
                    
                    # 檢查回答是否包含免責聲明
                    if "※" in answer:
                        print("✅ 包含免責聲明")
                    else:
                        print("❌ 缺少免責聲明")
                    
                    # 檢查回答長度
                    if len(answer) > 50:
                        print("✅ 回答長度適當")
                    else:
                        print("❌ 回答過短")
                    
                    # 檢查語言一致性
                    if language == "繁體中文" and any(char in answer for char in "繁體中文"):
                        print("✅ 語言一致性正確")
                    elif language == "简体中文" and any(char in answer for char in "简体中文"):
                        print("✅ 語言一致性正確")
                    elif language == "English" and "English" in answer:
                        print("✅ 語言一致性正確")
                    elif language == "ไทย" and any(char in answer for char in "ภาษาไทย"):
                        print("✅ 語言一致性正確")
                    else:
                        print("⚠️ 語言一致性需要檢查")
                        
                except Exception as e:
                    print(f"❌ 常識回答生成失敗: {str(e)}")
                
                print()
            
        except Exception as e:
            print(f"❌ 引擎創建失敗: {str(e)}")
        
        print("\n" + "="*70 + "\n")

def test_dynamic_rag_fallback():
    """測試動態RAG的回退功能"""
    
    print("=== 動態RAG回退功能測試 ===\n")
    
    languages = ["繁體中文", "简体中文", "English", "ไทย"]
    
    for language in languages:
        print(f"測試語言: {language}")
        print("-" * 30)
        
        try:
            # 創建動態RAG引擎
            dynamic_language = f"Dynamic_{language}"
            engine = get_rag_engine_for_language(
                language=dynamic_language,
                document_indexer=None,
                ollama_model="llama3.2:3b",
                ollama_embedding_model="nomic-embed-text"
            )
            
            # 測試回退功能
            fallback_answer = engine._get_general_fallback("測試查詢")
            print(f"回退回答: {fallback_answer}")
            
            # 檢查語言一致性
            if language == "繁體中文" and "繁體中文" in fallback_answer:
                print("✅ 語言一致性正確")
            elif language == "简体中文" and "简体中文" in fallback_answer:
                print("✅ 語言一致性正確")
            elif language == "English" and "English" in fallback_answer:
                print("✅ 語言一致性正確")
            elif language == "ไทย" and "ไทย" in fallback_answer:
                print("✅ 語言一致性正確")
            else:
                print("⚠️ 語言一致性需要檢查")
                
        except Exception as e:
            print(f"❌ 測試失敗: {str(e)}")
        
        print()

def test_dynamic_vs_traditional_comparison():
    """比較動態RAG和傳統RAG的常識回答功能"""
    
    print("=== 動態RAG vs 傳統RAG 常識回答比較 ===\n")
    
    test_question = "什麼是機器學習？"
    
    try:
        # 創建傳統RAG引擎
        from indexer.document_indexer import DocumentIndexer
        indexer = DocumentIndexer()
        
        traditional_engine = get_rag_engine_for_language(
            language="繁體中文",
            document_indexer=indexer,
            ollama_model="llama3.2:3b"
        )
        
        # 創建動態RAG引擎
        dynamic_engine = get_rag_engine_for_language(
            language="Dynamic_繁體中文",
            document_indexer=None,
            ollama_model="llama3.2:3b",
            ollama_embedding_model="nomic-embed-text"
        )
        
        print(f"測試問題: {test_question}\n")
        
        # 傳統RAG常識回答
        print("傳統RAG常識回答:")
        traditional_answer = traditional_engine._generate_general_knowledge_answer(test_question)
        print(f"{traditional_answer[:300]}...\n")
        
        # 動態RAG常識回答
        print("動態RAG常識回答:")
        dynamic_answer = dynamic_engine._generate_general_knowledge_answer(test_question)
        print(f"{dynamic_answer[:300]}...\n")
        
        # 比較分析
        print("比較分析:")
        if len(traditional_answer) > len(dynamic_answer):
            print("✅ 傳統RAG回答更詳細")
        elif len(dynamic_answer) > len(traditional_answer):
            print("✅ 動態RAG回答更詳細")
        else:
            print("⚖️ 兩者回答長度相似")
        
        if "※" in traditional_answer and "※" in dynamic_answer:
            print("✅ 兩者都包含免責聲明")
        elif "※" in traditional_answer:
            print("⚠️ 只有傳統RAG包含免責聲明")
        elif "※" in dynamic_answer:
            print("⚠️ 只有動態RAG包含免責聲明")
        else:
            print("❌ 兩者都缺少免責聲明")
            
    except Exception as e:
        print(f"❌ 比較測試失敗: {str(e)}")

if __name__ == "__main__":
    print("開始測試動態RAG的常識回答功能...\n")
    
    # 測試1: 常識回答功能
    test_dynamic_rag_knowledge_answer()
    
    # 測試2: 回退功能
    test_dynamic_rag_fallback()
    
    # 測試3: 與傳統RAG比較
    test_dynamic_vs_traditional_comparison()
    
    print("測試完成！")