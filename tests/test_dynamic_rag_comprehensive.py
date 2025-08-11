#!/usr/bin/env python3
"""
動態RAG功能全面測試
測試所有語言模型和語言選項的組合
"""

import requests
import json
import time
from typing import Dict, List, Any

API_BASE_URL = "http://127.0.0.1:8000"

def get_available_models() -> Dict[str, List[str]]:
    """獲取可用的Ollama模型"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/ollama/models/categorized", timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"❌ 無法獲取模型列表: {response.status_code}")
            return {}
    except Exception as e:
        print(f"❌ 獲取模型列表異常: {e}")
        return {}

def test_dynamic_rag_combination(language_model: str, embedding_model: str, language: str, question: str) -> Dict[str, Any]:
    """測試特定模型和語言組合"""
    payload = {
        "question": question,
        "use_dynamic_rag": True,
        "selected_model": language_model,
        "ollama_embedding_model": embedding_model,
        "language": language,
        "include_sources": True,
        "max_sources": 3,
        "show_relevance": True
    }
    
    try:
        start_time = time.time()
        response = requests.post(f"{API_BASE_URL}/ask", json=payload, timeout=60)
        end_time = time.time()
        
        result = {
            "status_code": response.status_code,
            "response_time": end_time - start_time,
            "success": response.status_code == 200
        }
        
        if response.status_code == 200:
            data = response.json()
            result.update({
                "answer_length": len(data.get("answer", "")),
                "sources_count": len(data.get("sources", [])),
                "has_rewritten_query": bool(data.get("rewritten_query")),
                "answer_preview": data.get("answer", "")[:100] + "..." if len(data.get("answer", "")) > 100 else data.get("answer", "")
            })
        else:
            result["error"] = response.text
            
        return result
        
    except Exception as e:
        return {
            "status_code": 0,
            "success": False,
            "error": str(e),
            "response_time": 0
        }

def main():
    print("🚀 動態RAG功能全面測試開始")
    print("=" * 60)
    
    # 獲取可用模型
    models = get_available_models()
    if not models:
        print("❌ 無法獲取模型列表，測試終止")
        return
    
    language_models = models.get("language_models", [])
    embedding_models = models.get("embedding_models", [])
    
    print(f"📋 可用語言模型: {language_models}")
    print(f"📋 可用嵌入模型: {embedding_models}")
    print()
    
    # 測試語言選項
    languages = ["繁體中文", "简体中文", "English", "ไทย", "Dynamic"]
    
    # 測試問題
    test_questions = [
        "什麼是ITPortal？",
        "公司的政策是什麼？",
        "How to use the system?",
        "測試問題"
    ]
    
    total_tests = 0
    successful_tests = 0
    failed_tests = []
    
    # 測試每個語言模型
    for lang_model in language_models:
        print(f"\n🤖 測試語言模型: {lang_model}")
        print("-" * 50)
        
        # 測試每個嵌入模型
        for embed_model in embedding_models:
            print(f"  🔤 嵌入模型: {embed_model}")
            
            # 測試每種語言
            for language in languages:
                print(f"    🌐 語言: {language}")
                
                # 測試第一個問題
                question = test_questions[0]
                total_tests += 1
                
                result = test_dynamic_rag_combination(lang_model, embed_model, language, question)
                
                if result["success"]:
                    successful_tests += 1
                    print(f"      ✅ 成功 - 響應時間: {result['response_time']:.2f}s")
                    print(f"         答案長度: {result.get('answer_length', 0)} 字符")
                    print(f"         來源數量: {result.get('sources_count', 0)}")
                    if result.get('answer_preview'):
                        print(f"         答案預覽: {result['answer_preview']}")
                else:
                    failed_tests.append({
                        "lang_model": lang_model,
                        "embed_model": embed_model,
                        "language": language,
                        "error": result.get("error", "未知錯誤")
                    })
                    print(f"      ❌ 失敗 - {result.get('error', '未知錯誤')}")
                
                print()
    
    # 測試摘要
    print("\n" + "=" * 60)
    print("📊 測試摘要")
    print("=" * 60)
    print(f"總測試數: {total_tests}")
    print(f"成功: {successful_tests}")
    print(f"失敗: {len(failed_tests)}")
    print(f"成功率: {(successful_tests/total_tests*100):.1f}%" if total_tests > 0 else "0%")
    
    if failed_tests:
        print("\n❌ 失敗的測試組合:")
        for i, test in enumerate(failed_tests, 1):
            print(f"{i}. {test['lang_model']} + {test['embed_model']} ({test['language']})")
            print(f"   錯誤: {test['error']}")
    
    # 語言功能特別測試
    print("\n" + "=" * 60)
    print("🌐 語言功能特別測試")
    print("=" * 60)
    
    if language_models and embedding_models:
        # 使用第一個可用的模型組合測試所有語言
        test_lang_model = language_models[0]
        test_embed_model = embedding_models[0]
        
        language_test_questions = {
            "繁體中文": "請介紹公司的組織架構",
            "简体中文": "请介绍公司的组织架构", 
            "English": "Please introduce the company's organizational structure",
            "ไทย": "กรุณาแนะนำโครงสร้างองค์กรของบริษัท",
            "Dynamic": "請用適當的語言回答：公司政策"
        }
        
        for language, question in language_test_questions.items():
            print(f"\n🌐 測試語言: {language}")
            print(f"📝 問題: {question}")
            
            result = test_dynamic_rag_combination(test_lang_model, test_embed_model, language, question)
            
            if result["success"]:
                print(f"✅ 成功 - 響應時間: {result['response_time']:.2f}s")
                print(f"📝 答案預覽: {result.get('answer_preview', 'N/A')}")
            else:
                print(f"❌ 失敗 - {result.get('error', '未知錯誤')}")

if __name__ == "__main__":
    main()