#!/usr/bin/env python3
"""
動態RAG功能簡化測試
專注測試核心功能是否正常工作
"""

import requests
import json
import sys
import os

# 添加項目根目錄到路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

API_BASE_URL = "http://127.0.0.1:8000"

def test_api_health():
    """測試API服務健康狀態"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def get_ollama_models():
    """獲取Ollama模型列表"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/ollama/models/categorized", timeout=10)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        print(f"獲取模型列表失敗: {e}")
        return None

def test_traditional_rag():
    """測試傳統RAG功能"""
    print("🔍 測試傳統RAG...")
    
    # 獲取可用的向量模型
    try:
        response = requests.get(f"{API_BASE_URL}/api/usable-models", timeout=10)
        if response.status_code != 200:
            print("❌ 無法獲取可用模型")
            return False
        
        models = response.json()
        if not models:
            print("❌ 沒有可用的向量模型")
            return False
        
        # 使用第一個可用模型
        model = models[0]['folder_name']
        print(f"   使用模型: {models[0]['display_name']}")
        
        payload = {
            "question": "測試問題",
            "selected_model": model,
            "use_dynamic_rag": False,
            "language": "繁體中文",
            "include_sources": True,
            "max_sources": 3
        }
        
        response = requests.post(f"{API_BASE_URL}/ask", json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ 成功 - 答案長度: {len(result.get('answer', ''))}")
            return True
        else:
            print(f"   ❌ 失敗 - 狀態碼: {response.status_code}")
            print(f"   錯誤: {response.text}")
            return False
            
    except Exception as e:
        print(f"   ❌ 異常: {e}")
        return False

def test_dynamic_rag():
    """測試動態RAG功能"""
    print("🚀 測試動態RAG...")
    
    models = get_ollama_models()
    if not models:
        print("   ❌ 無法獲取Ollama模型")
        return False
    
    language_models = models.get('language_models', [])
    embedding_models = models.get('embedding_models', [])
    
    if not language_models or not embedding_models:
        print("   ❌ 缺少必要的模型")
        print(f"   語言模型: {language_models}")
        print(f"   嵌入模型: {embedding_models}")
        return False
    
    # 使用第一個可用的模型組合
    lang_model = language_models[0]
    embed_model = embedding_models[0]
    
    print(f"   使用語言模型: {lang_model}")
    print(f"   使用嵌入模型: {embed_model}")
    
    # 測試不同語言
    languages = ["繁體中文", "English", "Dynamic"]
    
    for language in languages:
        print(f"   測試語言: {language}")
        
        payload = {
            "question": "什麼是ITPortal？" if language != "English" else "What is ITPortal?",
            "selected_model": lang_model,
            "ollama_embedding_model": embed_model,
            "use_dynamic_rag": True,
            "language": language,
            "include_sources": True,
            "max_sources": 3
        }
        
        try:
            response = requests.post(f"{API_BASE_URL}/ask", json=payload, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                answer_len = len(result.get('answer', ''))
                sources_count = len(result.get('sources', []))
                print(f"     ✅ 成功 - 答案: {answer_len}字符, 來源: {sources_count}個")
                
                # 顯示答案預覽
                answer = result.get('answer', '')
                if answer:
                    preview = answer[:100] + "..." if len(answer) > 100 else answer
                    print(f"     預覽: {preview}")
                
            else:
                print(f"     ❌ 失敗 - 狀態碼: {response.status_code}")
                print(f"     錯誤: {response.text}")
                return False
                
        except Exception as e:
            print(f"     ❌ 異常: {e}")
            return False
    
    return True

def main():
    print("🧪 動態RAG功能簡化測試")
    print("=" * 50)
    
    # 檢查API服務
    if not test_api_health():
        print("❌ API服務不可用，請確保後端服務正在運行")
        return
    
    print("✅ API服務正常")
    
    # 測試傳統RAG
    traditional_success = test_traditional_rag()
    
    # 測試動態RAG
    dynamic_success = test_dynamic_rag()
    
    # 總結
    print("\n" + "=" * 50)
    print("📊 測試結果總結")
    print("=" * 50)
    print(f"傳統RAG: {'✅ 通過' if traditional_success else '❌ 失敗'}")
    print(f"動態RAG: {'✅ 通過' if dynamic_success else '❌ 失敗'}")
    
    if traditional_success and dynamic_success:
        print("\n🎉 所有測試通過！")
    else:
        print("\n⚠️  部分測試失敗，請檢查配置和服務狀態")

if __name__ == "__main__":
    main()