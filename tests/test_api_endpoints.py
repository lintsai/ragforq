#!/usr/bin/env python
"""
測試API端點
"""

import os
import sys
import requests
import json

# 添加項目根目錄到路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_ollama_models_api():
    """測試Ollama模型API"""
    print("=== 測試Ollama模型API ===")
    
    api_url = "http://localhost:8000"
    
    # 測試基本健康檢查
    try:
        response = requests.get(f"{api_url}/health", timeout=5)
        if response.status_code == 200:
            print("✅ API服務正在運行")
        else:
            print(f"❌ API服務狀態異常: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ 無法連接到API服務: {str(e)}")
        print("請先啟動API服務: python app.py")
        return False
    
    # 測試原始Ollama模型端點
    try:
        response = requests.get(f"{api_url}/api/ollama/models", timeout=10)
        if response.status_code == 200:
            models = response.json()
            print(f"✅ 原始Ollama模型端點正常，找到 {len(models)} 個模型")
            for model in models:
                print(f"  - {model['name']}")
        else:
            print(f"❌ 原始Ollama模型端點失敗: {response.status_code}")
            print(f"錯誤信息: {response.text}")
    except Exception as e:
        print(f"❌ 原始Ollama模型端點測試失敗: {str(e)}")
    
    # 測試分類Ollama模型端點
    try:
        response = requests.get(f"{api_url}/api/ollama/models/categorized", timeout=10)
        if response.status_code == 200:
            models = response.json()
            print(f"✅ 分類Ollama模型端點正常")
            print(f"語言模型: {models.get('language_models', [])}")
            print(f"嵌入模型: {models.get('embedding_models', [])}")
            return True
        else:
            print(f"❌ 分類Ollama模型端點失敗: {response.status_code}")
            print(f"錯誤信息: {response.text}")
            return False
    except Exception as e:
        print(f"❌ 分類Ollama模型端點測試失敗: {str(e)}")
        return False

def test_dynamic_rag_api():
    """測試Dynamic RAG API"""
    print("\n=== 測試Dynamic RAG API ===")
    
    api_url = "http://localhost:8000"
    
    # 測試Dynamic RAG問答
    payload = {
        "question": "What is technology?",
        "use_dynamic_rag": True,
        "selected_model": "qwen2:0.5b-instruct",
        "ollama_embedding_model": "nomic-embed-text:latest",
        "language": "English",
        "include_sources": True
    }
    
    try:
        print("發送Dynamic RAG測試請求...")
        response = requests.post(f"{api_url}/ask", json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Dynamic RAG API調用成功")
            print(f"回答長度: {len(result.get('answer', ''))} 字符")
            print(f"來源數量: {len(result.get('sources', []))}")
            print(f"回答預覽: {result.get('answer', '')[:100]}...")
            return True
        else:
            print(f"❌ Dynamic RAG API調用失敗: {response.status_code}")
            print(f"錯誤信息: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Dynamic RAG API測試失敗: {str(e)}")
        return False

def main():
    """主測試函數"""
    print("開始API端點測試...")
    
    tests = [
        ("Ollama模型API", test_ollama_models_api),
        ("Dynamic RAG API", test_dynamic_rag_api)
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
    print("API測試結果總結")
    print('='*50)
    print(f"通過: {passed}/{total}")
    print(f"成功率: {passed/total*100:.1f}%")
    
    if passed == total:
        print("🎉 所有API端點測試通過！")
    elif passed > 0:
        print("⚠️ 部分API端點正常")
    else:
        print("❌ 所有API端點測試失敗")

if __name__ == "__main__":
    main()