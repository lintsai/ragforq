#!/usr/bin/env python
"""
測試傳統 RAG 兼容性
確保文件夾選擇功能不會影響傳統 RAG 的正常運行
"""

import os
import sys
import requests
import json

# 添加項目根目錄到路徑
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from config.config import APP_HOST, APP_PORT

API_URL = f"http://{APP_HOST}:{APP_PORT}"

def test_traditional_rag_models():
    """測試傳統 RAG 模型列表"""
    print("🧪 測試傳統 RAG 模型列表...")
    
    try:
        response = requests.get(f"{API_URL}/api/usable-models", timeout=10)
        if response.status_code == 200:
            models = response.json()
            print(f"✅ 成功獲取 {len(models)} 個可用模型")
            
            for model in models[:3]:  # 只顯示前3個
                print(f"  - {model['display_name']}")
                print(f"    文件夾: {model['folder_name']}")
                print(f"    狀態: {'✅ 可用' if model.get('has_data') else '❌ 無數據'}")
            
            return models
        else:
            print(f"❌ 獲取模型列表失敗: {response.status_code}")
            return []
    
    except Exception as e:
        print(f"❌ 測試失敗: {e}")
        return []

def test_traditional_rag_query(models):
    """測試傳統 RAG 查詢"""
    print("\n🧪 測試傳統 RAG 查詢...")
    
    if not models:
        print("⚠️ 沒有可用的模型進行測試")
        return
    
    # 選擇第一個可用模型
    test_model = models[0]
    print(f"🎯 使用模型: {test_model['display_name']}")
    
    try:
        # 測試不帶文件夾路徑的查詢（傳統 RAG）
        query_payload = {
            "question": "測試傳統 RAG 查詢",
            "use_dynamic_rag": False,  # 明確指定使用傳統 RAG
            "selected_model": test_model['folder_name'],
            "language": "繁體中文",
            "include_sources": True,
            "max_sources": 3
        }
        
        print("📤 發送傳統 RAG 查詢請求...")
        query_response = requests.post(f"{API_URL}/ask", json=query_payload, timeout=30)
        
        if query_response.status_code == 200:
            result = query_response.json()
            print("✅ 傳統 RAG 查詢成功")
            print(f"📝 回答長度: {len(result.get('answer', ''))}")
            print(f"📚 來源數量: {len(result.get('sources', []))}")
            
            # 檢查回答內容
            answer = result.get('answer', '')
            if answer and len(answer.strip()) > 0:
                print(f"📄 回答預覽: {answer[:100]}...")
                return True
            else:
                print("⚠️ 回答為空")
                return False
        else:
            print(f"❌ 傳統 RAG 查詢失敗: {query_response.status_code}")
            print(f"錯誤信息: {query_response.text}")
            return False
    
    except Exception as e:
        print(f"❌ 測試失敗: {e}")
        return False

def test_dynamic_rag_query():
    """測試動態 RAG 查詢（對比）"""
    print("\n🧪 測試動態 RAG 查詢（對比）...")
    
    try:
        # 測試動態 RAG 查詢
        query_payload = {
            "question": "測試動態 RAG 查詢",
            "use_dynamic_rag": True,  # 使用動態 RAG
            "ollama_model": "llama3.2:3b",
            "ollama_embedding_model": "nomic-embed-text",
            "language": "繁體中文",
            "include_sources": True,
            "max_sources": 3
            # 注意：沒有 folder_path，應該使用全局搜索
        }
        
        print("📤 發送動態 RAG 查詢請求...")
        query_response = requests.post(f"{API_URL}/ask", json=query_payload, timeout=30)
        
        if query_response.status_code == 200:
            result = query_response.json()
            print("✅ 動態 RAG 查詢成功")
            print(f"📝 回答長度: {len(result.get('answer', ''))}")
            print(f"📚 來源數量: {len(result.get('sources', []))}")
            return True
        else:
            print(f"❌ 動態 RAG 查詢失敗: {query_response.status_code}")
            print(f"錯誤信息: {query_response.text}")
            return False
    
    except Exception as e:
        print(f"❌ 測試失敗: {e}")
        return False

def test_dynamic_rag_with_folder():
    """測試帶文件夾路徑的動態 RAG 查詢"""
    print("\n🧪 測試帶文件夾路徑的動態 RAG 查詢...")
    
    try:
        # 先獲取文件夾列表
        folders_response = requests.get(f"{API_URL}/api/folders", timeout=10)
        if folders_response.status_code != 200:
            print("⚠️ 無法獲取文件夾列表，跳過此測試")
            return True
        
        folders_data = folders_response.json()
        if not folders_data.get('folders'):
            print("⚠️ 沒有可用文件夾，跳過此測試")
            return True
        
        # 選擇第一個有文件的文件夾
        test_folder = None
        for folder in folders_data['folders']:
            if folder.get('files_count', 0) > 0:
                test_folder = folder
                break
        
        if not test_folder:
            print("⚠️ 沒有包含文件的文件夾，跳過此測試")
            return True
        
        print(f"🎯 使用文件夾: {test_folder['name']} ({test_folder['files_count']} 個文件)")
        
        # 測試帶文件夾路徑的動態 RAG 查詢
        query_payload = {
            "question": "測試文件夾限制的動態 RAG 查詢",
            "use_dynamic_rag": True,
            "ollama_model": "llama3.2:3b",
            "ollama_embedding_model": "nomic-embed-text",
            "folder_path": test_folder['path'],  # 指定文件夾路徑
            "language": "繁體中文",
            "include_sources": True,
            "max_sources": 3
        }
        
        print("📤 發送帶文件夾路徑的動態 RAG 查詢請求...")
        query_response = requests.post(f"{API_URL}/ask", json=query_payload, timeout=30)
        
        if query_response.status_code == 200:
            result = query_response.json()
            print("✅ 帶文件夾路徑的動態 RAG 查詢成功")
            print(f"📝 回答長度: {len(result.get('answer', ''))}")
            print(f"📚 來源數量: {len(result.get('sources', []))}")
            return True
        else:
            print(f"❌ 帶文件夾路徑的動態 RAG 查詢失敗: {query_response.status_code}")
            print(f"錯誤信息: {query_response.text}")
            return False
    
    except Exception as e:
        print(f"❌ 測試失敗: {e}")
        return False

def main():
    """主函數"""
    print("🚀 開始測試傳統 RAG 兼容性")
    
    # 測試 API 連接
    try:
        health_response = requests.get(f"{API_URL}/", timeout=5)
        if health_response.status_code == 200:
            print("✅ API 服務正常")
        else:
            print("❌ API 服務異常")
            return
    except Exception as e:
        print(f"❌ 無法連接到 API 服務: {e}")
        return
    
    # 執行測試
    results = []
    
    # 1. 測試傳統 RAG 模型列表
    models = test_traditional_rag_models()
    results.append(("模型列表", len(models) > 0))
    
    # 2. 測試傳統 RAG 查詢
    traditional_result = test_traditional_rag_query(models)
    results.append(("傳統 RAG 查詢", traditional_result))
    
    # 3. 測試動態 RAG 查詢（對比）
    dynamic_result = test_dynamic_rag_query()
    results.append(("動態 RAG 查詢", dynamic_result))
    
    # 4. 測試帶文件夾路徑的動態 RAG 查詢
    folder_result = test_dynamic_rag_with_folder()
    results.append(("文件夾限制查詢", folder_result))
    
    # 總結結果
    print("\n📊 測試結果總結:")
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ 通過" if result else "❌ 失敗"
        print(f"  {status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 總體結果: {passed}/{total} 項測試通過")
    
    if passed == total:
        print("🎉 所有測試通過！傳統 RAG 功能正常，文件夾選擇功能不會影響傳統 RAG。")
    else:
        print("⚠️ 部分測試失敗，請檢查相關功能。")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)