#!/usr/bin/env python
"""
測試前端修復
模擬前端調用API的行為
"""

import os
import sys
import requests
import json

# 添加項目根目錄到路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.ollama_utils import ollama_utils

def test_ollama_utils():
    """測試ollama_utils工具"""
    print("=== 測試ollama_utils工具 ===")
    
    try:
        # 檢查連接
        is_connected = ollama_utils.check_ollama_connection()
        print(f"Ollama連接狀態: {'✅ 正常' if is_connected else '❌ 異常'}")
        
        if not is_connected:
            print("請確保Ollama服務正在運行")
            return False
        
        # 獲取模型列表
        models = ollama_utils.get_available_models()
        print(f"找到 {len(models)} 個模型:")
        
        for model in models:
            print(f"  - {model['name']} (大小: {model['size']} bytes)")
        
        # 獲取模型名稱
        model_names = ollama_utils.get_model_names()
        print(f"模型名稱列表: {model_names}")
        
        return True
        
    except Exception as e:
        print(f"❌ ollama_utils測試失敗: {str(e)}")
        return False

def test_model_categorization():
    """測試模型分類邏輯"""
    print("\n=== 測試模型分類邏輯 ===")
    
    try:
        models = ollama_utils.get_available_models()
        
        if not models:
            print("❌ 沒有找到模型")
            return False
        
        # 分類模型
        language_models = []
        embedding_models = []
        
        for model in models:
            model_name = model['name'].lower()
            
            # 根據模型名稱判斷類型
            if any(embed_keyword in model_name for embed_keyword in ['embed', 'embedding', 'nomic']):
                embedding_models.append(model['name'])
                print(f"嵌入模型: {model['name']}")
            else:
                language_models.append(model['name'])
                print(f"語言模型: {model['name']}")
        
        print(f"\n分類結果:")
        print(f"語言模型 ({len(language_models)}): {language_models}")
        print(f"嵌入模型 ({len(embedding_models)}): {embedding_models}")
        
        # 檢查是否有足夠的模型
        if not language_models:
            print("⚠️ 沒有找到語言模型")
        if not embedding_models:
            print("⚠️ 沒有找到嵌入模型")
        
        return len(language_models) > 0 and len(embedding_models) > 0
        
    except Exception as e:
        print(f"❌ 模型分類測試失敗: {str(e)}")
        return False

def simulate_frontend_api_call():
    """模擬前端API調用"""
    print("\n=== 模擬前端API調用 ===")
    
    # 模擬分類API調用的邏輯
    try:
        models = ollama_utils.get_available_models()
        
        # 分類模型（與API端點相同的邏輯）
        language_models = []
        embedding_models = []
        
        for model in models:
            model_name = model['name'].lower()
            
            if any(embed_keyword in model_name for embed_keyword in ['embed', 'embedding', 'nomic']):
                embedding_models.append(model['name'])
            else:
                language_models.append(model['name'])
        
        # 模擬API響應
        api_response = {
            "language_models": language_models,
            "embedding_models": embedding_models
        }
        
        print("模擬API響應:")
        print(json.dumps(api_response, indent=2, ensure_ascii=False))
        
        # 模擬前端處理邏輯
        print("\n模擬前端處理:")
        
        if api_response.get('language_models') and len(api_response['language_models']) > 0:
            selected_language_model = api_response['language_models'][0]
            print(f"✅ 選擇語言模型: {selected_language_model}")
        else:
            print("❌ 沒有可用的語言模型")
            selected_language_model = None
        
        if api_response.get('embedding_models') and len(api_response['embedding_models']) > 0:
            selected_embedding_model = api_response['embedding_models'][0]
            print(f"✅ 選擇嵌入模型: {selected_embedding_model}")
        else:
            print("❌ 沒有可用的嵌入模型")
            selected_embedding_model = None
        
        return selected_language_model is not None and selected_embedding_model is not None
        
    except Exception as e:
        print(f"❌ 前端API調用模擬失敗: {str(e)}")
        return False

def main():
    """主測試函數"""
    print("開始前端修復測試...")
    
    tests = [
        ("ollama_utils工具", test_ollama_utils),
        ("模型分類邏輯", test_model_categorization),
        ("前端API調用模擬", simulate_frontend_api_call)
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
    print("前端修復測試結果")
    print('='*50)
    print(f"通過: {passed}/{total}")
    print(f"成功率: {passed/total*100:.1f}%")
    
    if passed == total:
        print("🎉 前端修復測試全部通過！")
        print("\n下一步:")
        print("1. 啟動API服務: python app.py")
        print("2. 啟動前端服務: streamlit run frontend/streamlit_app.py")
        print("3. 在前端選擇Dynamic RAG模式進行測試")
    elif passed > 0:
        print("⚠️ 部分測試通過，基本功能正常")
    else:
        print("❌ 測試失敗，請檢查Ollama服務")

if __name__ == "__main__":
    main()