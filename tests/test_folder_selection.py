#!/usr/bin/env python
"""
測試文件夾選擇功能
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

def test_folder_api():
    """測試文件夾 API"""
    print("🧪 測試多層級文件夾 API...")
    
    try:
        # 測試獲取根目錄文件夾列表
        response = requests.get(f"{API_URL}/api/folders", timeout=15)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 成功獲取根目錄文件夾列表")
            print(f"📁 總文件夾數: {data['total_folders']}")
            print(f"📄 根目錄文件數: {data['files_count']}")
            print(f"💾 根目錄總大小: {data.get('size_mb', 0):.1f} MB")
            
            if data['folders']:
                print("\n📂 文件夾列表（前5個）:")
                for i, folder in enumerate(data['folders'][:5]):
                    print(f"  {i+1}. 📁 {folder['name']}")
                    print(f"     📄 文件數: {folder['files_count']}")
                    print(f"     💾 大小: {folder.get('size_mb', 0):.1f} MB")
                
                # 測試多層級導航
                first_folder = data['folders'][0]
                print(f"\n🔍 測試進入子文件夾: {first_folder['name']}")
                
                sub_response = requests.get(f"{API_URL}/api/folders?path={first_folder['path']}", timeout=15)
                if sub_response.status_code == 200:
                    sub_data = sub_response.json()
                    print(f"✅ 成功獲取子文件夾列表")
                    print(f"📍 當前路徑: {sub_data['current_path']}")
                    print(f"📁 子文件夾數: {sub_data['total_folders']}")
                    print(f"📄 文件數: {sub_data['files_count']}")
                    print(f"💾 總大小: {sub_data.get('size_mb', 0):.1f} MB")
                    
                    # 顯示路徑導航
                    if sub_data.get('path_parts'):
                        path_display = " > ".join([part["name"] for part in sub_data["path_parts"]])
                        print(f"🧭 路徑導航: 根目錄 > {path_display}")
                    
                    # 測試返回上級
                    if sub_data.get('parent_path') is not None:
                        print(f"⬆️ 上級路徑: {sub_data['parent_path']}")
                    
                    # 如果有更深層的文件夾，測試再進入一層
                    if sub_data['folders']:
                        deep_folder = sub_data['folders'][0]
                        print(f"\n🔍 測試進入更深層文件夾: {deep_folder['name']}")
                        
                        deep_response = requests.get(f"{API_URL}/api/folders?path={deep_folder['path']}", timeout=15)
                        if deep_response.status_code == 200:
                            deep_data = deep_response.json()
                            print(f"✅ 成功獲取深層文件夾列表")
                            print(f"📍 當前路徑: {deep_data['current_path']}")
                            print(f"📄 文件數: {deep_data['files_count']}")
                            print(f"💾 總大小: {deep_data.get('size_mb', 0):.1f} MB")
                        else:
                            print(f"❌ 獲取深層文件夾失敗: {deep_response.status_code}")
                else:
                    print(f"❌ 獲取子文件夾失敗: {sub_response.status_code}")
            else:
                print("⚠️ 沒有找到文件夾")
        else:
            print(f"❌ API 調用失敗: {response.status_code}")
            print(f"錯誤信息: {response.text}")
    
    except Exception as e:
        print(f"❌ 測試失敗: {e}")

def test_folder_query():
    """測試帶文件夾路徑的查詢"""
    print("\n🧪 測試帶文件夾路徑的查詢...")
    
    try:
        # 先獲取文件夾列表
        folders_response = requests.get(f"{API_URL}/api/folders", timeout=10)
        if folders_response.status_code != 200:
            print("❌ 無法獲取文件夾列表")
            return
        
        folders_data = folders_response.json()
        if not folders_data['folders']:
            print("⚠️ 沒有可用的文件夾進行測試")
            return
        
        # 選擇第一個文件夾進行測試
        test_folder = folders_data['folders'][0]
        print(f"🎯 測試文件夾: {test_folder['name']} ({test_folder['files_count']} 個文件)")
        
        # 測試查詢
        query_payload = {
            "question": "測試查詢",
            "use_dynamic_rag": True,
            "ollama_model": "llama3.2:3b",
            "ollama_embedding_model": "nomic-embed-text",
            "folder_path": test_folder['path'],
            "language": "繁體中文"
        }
        
        print("📤 發送查詢請求...")
        query_response = requests.post(f"{API_URL}/ask", json=query_payload, timeout=30)
        
        if query_response.status_code == 200:
            result = query_response.json()
            print("✅ 查詢成功")
            print(f"📝 回答: {result.get('answer', '無回答')[:100]}...")
            print(f"📚 來源數量: {len(result.get('sources', []))}")
        else:
            print(f"❌ 查詢失敗: {query_response.status_code}")
            print(f"錯誤信息: {query_response.text}")
    
    except Exception as e:
        print(f"❌ 測試失敗: {e}")

def main():
    """主函數"""
    print("🚀 開始測試文件夾選擇功能")
    
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
    test_folder_api()
    test_folder_query()
    
    print("\n🎉 測試完成")

if __name__ == "__main__":
    main()