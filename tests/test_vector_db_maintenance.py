#!/usr/bin/env python
"""
測試向量資料庫維護功能
"""
import requests
import json
from config.config import API_BASE_URL, ADMIN_TOKEN

def test_vector_db_maintenance():
    """測試向量資料庫維護API"""
    
    print("=== 測試向量資料庫維護功能 ===")
    
    # API基礎URL
    api_url = API_BASE_URL
    headers = {"admin_token": ADMIN_TOKEN}
    
    print(f"API URL: {api_url}")
    print(f"使用管理員Token: {ADMIN_TOKEN[:10]}...")
    
    try:
        # 1. 測試獲取向量模型列表
        print("\n1. 測試獲取向量模型列表...")
        response = requests.get(f"{api_url}/api/vector-models", timeout=10)
        if response.status_code == 200:
            models = response.json()
            print(f"✅ 成功獲取 {len(models)} 個向量模型")
            for model in models:
                print(f"   - {model['display_name']} ({model['folder_name']})")
                print(f"     有數據: {model['has_data']}, 訓練中: {model['is_training']}")
        else:
            print(f"❌ 獲取向量模型列表失敗: {response.status_code}")
            return
        
        # 2. 測試統計信息
        print("\n2. 測試獲取統計信息...")
        response = requests.get(f"{api_url}/admin/vector-db/stats", headers=headers, timeout=10)
        if response.status_code == 200:
            stats = response.json()
            print("✅ 成功獲取統計信息:")
            print(f"   總模型數: {stats['total_models']}")
            print(f"   總大小: {stats['total_size_mb']} MB")
            print(f"   總文件數: {stats['total_files']}")
            print(f"   狀態統計: {stats['stats_by_status']}")
        else:
            print(f"❌ 獲取統計信息失敗: {response.status_code}")
        
        # 3. 測試完整性檢查
        print("\n3. 測試完整性檢查...")
        response = requests.get(f"{api_url}/admin/vector-db/integrity-check", headers=headers, timeout=10)
        if response.status_code == 200:
            result = response.json()
            if result['all_valid']:
                print("✅ 所有模型完整性檢查通過")
            else:
                print(f"⚠️ 發現 {result['models_with_issues']} 個模型有問題")
                for issue in result['issues']:
                    print(f"   - {issue['model_name']}: {', '.join(issue['issues'])}")
        else:
            print(f"❌ 完整性檢查失敗: {response.status_code}")
        
        # 4. 如果有模型，測試獲取詳細信息
        if models:
            print("\n4. 測試獲取模型詳細信息...")
            first_model = models[0]
            response = requests.get(
                f"{api_url}/admin/vector-db/info",
                headers=headers,
                params={"folder_name": first_model['folder_name']},
                timeout=10
            )
            if response.status_code == 200:
                info = response.json()
                print(f"✅ 成功獲取模型 {first_model['display_name']} 的詳細信息")
                if 'filesystem' in info:
                    fs_info = info['filesystem']
                    print(f"   文件夾大小: {fs_info['folder_size_mb']} MB")
                    print(f"   文件數量: {fs_info['file_count']}")
                if 'key_files' in info:
                    print(f"   關鍵文件: {info['key_files']}")
            else:
                print(f"❌ 獲取模型詳細信息失敗: {response.status_code}")
        
        # 5. 測試清理空資料夾
        print("\n5. 測試清理空資料夾...")
        response = requests.post(f"{api_url}/admin/vector-db/cleanup-empty", headers=headers, timeout=10)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ 清理完成: 清理了 {result['cleaned_count']} 個空資料夾")
            if result['cleaned_folders']:
                print(f"   清理的資料夾: {', '.join(result['cleaned_folders'])}")
        else:
            print(f"❌ 清理空資料夾失敗: {response.status_code}")
        
        print("\n=== 測試完成 ===")
        
    except Exception as e:
        print(f"❌ 測試過程中發生錯誤: {str(e)}")

if __name__ == "__main__":
    test_vector_db_maintenance()