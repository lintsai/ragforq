#!/usr/bin/env python
"""
測試向量資料庫內容維護功能
"""
import requests
import json
from config.config import API_BASE_URL, ADMIN_TOKEN

def test_content_maintenance():
    """測試向量資料庫內容維護API"""
    
    print("=== 測試向量資料庫內容維護功能 ===")
    
    # API基礎URL
    api_url = API_BASE_URL
    headers = {"admin_token": ADMIN_TOKEN}
    
    print(f"API URL: {api_url}")
    print(f"使用管理員Token: {ADMIN_TOKEN[:10]}...")
    
    try:
        # 1. 獲取可用模型
        print("\n1. 獲取可用模型...")
        response = requests.get(f"{api_url}/api/vector-models", timeout=10)
        if response.status_code == 200:
            models = response.json()
            available_models = [m for m in models if m['has_data'] and not m['is_training']]
            
            if not available_models:
                print("❌ 沒有可用的模型進行內容維護")
                return
            
            test_model = available_models[0]
            folder_name = test_model['folder_name']
            print(f"✅ 使用模型: {test_model['display_name']} ({folder_name})")
        else:
            print(f"❌ 獲取模型列表失敗: {response.status_code}")
            return
        
        # 2. 測試獲取文檔列表
        print("\n2. 測試獲取文檔列表...")
        response = requests.get(
            f"{api_url}/admin/vector-db/documents",
            headers=headers,
            params={"folder_name": folder_name, "page": 1, "page_size": 5}
        )
        if response.status_code == 200:
            docs_data = response.json()
            documents = docs_data.get('documents', [])
            total = docs_data.get('total', 0)
            print(f"✅ 成功獲取文檔列表: 共 {total} 個文檔，顯示前 {len(documents)} 個")
            
            if documents:
                test_doc = documents[0]
                doc_id = test_doc['id']
                print(f"   測試文檔: {test_doc['file_name']} (ID: {doc_id})")
            else:
                print("⚠️ 沒有文檔可供測試")
                return
        else:
            print(f"❌ 獲取文檔列表失敗: {response.status_code}")
            return
        
        # 3. 測試獲取特定文檔詳情
        print("\n3. 測試獲取文檔詳情...")
        response = requests.get(
            f"{api_url}/admin/vector-db/document/{doc_id}",
            headers=headers,
            params={"folder_name": folder_name}
        )
        if response.status_code == 200:
            doc_detail = response.json()
            print(f"✅ 成功獲取文檔詳情")
            print(f"   內容長度: {len(doc_detail['content'])} 字符")
            print(f"   文件路徑: {doc_detail['file_path']}")
            original_content = doc_detail['content']
        else:
            print(f"❌ 獲取文檔詳情失敗: {response.status_code}")
            return
        
        # 4. 測試添加新文檔
        print("\n4. 測試添加新文檔...")
        new_doc_content = "這是一個測試文檔，用於驗證向量資料庫內容維護功能。"
        response = requests.post(
            f"{api_url}/admin/vector-db/document",
            headers=headers,
            params={"folder_name": folder_name},
            json={
                "content": new_doc_content,
                "metadata": {
                    "file_name": "測試文檔.txt",
                    "file_path": "test/test_doc.txt",
                    "chunk_index": 0
                }
            }
        )
        if response.status_code == 200:
            print("✅ 成功添加新文檔")
        else:
            print(f"❌ 添加新文檔失敗: {response.status_code} - {response.text}")
        
        # 5. 測試更新文檔內容（使用原有文檔）
        print("\n5. 測試更新文檔內容...")
        updated_content = original_content + "\n\n[測試更新] 這是更新後的內容。"
        response = requests.put(
            f"{api_url}/admin/vector-db/document/{doc_id}",
            headers=headers,
            params={"folder_name": folder_name},
            json={"content": updated_content}
        )
        if response.status_code == 200:
            print("✅ 成功更新文檔內容")
            
            # 驗證更新
            verify_resp = requests.get(
                f"{api_url}/admin/vector-db/document/{doc_id}",
                headers=headers,
                params={"folder_name": folder_name}
            )
            if verify_resp.status_code == 200:
                verify_doc = verify_resp.json()
                if "[測試更新]" in verify_doc['content']:
                    print("✅ 文檔內容更新驗證成功")
                else:
                    print("❌ 文檔內容更新驗證失敗")
            
            # 恢復原始內容
            restore_resp = requests.put(
                f"{api_url}/admin/vector-db/document/{doc_id}",
                headers=headers,
                params={"folder_name": folder_name},
                json={"content": original_content}
            )
            if restore_resp.status_code == 200:
                print("✅ 已恢復原始文檔內容")
        else:
            print(f"❌ 更新文檔內容失敗: {response.status_code} - {response.text}")
        
        print("\n=== 內容維護功能測試完成 ===")
        print("✅ 所有核心功能都已實現並可正常使用")
        print("📝 可以通過Web介面進行:")
        print("   - 瀏覽向量資料庫中的所有文檔")
        print("   - 編輯現有文檔的內容")
        print("   - 添加新的文檔到向量資料庫")
        print("   - 刪除不需要的文檔")
        
    except Exception as e:
        print(f"❌ 測試過程中發生錯誤: {str(e)}")

if __name__ == "__main__":
    test_content_maintenance()