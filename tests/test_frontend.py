#!/usr/bin/env python
"""
簡化版前端測試 - 只測試向量資料庫維護功能
"""
import streamlit as st
import requests
from config.config import API_BASE_URL, APP_HOST, APP_PORT

# API端點
API_URL = f"http://{APP_HOST}:{APP_PORT}"

st.set_page_config(
    page_title="向量資料庫維護測試",
    page_icon="🗄️",
    layout="wide"
)

st.header("🗄️ 向量資料庫維護測試")

admin_token = st.text_input("管理員Token", type="password", key="admin_token_test")

if admin_token:
    st.success("已輸入Token，可操作向量資料庫維護功能")
    
    # 資料庫概覽
    st.subheader("📊 資料庫概覽")
    
    try:
        vector_models_resp = requests.get(f"{API_URL}/api/vector-models", timeout=10)
        if vector_models_resp.status_code == 200:
            vector_models = vector_models_resp.json()
            
            if vector_models:
                # 統計信息
                total_models = len(vector_models)
                models_with_data = sum(1 for m in vector_models if m['has_data'])
                training_models = sum(1 for m in vector_models if m['is_training'])
                usable_models = sum(1 for m in vector_models if m['has_data'] and not m['is_training'])
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("總模型數", total_models)
                with col2:
                    st.metric("有數據模型", models_with_data)
                with col3:
                    st.metric("訓練中模型", training_models)
                with col4:
                    st.metric("可用模型", usable_models)
                
                st.markdown("---")
                
                # 模型詳細管理
                st.subheader("🔧 模型管理")
                
                for model in vector_models:
                    with st.expander(f"📁 {model['display_name']}", expanded=False):
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            st.write(f"**資料夾名稱:** {model['folder_name']}")
                            if model['model_info']:
                                st.write(f"**語言模型:** {model['model_info'].get('OLLAMA_MODEL', '未知')}")
                                st.write(f"**嵌入模型:** {model['model_info'].get('OLLAMA_EMBEDDING_MODEL', '未知')}")
                                if model['model_info'].get('created_at'):
                                    st.write(f"**創建時間:** {model['model_info']['created_at']}")
                            
                            # 狀態顯示
                            status_cols = st.columns(3)
                            with status_cols[0]:
                                if model['has_data']:
                                    st.success("✅ 有數據")
                                else:
                                    st.warning("⚠️ 無數據")
                            
                            with status_cols[1]:
                                if model['is_training']:
                                    st.warning("⏳ 訓練中")
                                else:
                                    st.success("✅ 可用")
                            
                            with status_cols[2]:
                                if model['has_data'] and not model['is_training']:
                                    st.success("🟢 可問答")
                                else:
                                    st.error("🔴 不可問答")
                        
                        with col2:
                            # 操作按鈕
                            if model['has_data']:
                                # 查看詳細信息按鈕
                                if st.button("📋 詳細信息", key=f"info_{model['folder_name']}"):
                                    try:
                                        info_resp = requests.get(
                                            f"{API_URL}/admin/vector-db/info",
                                            headers={"admin_token": admin_token},
                                            params={"folder_name": model['folder_name']}
                                        )
                                        if info_resp.status_code == 200:
                                            info_data = info_resp.json()
                                            st.json(info_data)
                                        else:
                                            st.error(f"獲取詳細信息失敗: {info_resp.text}")
                                    except Exception as e:
                                        st.error(f"獲取詳細信息失敗: {e}")
                                
                                # 備份按鈕
                                if st.button("💾 備份", key=f"backup_{model['folder_name']}"):
                                    try:
                                        backup_resp = requests.post(
                                            f"{API_URL}/admin/vector-db/backup",
                                            headers={"admin_token": admin_token},
                                            json={"folder_name": model['folder_name']}
                                        )
                                        if backup_resp.status_code == 200:
                                            result = backup_resp.json()
                                            st.success(f"✅ 備份成功: {result.get('backup_path', '未知路徑')}")
                                        else:
                                            st.error(f"備份失敗: {backup_resp.text}")
                                    except Exception as e:
                                        st.error(f"備份操作失敗: {e}")
            else:
                st.info("目前沒有任何向量模型")
        else:
            st.error("無法獲取向量模型列表")
    except Exception as e:
        st.error(f"獲取向量模型列表失敗: {e}")
    
    # 批量操作
    st.markdown("---")
    st.subheader("🔄 批量操作")
    
    batch_cols = st.columns(3)
    
    with batch_cols[0]:
        if st.button("🧹 清理空資料夾", key="cleanup_empty_test"):
            try:
                cleanup_resp = requests.post(
                    f"{API_URL}/admin/vector-db/cleanup-empty",
                    headers={"admin_token": admin_token}
                )
                if cleanup_resp.status_code == 200:
                    result = cleanup_resp.json()
                    st.success(f"✅ 清理完成: 清理了 {result.get('cleaned_count', 0)} 個空資料夾")
                else:
                    st.error(f"清理失敗: {cleanup_resp.text}")
            except Exception as e:
                st.error(f"清理操作失敗: {e}")
    
    with batch_cols[1]:
        if st.button("📊 統計資訊", key="stats_test"):
            try:
                stats_resp = requests.get(
                    f"{API_URL}/admin/vector-db/stats",
                    headers={"admin_token": admin_token}
                )
                if stats_resp.status_code == 200:
                    stats = stats_resp.json()
                    st.json(stats)
                else:
                    st.error(f"獲取統計失敗: {stats_resp.text}")
            except Exception as e:
                st.error(f"獲取統計失敗: {e}")
    
    with batch_cols[2]:
        if st.button("🔍 檢查完整性", key="integrity_check_test"):
            try:
                check_resp = requests.get(
                    f"{API_URL}/admin/vector-db/integrity-check",
                    headers={"admin_token": admin_token}
                )
                if check_resp.status_code == 200:
                    result = check_resp.json()
                    if result.get('all_valid', True):
                        st.success("✅ 所有模型完整性檢查通過")
                    else:
                        st.warning("⚠️ 發現完整性問題")
                        st.json(result.get('issues', []))
                else:
                    st.error(f"完整性檢查失敗: {check_resp.text}")
            except Exception as e:
                st.error(f"完整性檢查失敗: {e}")
                
else:
    st.warning("請輸入管理員Token以使用向量資料庫維護功能")