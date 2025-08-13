"""
設置流程前端組件
"""

import streamlit as st
import requests
from typing import Dict, Any, Optional
import time

def render_setup_flow(api_url: str) -> bool:
    """
    渲染設置流程界面
    
    Args:
        api_url: API 基礎 URL
        
    Returns:
        是否完成設置
    """
    
    # 獲取設置狀態
    try:
        response = requests.get(f"{api_url}/api/setup/status")
        if response.status_code == 200:
            setup_status = response.json()
        else:
            st.error("無法獲取設置狀態")
            return False
    except Exception as e:
        st.error(f"連接 API 失敗: {str(e)}")
        return False
    
    # 如果設置已完成，返回 True
    if setup_status.get("setup_completed", False):
        return True
    
    # 顯示設置流程
    st.markdown("# 🚀 系統初始設置")
    st.markdown("歡迎使用 Q槽文件智能助手！請完成以下設置步驟。")
    
    # 顯示進度條
    progress = setup_status.get("progress", {})
    progress_bar = st.progress(progress.get("progress_percentage", 0) / 100)
    st.markdown(f"**進度**: {progress.get('current_index', 0) + 1} / {progress.get('total_steps', 5)}")
    
    current_step = setup_status.get("current_step", "platform_selection")
    
    # 根據當前步驟渲染對應界面
    if current_step == "platform_selection":
        return render_platform_selection(api_url)
    elif current_step == "model_selection":
        return render_model_selection(api_url)
    elif current_step == "rag_mode_selection":
        return render_rag_mode_selection(api_url)
    elif current_step == "configuration_review":
        return render_configuration_review(api_url)
    elif current_step == "ready":
        return render_setup_complete(api_url)
    else:
        st.error(f"未知的設置步驟: {current_step}")
        return False

def render_platform_selection(api_url: str) -> bool:
    """渲染平台選擇界面"""
    st.markdown("## 步驟 1: 選擇 AI 平台")
    st.markdown("請選擇您要使用的 AI 模型平台：")
    
    try:
        response = requests.get(f"{api_url}/api/setup/platforms")
        if response.status_code != 200:
            st.error("無法獲取平台列表")
            return False
        
        data = response.json()
        platforms = data.get("platforms", [])
        
        if not platforms:
            st.error("沒有可用的平台")
            return False
        
        # 顯示平台選項
        selected_platform = None
        
        for platform in platforms:
            with st.container():
                col1, col2, col3 = st.columns([1, 6, 2])
                
                with col1:
                    if platform.get("recommended", False):
                        st.markdown("⭐")
                    
                    # 狀態指示器
                    if platform["status"] == "available":
                        st.markdown("🟢")
                    else:
                        st.markdown("🔴")
                
                with col2:
                    st.markdown(f"### {platform['name']}")
                    st.markdown(platform["description"])
                    
                    # 顯示特性
                    if platform.get("features"):
                        st.markdown("**特性:**")
                        for feature in platform["features"]:
                            st.markdown(f"• {feature}")
                    
                    if platform["status"] != "available":
                        st.warning("此平台當前不可用")
                
                with col3:
                    if platform["status"] == "available":
                        if st.button(f"選擇", key=f"select_{platform['type']}"):
                            selected_platform = platform["type"]
                    else:
                        st.button("不可用", disabled=True, key=f"disabled_{platform['type']}")
                
                st.markdown("---")
        
        # 處理平台選擇
        if selected_platform:
            with st.spinner("正在設置平台..."):
                try:
                    response = requests.post(
                        f"{api_url}/api/setup/platform",
                        json={"platform_type": selected_platform}
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        if result.get("success"):
                            st.success(result.get("message", "平台設置成功"))
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error(result.get("message", "平台設置失敗"))
                    else:
                        st.error("設置平台失敗")
                except Exception as e:
                    st.error(f"設置平台時出錯: {str(e)}")
        
    except Exception as e:
        st.error(f"獲取平台列表失敗: {str(e)}")
    
    return False

def render_model_selection(api_url: str) -> bool:
    """渲染模型選擇界面"""
    st.markdown("## 步驟 2: 選擇 AI 模型")
    st.markdown("請選擇語言模型和嵌入模型：")
    
    try:
        response = requests.get(f"{api_url}/api/setup/models")
        if response.status_code != 200:
            st.error("無法獲取模型列表")
            return False
        
        data = response.json()
        
        if "error" in data:
            st.error(data["error"])
            return False
        
        models = data.get("models", {})
        recommended = data.get("recommended", {})
        current_selection = data.get("current_selection", {})
        
        # 顯示推薦配置
        if recommended:
            st.info(f"💡 **推薦配置**: {recommended.get('reason', '')}")
        
        # 語言模型選擇
        st.markdown("### 🤖 語言模型")
        language_models = models.get("language_models", [])
        
        if not language_models:
            st.error("沒有找到本地語言模型")
            st.info("請先下載模型到 `models/cache` 目錄：")
            st.code("hf download Qwen/Qwen2-0.5B-Instruct --cache-dir ./models/cache")
            return False
        
        # 創建語言模型選項
        language_model_options = []
        language_model_map = {}
        
        for model in language_models:
            display_name = f"{model['name']} ({model['size']})"
            language_model_options.append(display_name)
            language_model_map[display_name] = model["id"]
        
        selected_language_display = st.selectbox(
            "選擇語言模型:",
            options=language_model_options,
            help="語言模型用於生成回答"
        )
        
        selected_language_model = language_model_map[selected_language_display]
        
        # 顯示選中模型的詳細信息
        selected_model_info = next(
            (m for m in language_models if m["id"] == selected_language_model), 
            None
        )
        
        if selected_model_info:
            with st.expander("模型詳細信息", expanded=False):
                st.markdown(f"**描述**: {selected_model_info.get('description', 'N/A')}")
                st.markdown(f"**大小**: {selected_model_info.get('size', 'N/A')}")
                
                if selected_model_info.get("path"):
                    st.markdown(f"**本地路徑**: `{selected_model_info['path']}`")
                
                if selected_model_info.get("requirements"):
                    st.markdown("**硬體需求**:")
                    for key, value in selected_model_info["requirements"].items():
                        st.markdown(f"• {key}: {value}")
        
        # 嵌入模型選擇
        st.markdown("### 🔤 嵌入模型")
        embedding_models = models.get("embedding_models", [])
        
        if not embedding_models:
            st.error("沒有找到本地嵌入模型")
            st.info("請先下載模型到 `models/cache` 目錄：")
            st.code("hf download sentence-transformers/paraphrase-multilingual-mpnet-base-v2 --cache-dir ./models/cache")
            return False
        
        # 創建嵌入模型選項
        embedding_model_options = []
        embedding_model_map = {}
        
        for model in embedding_models:
            display_name = f"{model['name']} ({model['size']})"
            embedding_model_options.append(display_name)
            embedding_model_map[display_name] = model["id"]
        
        selected_embedding_display = st.selectbox(
            "選擇嵌入模型:",
            options=embedding_model_options,
            help="嵌入模型用於文本向量化"
        )
        
        selected_embedding_model = embedding_model_map[selected_embedding_display]
        
        # 顯示選中嵌入模型的詳細信息
        selected_embedding_info = next(
            (m for m in embedding_models if m["id"] == selected_embedding_model), 
            None
        )
        
        if selected_embedding_info:
            with st.expander("嵌入模型詳細信息", expanded=False):
                st.markdown(f"**描述**: {selected_embedding_info.get('description', 'N/A')}")
                st.markdown(f"**大小**: {selected_embedding_info.get('size', 'N/A')}")
                
                if selected_embedding_info.get("path"):
                    st.markdown(f"**本地路徑**: `{selected_embedding_info['path']}`")
                
                if selected_embedding_info.get("languages"):
                    st.markdown(f"**支援語言**: {', '.join(selected_embedding_info['languages'])}")
        
        # 推理引擎選擇（僅 Hugging Face 平台）
        inference_engine = "transformers"  # 默認值
        
        if data.get("platform_type") == "huggingface":
            st.markdown("### ⚙️ 推理引擎")
            
            engine_options = [
                ("transformers", "Transformers", "標準推理引擎，穩定可靠，適合開發和測試"),
                ("vllm", "vLLM", "高性能推理引擎，適合生產環境，需要更多 GPU 記憶體")
            ]
            
            for engine_id, engine_name, engine_desc in engine_options:
                col1, col2, col3 = st.columns([1, 3, 1])
                
                with col1:
                    if engine_id == "vllm":
                        st.markdown("⚡")  # 高性能標記
                
                with col2:
                    st.markdown(f"**{engine_name}**")
                    st.markdown(engine_desc)
                
                with col3:
                    if st.button(f"選擇", key=f"engine_{engine_id}"):
                        inference_engine = engine_id
            
            st.markdown("---")
        
        # 確認按鈕
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("⬅️ 返回上一步"):
                # 重置到平台選擇
                try:
                    requests.post(f"{api_url}/api/setup/reset")
                    st.rerun()
                except:
                    pass
        
        with col2:
            if st.button("確認模型選擇 ➡️"):
                with st.spinner("正在保存模型選擇..."):
                    try:
                        response = requests.post(
                            f"{api_url}/api/setup/models",
                            json={
                                "language_model": selected_language_model,
                                "embedding_model": selected_embedding_model,
                                "inference_engine": inference_engine
                            }
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            if result.get("success"):
                                st.success(result.get("message", "模型選擇已保存"))
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error(result.get("message", "模型選擇失敗"))
                        else:
                            st.error("保存模型選擇失敗")
                    except Exception as e:
                        st.error(f"保存模型選擇時出錯: {str(e)}")
        
    except Exception as e:
        st.error(f"獲取模型列表失敗: {str(e)}")
    
    return False

def render_rag_mode_selection(api_url: str) -> bool:
    """渲染 RAG 模式選擇界面"""
    st.markdown("## 步驟 3: 選擇 RAG 模式")
    st.markdown("請選擇檢索增強生成 (RAG) 的工作模式：")
    
    try:
        response = requests.get(f"{api_url}/api/setup/rag-modes")
        if response.status_code != 200:
            st.error("無法獲取 RAG 模式列表")
            return False
        
        data = response.json()
        
        if "error" in data:
            st.error(data["error"])
            return False
        
        rag_modes = data.get("rag_modes", [])
        has_vector_db = data.get("has_vector_db", False)
        
        if not rag_modes:
            st.error("沒有可用的 RAG 模式")
            return False
        
        # 顯示向量資料庫狀態
        if has_vector_db:
            st.success("✅ 檢測到可用的向量資料庫")
        else:
            st.warning("⚠️ 沒有檢測到向量資料庫，建議使用動態 RAG")
        
        # 顯示 RAG 模式選項
        selected_mode = None
        
        for mode in rag_modes:
            with st.container():
                col1, col2, col3 = st.columns([1, 6, 2])
                
                with col1:
                    if mode.get("recommended", False):
                        st.markdown("⭐")
                
                with col2:
                    st.markdown(f"### {mode['name']}")
                    st.markdown(mode["description"])
                    
                    # 顯示特性
                    if mode.get("features"):
                        st.markdown("**特性:**")
                        for feature in mode["features"]:
                            st.markdown(f"• {feature}")
                    
                    if mode.get("requirements"):
                        st.markdown(f"**需求**: {mode['requirements']}")
                
                with col3:
                    # 檢查是否可選擇
                    can_select = True
                    if mode["type"] == "traditional" and not has_vector_db:
                        can_select = False
                        st.button("需要向量DB", disabled=True, key=f"disabled_{mode['type']}")
                    else:
                        if st.button("選擇", key=f"select_{mode['type']}"):
                            selected_mode = mode["type"]
                
                st.markdown("---")
        
        # 處理模式選擇
        if selected_mode:
            with st.spinner("正在設置 RAG 模式..."):
                try:
                    response = requests.post(
                        f"{api_url}/api/setup/rag-mode",
                        json={"rag_mode": selected_mode}
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        if result.get("success"):
                            st.success(result.get("message", "RAG 模式設置成功"))
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error(result.get("message", "RAG 模式設置失敗"))
                    else:
                        st.error("設置 RAG 模式失敗")
                except Exception as e:
                    st.error(f"設置 RAG 模式時出錯: {str(e)}")
        
        # 導航按鈕
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("⬅️ 返回上一步"):
                # 這裡可以實現返回邏輯
                pass
        
    except Exception as e:
        st.error(f"獲取 RAG 模式列表失敗: {str(e)}")
    
    return False

def render_configuration_review(api_url: str) -> bool:
    """渲染配置審查界面"""
    st.markdown("## 步驟 4: 配置確認")
    st.markdown("請確認您的配置選擇：")
    
    try:
        response = requests.get(f"{api_url}/api/setup/review")
        if response.status_code != 200:
            st.error("無法獲取配置信息")
            return False
        
        data = response.json()
        
        if "error" in data:
            st.error(data["error"])
            return False
        
        config = data.get("configuration", {})
        requirements = data.get("resource_requirements", {})
        
        # 顯示配置摘要
        st.markdown("### 📋 配置摘要")
        
        # 平台信息
        platform = config.get("platform", {})
        st.markdown(f"**AI 平台**: {platform.get('name', 'N/A')}")
        st.markdown(f"*{platform.get('description', '')}*")
        
        # 模型信息
        language_model = config.get("language_model", {})
        embedding_model = config.get("embedding_model", {})
        
        st.markdown(f"**語言模型**: {language_model.get('name', 'N/A')} ({language_model.get('size', 'N/A')})")
        st.markdown(f"**嵌入模型**: {embedding_model.get('name', 'N/A')} ({embedding_model.get('size', 'N/A')})")
        
        # RAG 模式
        rag_mode = config.get("rag_mode", {})
        st.markdown(f"**RAG 模式**: {rag_mode.get('name', 'N/A')}")
        
        # 資源需求
        st.markdown("### 💻 系統需求")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**GPU 記憶體**: {requirements.get('gpu_memory', 'N/A')}")
            st.markdown(f"**系統記憶體**: {requirements.get('system_memory', 'N/A')}")
        
        with col2:
            st.markdown(f"**儲存空間**: {requirements.get('storage', 'N/A')}")
            st.markdown(f"**網路**: {requirements.get('network', 'N/A')}")
        
        if requirements.get("special_notes"):
            st.warning(f"⚠️ {requirements['special_notes']}")
        
        # 確認按鈕
        st.markdown("---")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("⬅️ 修改配置"):
                # 重置設置
                try:
                    requests.post(f"{api_url}/api/setup/reset")
                    st.rerun()
                except:
                    pass
        
        with col2:
            if st.button("✅ 確認並完成設置"):
                with st.spinner("正在完成設置..."):
                    try:
                        response = requests.post(f"{api_url}/api/setup/complete")
                        
                        if response.status_code == 200:
                            result = response.json()
                            if result.get("success"):
                                st.success(result.get("message", "設置完成"))
                                time.sleep(2)
                                st.rerun()
                            else:
                                st.error(result.get("message", "設置失敗"))
                        else:
                            st.error("完成設置失敗")
                    except Exception as e:
                        st.error(f"完成設置時出錯: {str(e)}")
        
    except Exception as e:
        st.error(f"獲取配置信息失敗: {str(e)}")
    
    return False

def render_setup_complete(api_url: str) -> bool:
    """渲染設置完成界面"""
    st.markdown("## 🎉 設置完成！")
    st.markdown("恭喜！您已成功完成系統設置。")
    
    st.success("✅ 系統已準備就緒，您現在可以開始使用 Q槽文件智能助手。")
    
    # 顯示下一步操作
    st.markdown("### 下一步操作：")
    st.markdown("• 🔍 開始提問並搜索文檔")
    st.markdown("• 📚 查看已索引的文件")
    st.markdown("• ⚙️ 在管理員後台進行進階設置")
    
    if st.button("🚀 開始使用"):
        return True
    
    # 提供重新設置選項
    st.markdown("---")
    if st.button("🔄 重新設置"):
        try:
            requests.post(f"{api_url}/api/setup/reset")
            st.rerun()
        except Exception as e:
            st.error(f"重置設置失敗: {str(e)}")
    
    return True