"""
簡化的模型選擇組件
"""

import streamlit as st
import requests
import json
from pathlib import Path
from typing import Dict, Any, Optional

def load_user_config():
    """載入用戶配置"""
    config_file = Path("config/user_setup.json")
    if config_file.exists():
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            pass
    return {}

def save_user_config(config: Dict[str, Any]):
    """保存用戶配置"""
    config_file = Path("config/user_setup.json")
    config_file.parent.mkdir(exist_ok=True)
    
    try:
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        st.error(f"保存配置失敗: {str(e)}")
        return False

def render_model_selector(api_url: str):
    """渲染簡化的模型選擇器"""
    
    st.sidebar.markdown("## ⚙️ 模型配置")
    
    # 載入當前配置
    current_config = load_user_config()
    
    # 平台選擇
    st.sidebar.markdown("### 🎯 AI 平台")
    platform_options = {
        "huggingface": "🤗 Hugging Face",
        "ollama": "🏠 Ollama"
    }
    
    selected_platform = st.sidebar.selectbox(
        "選擇平台:",
        options=list(platform_options.keys()),
        format_func=lambda x: platform_options[x],
        index=0 if current_config.get("platform") == "huggingface" else 1,
        key="platform_selector"
    )
    
    # 根據平台顯示不同的模型選項
    if selected_platform == "huggingface":
        render_huggingface_models(current_config, api_url)
    else:
        render_ollama_models(current_config)
    
    # RAG 模式選擇 - 隱藏，因為已經在主界面顯示
    # st.sidebar.markdown("### 🔧 RAG 模式")
    # rag_options = {
    #     "traditional": "📚 傳統 RAG (快速)",
    #     "dynamic": "⚡ 動態 RAG (靈活)"
    # }
    # 
    # selected_rag_mode = st.sidebar.selectbox(
    #     "選擇 RAG 模式:",
    #     options=list(rag_options.keys()),
    #     format_func=lambda x: rag_options[x],
    #     index=0 if current_config.get("rag_mode") == "traditional" else 1,
    #     key="model_selector_rag_mode"
    # )
    
    # 使用默認的 RAG 模式（由於主界面已經處理）
    selected_rag_mode = "traditional"
    
    # 語言選擇 - 隱藏，因為在主界面已經處理
    # st.sidebar.markdown("### 🌐 語言設置")
    # language_options = {
    #     "traditional_chinese": "🇹🇼 繁體中文",
    #     "simplified_chinese": "🇨🇳 简体中文",
    #     "english": "🇺🇸 English",
    #     "thai": "🇹🇭 ไทย",
    #     "dynamic": "🌍 自動檢測"
    # }
    # 
    # selected_language = st.sidebar.selectbox(
    #     "選擇語言:",
    #     options=list(language_options.keys()),
    #     format_func=lambda x: language_options[x],
    #     index=4,  # 默認自動檢測
    #     key="model_selector_language_selector"
    # )
    
    # 使用默認語言（由於主界面已經處理）
    selected_language = "dynamic"
    
    # 保存配置按鈕
    if st.sidebar.button("💾 保存配置", key="save_config"):
        new_config = {
            "platform": selected_platform,
            "language_model": st.session_state.get("selected_language_model"),
            "embedding_model": st.session_state.get("selected_embedding_model"),
            "inference_engine": st.session_state.get("selected_inference_engine", "transformers"),
            "rag_mode": selected_rag_mode,
            "language": selected_language,
            "setup_completed": True
        }
        
        if save_user_config(new_config):
            st.sidebar.success("✅ 配置已保存")
            # 通知 API 更新配置
            try:
                response = requests.post(f"{api_url}/api/config/update", json=new_config)
                if response.status_code == 200:
                    st.sidebar.success("✅ 系統配置已更新")
                else:
                    st.sidebar.warning("⚠️ 系統配置更新失敗")
            except:
                st.sidebar.warning("⚠️ 無法連接到 API")
        else:
            st.sidebar.error("❌ 配置保存失敗")

def render_huggingface_models(current_config: Dict[str, Any], api_url: str = "http://localhost:8000"):
    """渲染 Hugging Face 模型選項"""
    
    try:
        # 從 API 獲取可用模型
        response = requests.get(f"{api_url}/api/setup/models", timeout=10)
        if response.status_code == 200:
            data = response.json()
            models = data.get("models", {})
            
            # 語言模型
            st.sidebar.markdown("#### 🤖 語言模型")
            language_models = models.get("language_models", [])
            
            if language_models:
                # 創建選項字典
                hf_language_models = {}
                for model in language_models:
                    display_name = f"{model['name']} ({model['size']})"
                    hf_language_models[model["id"]] = display_name
                
                selected_language_model = st.sidebar.selectbox(
                    "選擇語言模型:",
                    options=list(hf_language_models.keys()),
                    format_func=lambda x: hf_language_models[x],
                    index=0,
                    key="hf_language_model"
                )
                
                st.session_state.selected_language_model = selected_language_model
            else:
                st.sidebar.error("沒有找到本地語言模型")
                st.sidebar.info("請先下載模型")
                st.sidebar.code("hf download Qwen/Qwen2-0.5B-Instruct --cache-dir ./models/cache")
            
            # 嵌入模型
            st.sidebar.markdown("#### 🔤 嵌入模型")
            embedding_models = models.get("embedding_models", [])
            
            if embedding_models:
                # 創建選項字典
                hf_embedding_models = {}
                for model in embedding_models:
                    display_name = f"{model['name']} ({model['size']})"
                    hf_embedding_models[model["id"]] = display_name
                
                selected_embedding_model = st.sidebar.selectbox(
                    "選擇嵌入模型:",
                    options=list(hf_embedding_models.keys()),
                    format_func=lambda x: hf_embedding_models[x],
                    index=0,
                    key="hf_embedding_model"
                )
                
                st.session_state.selected_embedding_model = selected_embedding_model
            else:
                st.sidebar.error("沒有找到本地嵌入模型")
                st.sidebar.info("請先下載模型")
                st.sidebar.code("hf download sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 --cache-dir ./models/cache")
        
        else:
            st.sidebar.error("無法獲取模型列表")
            # 使用備用的硬編碼列表
            _render_fallback_huggingface_models()
    
    except Exception as e:
        st.sidebar.error(f"獲取模型列表時出錯: {str(e)}")
        # 使用備用的硬編碼列表
        _render_fallback_huggingface_models()

def _render_fallback_huggingface_models():
    """渲染備用的 Hugging Face 模型選項（API 失敗時）"""
    st.sidebar.markdown("#### 🤖 語言模型")
    st.sidebar.error("無法獲取模型列表")
    st.sidebar.info("請先下載模型：")
    st.sidebar.code("hf download Qwen/Qwen2-0.5B-Instruct --cache-dir ./models/cache")
    
    # 嵌入模型
    st.sidebar.markdown("#### 🔤 嵌入模型")
    st.sidebar.error("無法獲取模型列表")
    st.sidebar.info("請先下載模型：")
    st.sidebar.code("hf download sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 --cache-dir ./models/cache")
    
    # 設置空的選擇狀態
    st.session_state.selected_language_model = None
    st.session_state.selected_embedding_model = None
    
    # 推理引擎
    st.sidebar.markdown("#### ⚙️ 推理引擎")
    inference_engines = {
        "transformers": "🔧 Transformers (穩定)",
        "vllm": "⚡ vLLM (高性能)"
    }
    
    selected_inference_engine = st.sidebar.selectbox(
        "選擇推理引擎:",
        options=list(inference_engines.keys()),
        format_func=lambda x: inference_engines[x],
        index=0,
        key="hf_inference_engine"
    )
    
    st.session_state.selected_inference_engine = selected_inference_engine

def render_ollama_models(current_config: Dict[str, Any]):
    """渲染 Ollama 模型選項"""
    
    # 語言模型
    st.sidebar.markdown("#### 🤖 語言模型")
    ollama_language_models = {
        "llama3.2:3b": "Llama 3.2 3B (2GB)",
        "llama3.1:8b": "Llama 3.1 8B (4.7GB)",
        "llama3.1:70b": "Llama 3.1 70B (40GB)",
        "qwen2.5:7b": "Qwen 2.5 7B (4.4GB)",
        "gemma2:9b": "Gemma 2 9B (5.4GB)"
    }
    
    selected_language_model = st.sidebar.selectbox(
        "選擇語言模型:",
        options=list(ollama_language_models.keys()),
        format_func=lambda x: ollama_language_models[x],
        index=0,
        key="ollama_language_model"
    )
    
    st.session_state.selected_language_model = selected_language_model
    
    # 嵌入模型
    st.sidebar.markdown("#### 🔤 嵌入模型")
    ollama_embedding_models = {
        "nomic-embed-text": "🌍 Nomic Embed Text (274MB) - 多語言推薦",
        "mxbai-embed-large": "🚀 MxBai Embed Large (669MB) - 高精度"
    }
    
    selected_embedding_model = st.sidebar.selectbox(
        "選擇嵌入模型:",
        options=list(ollama_embedding_models.keys()),
        format_func=lambda x: ollama_embedding_models[x],
        index=0,
        key="ollama_embedding_model"
    )
    
    st.session_state.selected_embedding_model = selected_embedding_model
    
    # Ollama 只使用自己的推理引擎
    st.session_state.selected_inference_engine = "ollama"

def get_current_config():
    """獲取當前配置"""
    return load_user_config()

def is_setup_completed():
    """檢查是否完成設置"""
    config = load_user_config()
    return config.get("setup_completed", False)
