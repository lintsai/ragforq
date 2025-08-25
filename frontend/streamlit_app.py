#!/usr/bin/env python
"""
Streamlit前端界面 - 提供用戶友好的界面來查詢Q槽文件
"""
import os
import sys
import logging
import json
import time
import requests
from typing import List, Dict, Any, Optional
import streamlit as st
from streamlit_autorefresh import st_autorefresh
from datetime import datetime
import pytz

# 添加項目根目錄到路徑
frontend_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(frontend_dir)
sys.path.append(project_root)

import uuid

# 為每個前端 session 建立唯一 ID
if 'frontend_session_id' not in st.session_state:
    st.session_state.frontend_session_id = str(uuid.uuid4())

from config.config import APP_HOST, APP_PORT, STREAMLIT_PORT, API_BASE_URL, is_q_drive_accessible, Q_DRIVE_PATH, DISPLAY_DRIVE_NAME
from frontend.help_system import render_help_sidebar, show_help_modal
from frontend.model_selector import render_model_selector, is_setup_completed
from frontend.folder_browser import FolderBrowser

# 設置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API端點
API_URL = f"http://{APP_HOST}:{APP_PORT}"
ASK_ENDPOINT = f"{API_URL}/ask"
STATUS_ENDPOINT = f"{API_URL}/health"
FILES_ENDPOINT = f"{API_URL}/files"

# 頁面配置
st.set_page_config(
    page_title="Q槽文件智能助手",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定義CSS
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; color: #1E88E5; margin-bottom: 0; }
    .sub-header { font-size: 1.2rem; color: #424242; margin-top: 0; }
    .source-title { font-size: 1.2rem; font-weight: bold; color: #1E88E5; margin-top: 1rem; }
    .source-item { background-color: #f0f2f6; border-radius: 5px; padding: 10px; margin-bottom: 5px; }
    .footer { text-align: center; color: #9e9e9e; font-size: 0.8rem; margin-top: 3rem; }
    .stTextInput>div>div>input { font-size: 1.1rem; }
</style>
""", unsafe_allow_html=True)

# 初始化 session state
if 'api_status' not in st.session_state:
    st.session_state.api_status = None
if 'retry_count' not in st.session_state:
    st.session_state.retry_count = 0
if 'last_error' not in st.session_state:
    st.session_state.last_error = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_question' not in st.session_state:
    st.session_state.current_question = ""
if 'current_answer' not in st.session_state:
    st.session_state.current_answer = None
if 'selected_language' not in st.session_state:
    st.session_state.selected_language = "繁體中文"
if 'dynamic_scope_info' not in st.session_state:
    st.session_state.dynamic_scope_info = None
if 'dynamic_block_recommended' not in st.session_state:
    st.session_state.dynamic_block_recommended = False
if 'dynamic_block_reason' not in st.session_state:
    st.session_state.dynamic_block_reason = None

# 檢查API是否正常運行
def check_api_status() -> bool:
    """檢查 API 服務狀態"""
    try:
        response = requests.get(f"{API_BASE_URL}/")
        if response.status_code == 200:
            st.session_state.api_status = response.json()
            st.session_state.retry_count = 0
            st.session_state.last_error = None
            return True
        else:
            st.session_state.last_error = f"API 服務返回錯誤狀態碼: {response.status_code}"
            return False
    except requests.exceptions.RequestException as e:
        st.session_state.last_error = f"無法連接到 API 服務: {str(e)}"
        return False

def retry_with_backoff(func, max_retries=3, initial_delay=1):
    """使用指數退避重試函數"""
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            time.sleep(delay)
            delay *= 2

def get_answer(question: str, include_sources: bool = True, max_sources: Optional[int] = None, use_query_rewrite: bool = True, show_relevance: bool = True, selected_model: Optional[str] = None, language: str = "繁體中文", use_dynamic_rag: bool = False, dynamic_ollama_model: Optional[str] = None, dynamic_embedding_model: Optional[str] = None, platform: Optional[str] = None, folder_path: Optional[str] = None) -> Dict[str, Any]:
    """獲取問題答案"""
    try:
        payload = {
            "question": question,
            "include_sources": include_sources,
            "max_sources": max_sources,
            "use_query_rewrite": use_query_rewrite,
            "show_relevance": show_relevance,
            "language": language,  # 將語言作為獨立參數傳遞
            "use_dynamic_rag": use_dynamic_rag,
            "ollama_embedding_model": dynamic_embedding_model,
            "folder_path": folder_path
        }
        
        if platform:
            payload["platform"] = platform
        
        if use_dynamic_rag and dynamic_ollama_model:
            payload["ollama_model"] = dynamic_ollama_model  # 修復：使用正確的參數名
        elif selected_model:
            payload["selected_model"] = selected_model
        
        response = requests.post(
            f"{API_URL}/ask",
            json=payload
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.session_state.last_error = f"獲取答案時發生錯誤: {str(e)}"
        raise

def fetch_dynamic_scope_info(language: str, ollama_model: str, embedding_model: str, platform: Optional[str], folder_path: Optional[str]):
    """呼叫後端 scope-info 端點，取得動態RAG範圍資訊"""
    try:
        payload = {
            "language": language,
            "ollama_model": ollama_model,
            "ollama_embedding_model": embedding_model,
            "platform": platform,
            "folder_path": folder_path
        }
        resp = requests.post(f"{API_URL}/api/dynamic/scope-info", json=payload, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            st.session_state.dynamic_scope_info = data.get('scope_info')
            st.session_state.dynamic_block_recommended = data.get('block_recommended')
            st.session_state.dynamic_block_reason = data.get('blocking_reason')
        else:
            st.session_state.dynamic_scope_info = None
            st.session_state.dynamic_block_recommended = False
            st.session_state.dynamic_block_reason = None
    except Exception as e:
        st.session_state.dynamic_scope_info = None
        st.session_state.dynamic_block_recommended = False
        st.session_state.dynamic_block_reason = None
        logger.warning(f"取得動態範圍資訊失敗: {e}")

def get_indexed_files() -> List[Dict[str, Any]]:
    """獲取已索引的文件列表"""
    try:
        response = requests.get(FILES_ENDPOINT)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.session_state.last_error = f"獲取文件列表時發生錯誤: {str(e)}"
        logger.error(f"獲取文件列表時發生錯誤: {str(e)}")
        return []

# 更新聊天歷史
def update_chat_history(question, answer, sources=None, rewritten_question=None, file_count_warning=None):
    """更新聊天歷史"""
    if len(st.session_state.chat_history) >= 10:  # 限制歷史記錄數量
        st.session_state.chat_history.pop(0)
    
    st.session_state.chat_history.append({
        "question": question,
        "answer": answer,
        "sources": sources,
        "rewritten_question": rewritten_question,
        "file_count_warning": file_count_warning,
        "timestamp": datetime.now(pytz.timezone('Asia/Taipei')).strftime("%Y-%m-%d %H:%M:%S")
    })

# 處理範例問題點擊
def handle_example_click(example_question):
    """處理範例問題點擊"""
    st.session_state.current_question = example_question
    st.session_state.run_search = True

# 處理文本輸入變更
def handle_text_input_change():
    """處理文本輸入框變更（Enter鍵提交）"""
    if "question_input" in st.session_state:
        # 當用戶按下Enter鍵時，檢查輸入框當前值
        current_input = st.session_state.question_input
        
        # 僅當輸入不為空且發生變化時才設置搜索標誌
        if current_input:
            st.session_state.current_question = current_input
            st.session_state.run_search = True

# 主應用
def main():
    """主應用函數"""
    
    # 檢查API狀態
    api_status = check_api_status()
    
    if not api_status:
        st.error(f"無法連接到API服務，請確保API服務正在運行: {API_URL}")
        st.info("提示: 您可以通過運行 `python app.py` 啟動API服務")
        return
    
    # Web 應用直接可用，無需複雜設置
    
    # 標題
    st.markdown('<p class="main-header">Q槽文件智能助手</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">讓您的文檔知識觸手可及</p>', unsafe_allow_html=True)
    
    # 首先在 sidebar 中設置基本結構
    with st.sidebar:
        # 首先顯示幫助中心和系統狀態
        render_help_sidebar()
        
        st.markdown("---")
        
        # 顯示系統狀態
        st.markdown("### 📊 系統狀態")
        status = st.session_state.api_status
        if status:
            st.success(f"✅ API 服務: {status.get('status', '未知')}")
            st.info(f"🗄️ Q槽訪問: {'✅ 可訪問' if status.get('q_drive_accessible') else '❌ 不可訪問'}")
            st.info(f"🔖 API 版本: {status.get('version', '未知')}")
            # 顯示 runtime_state 基本索引資訊
            rt = status.get('runtime_state') or {}
            if rt:
                with st.expander("📦 索引狀態", expanded=False):
                    import datetime
                    def _fmt(ts):
                        if not ts:
                            return '-'
                        try:
                            return datetime.datetime.fromtimestamp(int(ts), tz=pytz.timezone('Asia/Taipei')).strftime('%Y-%m-%d %H:%M:%S')
                        except Exception:
                            return str(ts)
                    st.write(f"最後全量索引時間: {_fmt(rt.get('last_index_full_ts'))}")
                    st.write(f"最後增量索引時間: {_fmt(rt.get('last_index_incremental_ts'))}")
                    st.write(f"最後索引文檔數: {rt.get('last_index_doc_count') or '-'}")
                    if rt.get('language_model') or rt.get('embedding_model'):
                        st.write(f"模型: {rt.get('language_model') or '-'} | 嵌入: {rt.get('embedding_model') or '-'}")
            # 安全警報快速視圖
            with st.expander("🔐 安全警報 (最近)", expanded=False):
                try:
                    import requests as _r
                    alerts_resp = _r.get(f"{API_URL}/api/dynamic/security-alerts", params={"limit": 10}, timeout=4)
                    if alerts_resp.status_code == 200:
                        data = alerts_resp.json()
                        alerts = data.get('alerts', [])
                        if alerts:
                            for a in alerts:
                                ts = a.get('ts') or a.get('time') or ''
                                level = a.get('level','?')
                                recent_10m = a.get('recent_10m')
                                st.write(f"[{level}] ts={ts} 10m={recent_10m} reason={a.get('last_event_reason')}")
                        else:
                            st.caption("無最近警報")
                    else:
                        st.caption("無法取得警報")
                except Exception as _e:
                    st.caption(f"讀取警報失敗: {_e}")
            # 安全事件趨勢
            with st.expander("📈 安全事件趨勢", expanded=False):
                try:
                    import requests as _r
                    metrics_resp = _r.get(f"{API_URL}/api/dynamic/security-metrics", params={"window_minutes": 180}, timeout=5)
                    if metrics_resp.status_code == 200:
                        mdata = metrics_resp.json()
                        series = mdata.get('series', [])
                        if series:
                            import pandas as pd, datetime as dt
                            df = pd.DataFrame(series)
                            df['time'] = df['minute_ts'].apply(lambda t: dt.datetime.fromtimestamp(t))
                            st.line_chart(df.set_index('time')['count'])
                        st.caption(f"最近10分鐘事件: {mdata.get('recent_10m')} 等級: {mdata.get('level') or 'normal'} 窗口分鐘: {mdata.get('window_minutes')}")
                    else:
                        st.caption("無法取得安全指標")
                except Exception as e:
                    st.caption(f"趨勢讀取失敗: {e}")
        
        st.markdown("---")
        
        # RAG 模式選擇
        st.markdown("### 🔧 RAG 模式")
        rag_mode_main = st.radio(
            "選擇 RAG 模式：",
            options=["傳統RAG", "Dynamic RAG"],
            index=0,
            help="傳統RAG使用預建向量資料庫，Dynamic RAG即時檢索文件",
            key="main_rag_mode_selector"
        )
    
    # --- 分頁設計 ---
    if rag_mode_main == "傳統RAG":
        tab_names = ["💬 智能問答", "🛠️ 管理員後台", "🗄️ 向量資料庫維護"]
    else:
        # Dynamic RAG 模式下隱藏管理員功能
        tab_names = ["💬 智能問答"]
    tabs = st.tabs(tab_names)

    # --- sidebar 保留管理入口 ---
    if 'admin_tab' not in st.session_state:
        st.session_state.admin_tab = 0

    def goto_admin():
        st.session_state.admin_tab = 1
        st.rerun()

    with st.sidebar:
        st.markdown("---")
        
        # 簡化的線上模型選擇
        st.markdown("### 🤖 AI 模型選擇")
        
        if rag_mode_main == "傳統RAG":
            st.info("💡 傳統RAG 使用預建向量資料庫，響應更快")
            # 傳統RAG的模型選擇保持原有邏輯
        else:
            st.info("💡 Dynamic RAG 即時檢索文件，無需預建資料庫")
            
            # 簡化的平台選擇
            platform_choice = st.selectbox(
                "🏠 AI 平台:",
                options=["Hugging Face", "Ollama"],
                index=0,  # 默認 Hugging Face
                help="選擇 AI 推理平台",
                key="simple_platform_choice"
            )
            
            if platform_choice == "Hugging Face":
                # 預設輕量級模型組合
                # 從 API 獲取可用模型
                try:
                    models_response = requests.get(f"{API_URL}/api/setup/models", timeout=10)
                    if models_response.status_code == 200:
                        models_data = models_response.json()
                        if "error" not in models_data:
                            models = models_data.get("models", {})
                            language_models = models.get("language_models", [])
                            embedding_models = models.get("embedding_models", [])
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                if language_models:
                                    language_model_options = {}
                                    for model in language_models:
                                        display_name = f"{model['name']} ({model['size']})"
                                        language_model_options[model["id"]] = display_name
                                    
                                    selected_language_model = st.selectbox(
                                        "🧠 語言模型:",
                                        options=list(language_model_options.keys()),
                                        format_func=lambda x: language_model_options[x],
                                        help="用於生成回答的模型",
                                        key="simple_language_model"
                                    )
                                else:
                                    st.error("沒有找到本地語言模型")
                                    st.info("請先下載模型：")
                                    st.code("hf download Qwen/Qwen2-0.5B-Instruct --cache-dir ./models/cache")
                                    selected_language_model = None
                            
                            with col2:
                                if embedding_models:
                                    embedding_model_options = {}
                                    for model in embedding_models:
                                        display_name = f"{model['name']} ({model['size']})"
                                        embedding_model_options[model["id"]] = display_name
                                    
                                    selected_embedding_model = st.selectbox(
                                        "🔤 嵌入模型:",
                                        options=list(embedding_model_options.keys()),
                                        format_func=lambda x: embedding_model_options[x],
                                        help="用於文本向量化的模型",
                                        key="simple_embedding_model"
                                    )
                                else:
                                    st.error("沒有找到本地嵌入模型")
                                    st.info("請先下載模型：")
                                    st.code("hf download sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 --cache-dir ./models/cache")
                                    selected_embedding_model = None
                        else:
                            st.error(f"獲取模型列表失敗: {models_data['error']}")
                            selected_language_model = None
                            selected_embedding_model = None
                    else:
                        st.error("無法連接到 API 服務")
                        selected_language_model = None
                        selected_embedding_model = None
                
                except Exception as e:
                    st.error(f"獲取模型列表時出錯: {str(e)}")
                    selected_language_model = None
                    selected_embedding_model = None
                
                # 推理引擎選擇
                st.markdown("#### ⚙️ 推理引擎")
                inference_engine_options = {
                    "transformers": "🔧 Transformers (穩定，兼容性好)",
                    "vllm": "⚡ vLLM (高性能，需要更多 GPU 記憶體)"
                }
                
                selected_inference_engine = st.selectbox(
                    "選擇推理引擎:",
                    options=list(inference_engine_options.keys()),
                    format_func=lambda x: inference_engine_options[x],
                    help="選擇推理引擎，vLLM 更快但需要更多 GPU 記憶體",
                    key="simple_inference_engine"
                )
                
                # 模型選擇完成
                if selected_language_model and selected_embedding_model:
                    st.success("✅ 模型選擇完成，可以開始使用")
            
            else:  # Ollama
                st.info("🏠 使用本地 Ollama 服務")
                
                # 從 API 獲取 Ollama 模型
                try:
                    ollama_models_response = requests.get(f"{API_URL}/api/ollama/models/categorized", timeout=5)
                    if ollama_models_response.status_code == 200:
                        ollama_models = ollama_models_response.json()
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # 語言模型選擇
                            if ollama_models.get('language_models') and len(ollama_models['language_models']) > 0:
                                selected_language_model = st.selectbox(
                                    "🧠 語言模型:",
                                    options=ollama_models['language_models'],
                                    help="用於回答生成的語言模型",
                                    key="ollama_language_model"
                                )
                            else:
                                st.warning("⚠️ 沒有找到可用的語言模型")
                                st.info("💡 請確保 Ollama 服務正在運行並已下載模型")
                                # 提供常見模型選項
                                common_models = ["llama3.2:3b", "llama3.1:8b", "qwen2:7b", "gemma2:9b"]
                                selected_language_model = st.selectbox(
                                    "選擇常見模型:",
                                    options=common_models,
                                    help="這些是常見的 Ollama 模型，需要先下載",
                                    key="ollama_common_language_model"
                                )
                        
                        with col2:
                            # 嵌入模型選擇
                            if ollama_models.get('embedding_models') and len(ollama_models['embedding_models']) > 0:
                                selected_embedding_model = st.selectbox(
                                    "🔤 嵌入模型:",
                                    options=ollama_models['embedding_models'],
                                    help="用於文本向量化的嵌入模型",
                                    key="ollama_embedding_model"
                                )
                            else:
                                st.warning("⚠️ 沒有找到可用的嵌入模型")
                                st.info("💡 請確保 Ollama 服務正在運行並已下載嵌入模型")
                                # 提供常見嵌入模型選項
                                common_embed_models = ["nomic-embed-text", "mxbai-embed-large"]
                                selected_embedding_model = st.selectbox(
                                    "選擇常見嵌入模型:",
                                    options=common_embed_models,
                                    help="這些是常見的 Ollama 嵌入模型，需要先下載",
                                    key="ollama_common_embedding_model"
                                )
                        
                        # 顯示 Ollama 連接狀態
                        if not ollama_models.get('language_models') and not ollama_models.get('embedding_models'):
                            st.error("🔌 無法連接到 Ollama 服務或沒有可用模型")
                            st.markdown("""
                            **解決方案:**
                            1. 確保 Ollama 服務正在運行
                            2. 下載所需模型: `ollama pull llama3.2:3b`
                            3. 下載嵌入模型: `ollama pull nomic-embed-text`
                            """)
                    else:
                        st.error(f"❌ 無法獲取 Ollama 模型列表，狀態碼: {ollama_models_response.status_code}")
                        # 使用默認值
                        selected_language_model = "llama3.2:3b"
                        selected_embedding_model = "nomic-embed-text"
                        st.info("使用默認模型配置")
                except Exception as e:
                    st.error(f"❌ 獲取 Ollama 模型時出錯: {str(e)}")
                    # 使用默認值
                    selected_language_model = "llama3.2:3b"
                    selected_embedding_model = "nomic-embed-text"
                    st.info("使用默認模型配置")
            
            # 保存到 session state
            st.session_state.dynamic_platform = platform_choice.lower().replace(" ", "")
            st.session_state.dynamic_language_model = selected_language_model
            st.session_state.dynamic_embedding_model = selected_embedding_model
            if platform_choice == "Hugging Face":
                st.session_state.dynamic_inference_engine = selected_inference_engine
            else:
                st.session_state.dynamic_inference_engine = "ollama"
        


    # --- 問答主頁 ---
    with tabs[0]:
        if st.session_state.get('admin_tab', 0) == 1:
            st.session_state.admin_tab = 0  # 自動切回主頁時重置
        
        # 模型選擇變數初始化
        selected_model_folder = None
        
        # 創建側邊欄
        with st.sidebar:
            st.markdown("### 💡 關於")
            st.write("Q槽文件智能助手，輸入問題即可開始對話。")
            
            st.markdown("---")

            # 根據 RAG 模式顯示相應的設置
            if rag_mode_main == "傳統RAG":
                # 傳統RAG模型選擇
                st.markdown("### 🤖 向量模型")
                try:
                    usable_models_response = requests.get(f"{API_URL}/api/usable-models", timeout=5)
                    if usable_models_response.status_code == 200:
                        usable_models = usable_models_response.json()
                        if usable_models:
                            # 簡化顯示
                            model_options = [model['display_name'] for model in usable_models]
                            model_folder_map = {model['display_name']: model['folder_name'] for model in usable_models}
                            
                            selected_display_name = st.selectbox(
                                "選擇向量模型：",
                                options=model_options,
                                help="選擇預建的向量模型",
                                key="main_model_selector"
                            )
                            
                            selected_model_folder = model_folder_map.get(selected_display_name)
                            
                            # 簡化狀態顯示
                            current_model = next((m for m in usable_models if m['display_name'] == selected_display_name), None)
                            if current_model:
                                if current_model.get('is_training'):
                                    status_text = "🔄 訓練中"
                                elif current_model.get('has_data'):
                                    status_text = "✅ 可用"
                                else:
                                    status_text = "⚠️ 無數據"
                                st.info(f"狀態: {status_text}")
                        else:
                            st.warning("沒有可用的向量模型")
                            selected_model_folder = None
                    else:
                        st.warning("無法獲取模型列表")
                        selected_model_folder = None
                except Exception as e:
                    st.warning(f"獲取模型列表失敗: {str(e)}")
                    selected_model_folder = None
            
            st.markdown("---")
            
            # 語言選擇
            st.markdown("### 🌐 語言設置")
            language_options = ["繁體中文", "简体中文", "English", "ไทย"]
            
            selected_language = st.selectbox(
                "🌐 回答語言：",
                options=language_options,
                index=language_options.index(st.session_state.selected_language) if st.session_state.selected_language in language_options else 0,
                help="選擇AI回答時使用的語言",
                key="main_language_selector"
            )
            st.session_state.selected_language = selected_language
            
            # 文件夾選擇（僅在動態RAG模式下顯示）
            selected_folder_path = None
            if rag_mode_main == "Dynamic RAG":
                st.markdown("---")
                st.markdown("### 📁 搜索範圍")
                
                # 文件夾選擇器
                folder_enabled = st.checkbox("限制搜索範圍", value=False, help="限制在特定文件夾內搜索", key="folder_enabled")
                
                if folder_enabled:
                    # 使用文件夾瀏覽器組件
                    folder_browser = FolderBrowser(API_URL)
                    selected_folder_path = folder_browser.render()
                    # 若本次沒有新輸出（返回 None）則沿用既有 session 選擇
                    if selected_folder_path is None:
                        selected_folder_path = st.session_state.get('selected_folder_path')
                    else:
                        # 寫回 session 以便後續估算與問答使用
                        st.session_state.selected_folder_path = selected_folder_path

                    # 顯示當前選擇和驗證結果
                    if selected_folder_path is not None:
                        display_path = selected_folder_path if selected_folder_path else "根目錄"
                        st.success(f"🎯 當前選擇的搜索範圍：{display_path}")
                    elif st.session_state.get('selected_folder_path'):
                        # 防禦式：render 未回傳但 session 中已有
                        sp = st.session_state.get('selected_folder_path') or ''
                        st.success(f"🎯 當前選擇的搜索範圍：{sp if sp else '根目錄'} (已保持)")

                        # 調用API進行文件夾驗證
                        try:
                            validate_response = requests.get(
                                f"{API_URL}/api/validate-folder",
                                params={"folder_path": selected_folder_path},
                                timeout=10  # 設置超時
                            )
                            if validate_response.status_code == 200:
                                validation_result = validate_response.json()
                                if validation_result.get("exists"):
                                    count_type = validation_result.get('count_type', '估算')
                                    file_count = validation_result.get('file_count', 0)
                                    message = f"文件夾有效，{count_type}約 {file_count} 個文件。"
                                    st.info(message)

                                    warning_level = validation_result.get("warning_level")
                                    if warning_level == "high":
                                        st.error(f"⚠️ 警告: {validation_result.get('suggestion')}")
                                    elif warning_level == "medium":
                                        st.warning(f"💡 提示: {validation_result.get('suggestion')}")

                                else:
                                    st.error("選擇的文件夾不存在或無法訪問。")
                            else:
                                st.error("無法驗證文件夾，請稍後再試。")
                        except requests.exceptions.RequestException as e:
                            st.error(f"驗證文件夾時出錯: {e}")

                        # 清除選擇按鈕
                        if st.button("🗑️ 清除選擇", key="clear_folder_selection"):
                            folder_browser.clear_selection()
                            selected_folder_path = None
                            st.session_state.selected_folder_path = None
                            st.rerun()
                else:
                    # 取消勾選時清空已設定範圍
                    if st.session_state.get('selected_folder_path') is not None:
                        st.session_state.selected_folder_path = None
                # 在動態模式下，進行背景估算文件數量和狀態
                dyn_lang_model = st.session_state.get('dynamic_language_model')
                dyn_embed_model = st.session_state.get('dynamic_embedding_model')
                dyn_platform = st.session_state.get('dynamic_platform')
                
                # 背景估算（重構：明確狀態機）
                if rag_mode_main == "Dynamic RAG" and dyn_lang_model and dyn_embed_model:
                    cache_key_components = [dyn_lang_model, dyn_embed_model, dyn_platform, (selected_folder_path if folder_enabled else None) or '__root__']
                    new_cache_key = '|'.join(str(c) for c in cache_key_components)
                    if 'dynamic_estimation_status' not in st.session_state:
                        st.session_state.dynamic_estimation_status = 'uninitialized'
                    status = st.session_state.dynamic_estimation_status
                    estimation_id = st.session_state.get('dynamic_estimation_id')

                    # 估算按鈕靠左
                    if st.button("📊 估算檔案", key="trigger_estimation"):
                            # 取消舊任務（若存在且未完成）
                            old_id = st.session_state.get('dynamic_estimation_id')
                            if old_id and st.session_state.get('dynamic_estimation_status') in ['running','triggered']:
                                try:
                                    requests.delete(f"{API_URL}/api/dynamic/background-estimate/{old_id}", timeout=3)
                                except Exception:
                                    pass
                            # 重置狀態
                            st.session_state.dynamic_estimated_count = 0
                            st.session_state.dynamic_warning_level = 'none'
                            st.session_state.dynamic_warning_message = None
                            st.session_state.dynamic_partial_estimate = None
                            st.session_state.dynamic_estimation_progress = 0
                            st.session_state.dynamic_estimation_status = 'triggered'
                            st.session_state.dynamic_cache_key = new_cache_key
                            st.session_state.last_estimation_check_time = 0
                            st.rerun()

                    # 狀態：triggered -> 啟動任務
                    if status == 'triggered' and st.session_state.get('dynamic_cache_key') == new_cache_key:
                        try:
                            resp = requests.post(
                                f"{API_URL}/api/dynamic/background-estimate",
                                json={"folder_path": selected_folder_path, "session_id": st.session_state.frontend_session_id},
                                timeout=8
                            )
                            if resp.status_code == 200:
                                data = resp.json()
                                st.session_state.dynamic_estimation_id = data.get('estimation_id')
                                st.session_state.dynamic_estimation_status = 'running'
                                st.session_state.last_estimation_check_time = 0
                            else:
                                st.session_state.dynamic_estimation_status = 'error'
                                st.session_state.dynamic_warning_message = '無法啟動估算任務'
                        except Exception as e:
                            st.session_state.dynamic_estimation_status = 'error'
                            st.session_state.dynamic_warning_message = f"估算啟動失敗: {e}"
                        st.rerun()

                    # 狀態：running -> 輪詢進度
                    if st.session_state.dynamic_estimation_status == 'running' and estimation_id:
                        last_check = st.session_state.get('last_estimation_check_time', 0)
                        now_ts = time.time()
                        # 預設 2 秒節流，自動刷新
                        if now_ts - last_check >= 2:
                            try:
                                status_resp = requests.get(f"{API_URL}/api/dynamic/background-estimate/{estimation_id}", timeout=6)
                                st.session_state.last_estimation_check_time = now_ts
                                if status_resp.status_code == 200:
                                    sd = status_resp.json()
                                    t_status = sd.get('status')
                                    if t_status == 'running':
                                        st.session_state.dynamic_estimation_progress = sd.get('progress', 0)
                                        pe = sd.get('partial_estimate')
                                        if pe is not None:
                                            st.session_state.dynamic_partial_estimate = pe
                                    elif t_status == 'completed':
                                        result = sd.get('result', {})
                                        st.session_state.dynamic_estimated_count = result.get('estimated_file_count', 0)
                                        st.session_state.dynamic_warning_level = result.get('warning_level', 'none')
                                        st.session_state.dynamic_warning_message = result.get('warning_message')
                                        st.session_state.dynamic_should_block = result.get('should_block', False)
                                        st.session_state.dynamic_confidence = result.get('confidence', 'unknown')
                                        st.session_state.dynamic_estimation_method = result.get('method', 'unknown')
                                        st.session_state.dynamic_estimation_details = result.get('estimation_details', {})
                                        st.session_state.dynamic_estimation_status = 'completed'
                                    elif t_status == 'error':
                                        st.session_state.dynamic_estimation_status = 'error'
                                        st.session_state.dynamic_warning_level = 'error'
                                        st.session_state.dynamic_warning_message = sd.get('error','估算失敗')
                                else:
                                    st.session_state.dynamic_warning_level = 'error'
                                    st.session_state.dynamic_warning_message = '狀態查詢失敗'
                            except Exception as e:
                                logger.error(f"估算輪詢失敗: {e}")

                    # UI 顯示與 gating
                    status = st.session_state.dynamic_estimation_status
                    estimated_count = st.session_state.get('dynamic_estimated_count', 0)
                    warning_level = st.session_state.get('dynamic_warning_level', 'none')
                    warning_message = st.session_state.get('dynamic_warning_message')
                    confidence = st.session_state.get('dynamic_confidence', 'unknown')
                    should_block = st.session_state.get('dynamic_should_block', False)

                    if status == 'running':
                        prog = st.session_state.get('dynamic_estimation_progress', 0)
                        partial = st.session_state.get('dynamic_partial_estimate')
                        line = f"📊 正在估算... {prog}%"
                        line += f" | 暫估≈{partial:,}" if partial is not None else " | 暫估準備中…"
                        st.info(line)
                        st.progress(min(prog,99)/100)
                        # 自動刷新：新版 Streamlit 使用 st.rerun；避免立即無限重跑，可在前面設定 st_autorefresh 或此處條件跳出
                        st.rerun()
                        should_block = True
                    elif status == 'completed':
                        if estimated_count > 0:
                            folder_status = "已限制" if selected_folder_path else "全範圍"
                            confidence_indicator = {"high": "🟢", "medium": "🟡", "low": "🟠", "unknown": "⚪"}.get(confidence, "⚪")
                            est_method = st.session_state.get('dynamic_estimation_method')
                            method_tag = " (early-stop)" if est_method == 'early-stop' else ""
                            if warning_level == 'critical' or should_block:
                                st.error(f"📦 估算文件數: **{estimated_count:,}**{method_tag} | 範圍: {folder_status} | {confidence_indicator} 信心度: {confidence}")
                            elif warning_level == 'high':
                                st.warning(f"📦 估算文件數: **{estimated_count:,}**{method_tag} | 範圍: {folder_status} | {confidence_indicator} 信心度: {confidence}")
                            else:
                                st.success(f"📦 估算文件數: **{estimated_count:,}**{method_tag} | 範圍: {folder_status} | {confidence_indicator} 信心度: {confidence}")
                        else:
                            st.warning('估算完成，但結果為 0。')
                    elif status == 'error':
                        st.error(f"⚠️ 估算失敗：{warning_message or '未知錯誤'}")
                        should_block = True
                    else:  # uninitialized / idle
                        st.info("🛈 尚未估算，請按『📊 估算檔案』開始。估算完成後才能開始對話。")
                        should_block = True

                    # 更新 gating (只有完成且 should_block=False 才放行)
                    st.session_state.dynamic_should_block = should_block or status != 'completed'
                    
                    # 取消按鈕（僅 running 狀態）
                    if status == 'running' and estimation_id:
                        if st.button('❌ 取消', key='cancel_estimation'):
                                try:
                                    requests.delete(f"{API_URL}/api/dynamic/background-estimate/{estimation_id}", timeout=4)
                                except Exception:
                                    pass
                                st.session_state.dynamic_estimation_status = 'uninitialized'
                                st.session_state.dynamic_estimation_id = None
                                st.session_state.dynamic_partial_estimate = None
                                st.rerun()

                    # 詳細統計（完成後）
                    if status == 'completed':
                        estimation_details = st.session_state.get('dynamic_estimation_details', {})
                        if estimation_details and estimated_count > 0:
                            with st.expander('🔍 估算詳細資訊', expanded=False):
                                cols = st.columns(3)
                                cols[0].metric('採樣目錄數', estimation_details.get('sampled_dirs', 0))
                                cols[1].metric('總目錄數', estimation_details.get('total_dirs', 0))
                                cols[2].metric('平均每目錄檔案', f"{estimation_details.get('mean_files_per_dir', 0):.1f}")
                                method = st.session_state.get('dynamic_estimation_method','unknown')
                                ciw = estimation_details.get('confidence_interval_width')
                                st.write(f"方法: `{method}` | CI寬度: {ciw}")
                                if confidence == 'low':
                                    st.info('💡 信心較低，可嘗試限制資料夾範圍以提升準確度')
                                elif confidence == 'medium':
                                    st.warning('⚠️ 信心中等，結果可能有偏差')
                                else:
                                    st.success('✅ 信心高')
                    
                    # 顯示警告訊息
                    if warning_message and warning_level != 'error':
                        if should_block:
                            st.error(f"⛔ {warning_message}")
                            st.info("**解決方案：**\n1. 勾選「限制搜索範圍」\n2. 選擇更具體的資料夾\n3. 使用更精確的搜索關鍵詞")
                        elif warning_level == 'high':
                            st.warning(f"⚠️ {warning_message}")
                        elif warning_level in ['medium', 'low']:
                            st.info(f"💡 {warning_message}")
                    elif warning_level == 'error' and warning_message:
                        st.error(f"⚠️ {warning_message}")
                        st.info("**建議：**\n1. 檢查網路連線\n2. 重新選擇資料夾\n3. 稍後再試")
            
            # 固定設置，不再提供用戶選項
            include_sources = True  # 總是包含相關文件
            max_sources = 5  # 固定回應5筆結果
            show_relevance = st.checkbox("顯示相關性理由", value=True, help="顯示為什麼這些文件與查詢相關", key="show_relevance_checkbox")
            use_query_rewrite = st.checkbox("使用查詢優化", value=True, help="自動改寫查詢以獲得更準確的結果", key="use_query_rewrite_checkbox")

            # 操作按鈕
            if st.button("🗑️ 清除歷史", key="clear_history"):
                st.session_state.chat_history = []
                st.session_state.current_answer = None
                st.rerun()

        # 主要聊天界面
        st.header("💬 智能問答聊天")
        
        # 聊天容器 - 顯示對話歷史
        chat_container = st.container()
        
        with chat_container:
            # 如果有聊天歷史，顯示所有對話
            if st.session_state.chat_history:
                for i, chat in enumerate(st.session_state.chat_history):
                    # 用戶問題氣泡
                    st.markdown(f"""
                    <div style="display: flex; justify-content: flex-end; margin: 10px 0;">
                        <div style="background-color: #007bff; color: white; padding: 10px 15px; border-radius: 18px; max-width: 70%; word-wrap: break-word;">
                            <strong>您:</strong> {chat['question']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # AI 回答氣泡
                    if chat.get("rewritten_question"):
                        st.markdown(f"""
                        <div style="display: flex; justify-content: center; margin: 10px 0;">
                            <div style="background-color: #e0e0e0; color: #555; padding: 5px 10px; border-radius: 10px; max-width: 70%; font-size: 0.9em;">
                                🔍 <strong>優化後查詢:</strong> {chat['rewritten_question']}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    st.markdown(f"""
                    <div style="display: flex; justify-content: flex-start; margin: 10px 0;">
                        <div style="background-color: #f1f3f4; color: #333; padding: 10px 15px; border-radius: 18px; max-width: 70%; word-wrap: break-word;">
                            <strong>🤖 AI助手:</strong><br>{chat['answer']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # 顯示文件數量警告（如果有）
                    if chat.get("file_count_warning"):
                        st.warning(f"⚠️ {chat['file_count_warning']}")
                    
                    # 顯示相關文件（如果有）
                    if "sources" in chat and chat["sources"]:
                        # 去重處理
                        unique_files = {}
                        for source in chat["sources"]:
                            file_path = source["file_path"]
                            if file_path not in unique_files:
                                unique_files[file_path] = source
                        
                        # 詳細信息展開器
                        with st.expander(f"查看第 {i+1} 次對話的詳細文件信息", expanded=False):
                            for idx, (_, source) in enumerate(unique_files.items(), 1):
                                st.markdown(f"**文件 {idx}: {source['file_name']}**")
                                display_path = source["file_path"].replace(Q_DRIVE_PATH, DISPLAY_DRIVE_NAME)
                                st.write(f"📁 路徑: {display_path}")
                                
                                if source.get("location_info"):
                                    st.write(f"📍 位置: {source['location_info']}")
                                
                                if source.get("score") is not None:
                                    st.write(f"📊 相關度: {source['score']:.4f}")
                                
                                if show_relevance and source.get("relevance_reason"):
                                    st.markdown("**🔍 相關性理由:**")
                                    st.info(source["relevance_reason"])
                                
                                st.markdown("---")
            
            # 如果沒有聊天歷史，顯示歡迎信息
            else:
                st.markdown("""
                <div style="text-align: center; padding: 40px 20px; color: #666;">
                    <h3>👋 歡迎使用 Q槽文件智能助手</h3>
                    <p>我可以幫助您快速查找和了解公司內部文檔中的信息</p>
                    <p>請在下方輸入您的問題開始對話</p>
                </div>
                """, unsafe_allow_html=True)
        
        # 輸入區域 - 固定在底部
        st.markdown("---")
        
        # 使用 st.chat_input 以獲得更好的聊天體驗
        input_disabled = False
        block_reason = ""
        
        # 檢查是否應該阻擋輸入
        if rag_mode_main == "Dynamic RAG":
            should_block = st.session_state.get('dynamic_should_block', False)
            warning_level = st.session_state.get('dynamic_warning_level', 'none')
            estimation_status = st.session_state.get('dynamic_estimation_status')
            if should_block:
                input_disabled = True
                if estimation_status != 'completed':
                    block_reason = "尚未完成估算，請先執行並等待『📊 估算檔案』完成"
                else:
                    # 已完成但仍需阻擋（例如檔案過多）
                    block_reason = st.session_state.get('dynamic_warning_message') or "檔案數量過多，請縮小搜索範圍"
            elif warning_level == "high":
                # 高風險但不完全阻擋，顯示警告但允許輸入
                pass

        placeholder_text = "請輸入您的問題，例如：ITPortal是什麼？"
        if input_disabled:
            placeholder_text = f"🚫 {block_reason}"

        if question := st.chat_input(placeholder_text, disabled=input_disabled):
            if input_disabled:
                st.warning(f"🚫 {block_reason}")
                st.stop()
            
            with st.spinner("🤖 AI助手正在思考..."):
                try:
                    # 直接調用問答API
                    result = get_answer(
                        question,
                        include_sources,
                        max_sources,
                        use_query_rewrite,
                        show_relevance,
                        selected_model_folder,
                        selected_language,
                        use_dynamic_rag=(rag_mode_main == "Dynamic RAG"),
                        dynamic_ollama_model=st.session_state.get('dynamic_language_model'),
                        dynamic_embedding_model=st.session_state.get('dynamic_embedding_model'),
                        platform=st.session_state.get('dynamic_platform') if rag_mode_main == "Dynamic RAG" else None,
                        folder_path=selected_folder_path
                    )

                    answer_text = result.get("answer", "無法獲取答案")
                    sources = result.get("sources", [])
                    rewritten_question = result.get("rewritten_query")
                    file_count_warning = result.get("file_count_warning")

                    # 如果有文件數量警告，顯示警告消息
                    if file_count_warning:
                        st.warning(f"⚠️ {file_count_warning}")

                    # 更新聊天歷史
                    update_chat_history(question, answer_text, sources, rewritten_question, file_count_warning)

                    # 重新運行以更新界面
                    st.rerun()

                except Exception as e:
                    error_msg = f"生成過程中發生錯誤，請稍後再試。"
                    
                    # 檢查是否是模型相關錯誤
                    error_str = str(e)
                    if "模型" in error_str or "model" in error_str.lower():
                        error_msg += "\n\n💡 這可能是因為模型尚未完全下載或初始化。如果您是首次使用，請等待模型下載完成後再試。"
                        error_msg += "\n\n建議：\n- 檢查網路連接\n- 選擇較小的模型進行測試\n- 查看系統狀態確認模型是否就緒"
                    
                    logger.error(f"處理問題時發生錯誤: {error_str}")
                    update_chat_history(question, error_msg, [], None, None)
                    st.rerun()
        
        # 頁腳
        st.markdown(
            '<div class="footer">© 2025 公司名稱 - Q槽文件智能助手 v1.0.0</div>',
            unsafe_allow_html=True
        )

    # --- 管理員後台分頁（僅在傳統RAG模式下顯示） ---
    if len(tabs) > 1:
        with tabs[1]:
            st.header("🛠️ 管理員後台")
            admin_token = st.text_input("請輸入管理員Token", type="password", key="admin_token_tab")
            
            if admin_token:
                # 模型訓練管理（僅 Ollama 平台）
                st.subheader("📚 模型訓練管理 (Ollama)")
                
                # 獲取 Ollama 模型列表
                try:
                    ollama_models_resp = requests.get(f"{API_URL}/api/ollama/models", timeout=10)
                    if ollama_models_resp.status_code == 200:
                        ollama_models = ollama_models_resp.json()
                        model_names = [model['name'] for model in ollama_models]
                        
                        # 檢查是否有保持的模型選擇狀態
                        preserved_selection = st.session_state.get('preserve_model_selection')
                        default_ollama_index = 0
                        default_embedding_index = 0
                        
                        if preserved_selection:
                            # 如果保持狀態未過期（5分鐘內），使用保存的選擇
                            if time.time() - preserved_selection.get('timestamp', 0) < 300:
                                try:
                                    if preserved_selection['ollama_model'] in model_names:
                                        default_ollama_index = model_names.index(preserved_selection['ollama_model'])
                                    if preserved_selection['embedding_model'] in model_names:
                                        default_embedding_index = model_names.index(preserved_selection['embedding_model'])
                                except (ValueError, KeyError):
                                    pass
                            else:
                                # 過期後清除保持狀態
                                del st.session_state.preserve_model_selection
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            selected_ollama_model = st.selectbox(
                                "選擇 Ollama 語言模型：",
                                options=model_names,
                                index=default_ollama_index,
                                help="用於問答的語言模型",
                                key="admin_ollama_model_selector"
                            )
                        with col2:
                            selected_embedding_model = st.selectbox(
                                "選擇 Ollama 嵌入模型：",
                                options=model_names,
                                index=default_embedding_index,
                                help="用於文本嵌入的模型",
                                key="admin_embedding_model_selector"
                            )
                        
                        # 版本選擇
                        st.markdown("### 版本管理")
                        version_options = ["✨ 建立新版本"]
                        existing_versions = []
                        try:
                            versions_resp = requests.get(
                                f"{API_URL}/api/model-versions",
                                params={
                                    "ollama_model": selected_ollama_model,
                                    "ollama_embedding_model": selected_embedding_model
                                },
                                timeout=5
                            )
                            if versions_resp.status_code == 200:
                                existing_versions = [v['version'] for v in versions_resp.json() if v.get('version')]
                                version_options.extend(sorted(existing_versions, reverse=True))
                        except Exception as e:
                            st.warning(f"無法獲取版本列表: {e}")

                        selected_version_option = st.selectbox(
                            "選擇訓練版本:",
                            options=version_options,
                            help="選擇一個現有版本進行增量訓練，或建立一個帶有今天日期的新版本。",
                            key="admin_version_selector"
                        )

                        # 確定最終要發送到API的版本號
                        final_version = None
                        if selected_version_option == "✨ 建立新版本":
                            from datetime import datetime
                            import pytz
                            final_version = datetime.now(pytz.timezone('Asia/Taipei')).strftime('%Y%m%d')
                            st.info(f"將建立新版本: **{final_version}**")
                        else:
                            final_version = selected_version_option

                        # 檢查向量數據是否存在
                        try:
                            vector_models_resp = requests.get(f"{API_URL}/api/vector-models", timeout=5)
                            if vector_models_resp.status_code == 200:
                                vector_models = vector_models_resp.json()
                                
                                # 查找匹配的模型
                                current_model_exists = False
                                current_model_has_data = False
                                current_model_training = False
                                
                                # 構建目標資料夾名稱以進行精確匹配
                                # 注意：此處的前端邏輯無法完美複製後端的 folder_name 生成，但可以模擬
                                target_folder_part = f"{selected_ollama_model.replace(':', '_')}@{selected_embedding_model.replace(':', '_')}"
                                if final_version:
                                    target_folder_part += f"#{final_version}"

                                for model in vector_models:
                                    if target_folder_part in model['folder_name']:
                                        current_model_exists = True
                                        current_model_has_data = model['has_data']
                                        current_model_training = model['is_training']
                                        break
                                
                                # 模型狀態檢查完成
                        except:
                            pass
                        
                        # 訓練按鈕
                        st.markdown("### 訓練操作")
                        btn_cols = st.columns(3)
                        
                        with btn_cols[0]:
                            if st.button("🚀 初始訓練", key="new_initial_training", 
                                       disabled=current_model_training or (current_model_exists and current_model_has_data)):
                                try:
                                    resp = requests.post(
                                        f"{API_URL}/admin/training/initial",
                                        headers={"admin_token": admin_token},
                                        json={
                                            "ollama_model": selected_ollama_model,
                                            "ollama_embedding_model": selected_embedding_model,
                                            "version": final_version
                                        }
                                    )
                                    if resp.status_code == 200:
                                        st.success(f"✅ 初始訓練已開始 (PID: {resp.json().get('pid')})")
                                        st.rerun()
                                    else:
                                        st.error(f"❌ 訓練失敗: {resp.text}")
                                except Exception as e:
                                    st.error(f"❌ API調用失敗: {e}")
                        
                        with btn_cols[1]:
                            if st.button("📈 增量訓練", key="new_incremental_training",
                                       disabled=current_model_training or not (current_model_exists and current_model_has_data)):
                                try:
                                    resp = requests.post(
                                        f"{API_URL}/admin/training/incremental",
                                        headers={"admin_token": admin_token},
                                        json={
                                            "ollama_model": selected_ollama_model,
                                            "ollama_embedding_model": selected_embedding_model,
                                            "version": final_version
                                        }
                                    )
                                    if resp.status_code == 200:
                                        st.success(f"✅ 增量訓練已開始 (PID: {resp.json().get('pid')})")
                                    else:
                                        st.error(f"❌ 訓練失敗: {resp.text}")
                                except Exception as e:
                                    st.error(f"❌ API調用失敗: {e}")
                        
                        with btn_cols[2]:
                            if st.button("🔄 重新索引", key="new_reindex_training",
                                       disabled=current_model_training or not (current_model_exists and current_model_has_data)):
                                try:
                                    resp = requests.post(
                                        f"{API_URL}/admin/training/reindex",
                                        headers={"admin_token": admin_token},
                                        json={
                                            "ollama_model": selected_ollama_model,
                                            "ollama_embedding_model": selected_embedding_model,
                                            "version": final_version
                                        }
                                    )
                                    if resp.status_code == 200:
                                        st.success(f"✅ 重新索引已開始 (PID: {resp.json().get('pid')})")
                                        # 保持模型選擇狀態，避免被清除
                                        st.session_state.preserve_model_selection = {
                                            "ollama_model": selected_ollama_model,
                                            "embedding_model": selected_embedding_model,
                                            "version": final_version,
                                            "timestamp": time.time()
                                        }
                                        # 延遲刷新，讓用戶看到成功消息
                                        time.sleep(1)
                                        st.rerun()
                                    else:
                                        st.error(f"❌ 重新索引失敗: {resp.text}")
                                except Exception as e:
                                    st.error(f"❌ API調用失敗: {e}")
                        
                    else:
                        st.error("無法獲取 Ollama 模型列表")
                except Exception as e:
                    st.error(f"獲取模型列表失敗: {e}")
            


                
                # 鎖定狀態管理
                st.markdown("---")
                st.subheader("🔒 鎖定狀態管理")
                
                try:
                    lock_status_resp = requests.get(f"{API_URL}/admin/lock-status", headers={"admin_token": admin_token}, timeout=10)
                    if lock_status_resp.status_code == 200:
                        lock_status_list = lock_status_resp.json()
                        
                        if lock_status_list:
                            for status in lock_status_list:
                                with st.expander(f"🔐 {status['model_name']}", expanded=False):
                                    col1, col2, col3 = st.columns(3)
                                    
                                    with col1:
                                        if status['is_locked']:
                                            if status['is_lock_valid']:
                                                st.warning("🔒 已鎖定 (有效)")
                                            else:
                                                st.error("🔒 已鎖定 (無效)")
                                        else:
                                            st.success("🔓 未鎖定")
                                    
                                    with col2:
                                        if status['has_data']:
                                            st.success("✅ 有數據")
                                        else:
                                            st.warning("⚠️ 無數據")
                                    
                                    with col3:
                                        if status['can_use']:
                                            st.success("🟢 可使用")
                                        else:
                                            st.error("🔴 不可使用")
                                    
                                    st.write(f"**狀態說明:** {status['lock_reason']}")
                                    
                                    if status['lock_info']:
                                        st.write("**鎖定詳情:**")
                                        lock_info = status['lock_info']
                                        if 'created_at' in lock_info:
                                            st.write(f"- 鎖定時間: {lock_info['created_at']}")
                                        if 'pid' in lock_info:
                                            st.write(f"- 進程ID: {lock_info['pid']}")
                                        if 'process_name' in lock_info:
                                            st.write(f"- 進程名稱: {lock_info['process_name']}")
                                    
                                    # 解鎖按鈕
                                    if status['is_locked']:
                                        unlock_reason = st.text_input(
                                            "解鎖原因:", 
                                            value="管理員手動解鎖", 
                                            key=f"unlock_reason_{status['folder_name']}"
                                        )
                                        
                                        if st.button(f"🔓 強制解鎖", key=f"unlock_{status['folder_name']}"):
                                            try:
                                                unlock_resp = requests.post(
                                                    f"{API_URL}/admin/force-unlock",
                                                    headers={"admin_token": admin_token},
                                                    json={
                                                        "folder_name": status['folder_name'],
                                                        "reason": unlock_reason
                                                    }
                                                )
                                                if unlock_resp.status_code == 200:
                                                    result = unlock_resp.json()
                                                    st.success(f"✅ {result['message']}")
                                                    st.rerun()
                                                else:
                                                    st.error(f"❌ 解鎖失敗: {unlock_resp.text}")
                                            except Exception as e:
                                                st.error(f"❌ 解鎖操作失敗: {e}")
                            
                            # 批量清理無效鎖定
                            st.markdown("### 批量操作")
                            if st.button("🧹 清理所有無效鎖定", key="cleanup_locks"):
                                try:
                                    cleanup_resp = requests.post(
                                        f"{API_URL}/admin/cleanup-invalid-locks",
                                        headers={"admin_token": admin_token}
                                    )
                                    if cleanup_resp.status_code == 200:
                                        result = cleanup_resp.json()
                                        st.success("✅ 清理完成")
                                        for model_name, message in result['results'].items():
                                            st.info(f"- {model_name}: {message}")
                                        st.rerun()
                                    else:
                                        st.error(f"❌ 清理失敗: {cleanup_resp.text}")
                                except Exception as e:
                                    st.error(f"❌ 清理操作失敗: {e}")
                        else:
                            st.info("沒有找到任何模型")
                    else:
                        st.error("無法獲取鎖定狀態")
                except Exception as e:
                    st.error(f"獲取鎖定狀態失敗: {e}")

                # log下載鈕
                st.markdown("---")
                with st.expander("📥 Log 下載 (根據上方選擇的模型)"):
                    try:
                        # 獲取所有日誌文件列表
                        log_list_resp = requests.get(f"{API_URL}/admin/logs", headers={"admin_token": admin_token}, timeout=10)
                        if log_list_resp.status_code == 200:
                            all_log_files = log_list_resp.json()
                            
                            # 根據當前選擇的模型進行篩選
                            # 清理模型名稱以匹配日誌文件名中的格式
                            clean_model = selected_ollama_model.replace(':', '_').replace('/', '_').replace('\\', '_')
                            clean_embedding = selected_embedding_model.replace(':', '_').replace('/', '_').replace('\\', '_')
                            
                            relevant_logs = [
                                log for log in all_log_files 
                                if clean_model in log and clean_embedding in log
                            ]

                            if relevant_logs:
                                selected_log = st.selectbox("選擇要下載的日誌文件:", options=relevant_logs, key="log_selector")
                                
                                if selected_log:
                                    # 準備下載按鈕
                                    log_content_resp = requests.get(
                                        f"{API_URL}/admin/download_log",
                                        params={"filename": selected_log},
                                        headers={"admin_token": admin_token},
                                        timeout=20
                                    )
                                    if log_content_resp.status_code == 200:
                                        st.download_button(
                                            label=f"下載 {selected_log}",
                                            data=log_content_resp.content,
                                            file_name=selected_log,
                                            mime="text/plain",
                                            key=f"download_{selected_log}"
                                        )
                                    else:
                                        st.error(f"無法獲取日誌 '{selected_log}' 的內容")
                            else:
                                st.info("找不到與當前所選模型相關的日誌文件。")
                        else:
                            st.error("無法獲取日誌文件列表。")
                    except Exception as e:
                        st.error(f"獲取日誌時發生錯誤: {e}")
                
                # --- 監控當前狀態 ---
                st.markdown("---")
                st.subheader("📈 監控當前狀態")
                st_autorefresh(interval=60000, key="monitor_all_autorefresh")
                
                # 獲取當前選擇模型的 folder_name
                model_folder_name = None
                try:
                    folder_name_resp = requests.get(
                        f"{API_URL}/api/internal/get-model-folder-name",
                        params={
                            "ollama_model": selected_ollama_model,
                            "ollama_embedding_model": selected_embedding_model,
                            "version": final_version
                        },
                        timeout=5
                    )
                    if folder_name_resp.status_code == 200:
                        model_folder_name = folder_name_resp.json().get("folder_name")
                except Exception as e:
                    st.warning(f"無法獲取模型文件夾名稱: {e}")

                try:
                    params = {}
                    if model_folder_name:
                        params["model_folder_name"] = model_folder_name

                    resp = requests.get(f"{API_URL}/admin/monitor_all", headers={"admin_token": admin_token}, params=params, timeout=10)
                    
                    if resp.status_code == 200:
                        data = resp.json()
                        status_text = data.get('status', '')
                        progress_text = data.get('progress', '')
                        realtime_text = data.get('realtime', '')
                        st.markdown("#### 狀態 Console")
                        st.code(status_text, language="bash")
                        st.markdown("#### 進度 Console")
                        st.code(progress_text, language="bash")
                        st.markdown("#### 實時監控 Console")
                        st.code(realtime_text, language="bash")
                    else:
                        st.error(f"監控API回應異常: {resp.status_code}")
                except Exception as e:
                    st.error(f"監控API錯誤: {e}")

                # 依賴狀態
                st.markdown("---")
                st.subheader("📦 依賴核心版本健康")
                if st.button("刷新依賴狀態", key="refresh_dep_status"):
                    st.session_state['_dep_status_reload'] = True
                need_dep = st.session_state.get('_dep_status_reload', True)
                if need_dep:
                    try:
                        dep_resp = requests.get(f"{API_URL}/admin/dependencies/status", headers={"admin_token": admin_token}, timeout=8)
                        if dep_resp.status_code == 200:
                            dep_data = dep_resp.json()
                            items = dep_data.get('items', [])
                            mismatch_cnt = dep_data.get('mismatch_count')
                            if mismatch_cnt:
                                st.warning(f"發現 {mismatch_cnt} 個未對齊 (mismatch/missing)")
                            for it in items:
                                icon = '✅' if it.get('status')=='aligned' else ('⚠️' if it.get('status')=='mismatch' else '❌')
                                st.write(f"{icon} {it.get('package')} py:{it.get('pyproject') or '-'} req:{it.get('requirements') or '-'} inst:{it.get('installed') or '-'}")
                            st.caption("對齊策略：requirements.txt 為真實鎖定來源。")
                            # 觸發 lock & export
                            if st.button("執行 lock + export (Poetry)", key="run_lock_export"):
                                with st.spinner("執行中..."):
                                    try:
                                        resp_run = requests.post(f"{API_URL}/admin/dependencies/lock-export", headers={"admin_token": admin_token}, timeout=600)
                                        if resp_run.status_code == 200:
                                            run_data = resp_run.json()
                                            st.success(f"完成: {run_data.get('message')} 用時 {run_data.get('elapsed_seconds')}s")
                                            diff_lines = run_data.get('changed_requirements') or []
                                            if diff_lines:
                                                with st.expander("requirements.txt 變更 (diff)", expanded=False):
                                                    st.code('\n'.join(diff_lines), language='diff')
                                            st.session_state['_dep_status_reload'] = True
                                        else:
                                            st.error(f"操作失敗: {resp_run.text}")
                                    except Exception as e:
                                        st.error(f"操作錯誤: {e}")
                            # 依賴審計區塊
                            st.markdown("### 🧾 依賴審計")
                            col_a, col_b = st.columns(2)
                            with col_a:
                                if st.button("執行審計紀錄", key="run_dep_audit"):
                                    try:
                                        audit_run = requests.post(f"{API_URL}/admin/dependencies/audit-run", headers={"admin_token": admin_token}, timeout=20)
                                        if audit_run.status_code == 200:
                                            st.success("審計已寫入")
                                        else:
                                            st.error("審計執行失敗")
                                    except Exception as e:
                                        st.error(f"審計錯誤: {e}")
                            with col_b:
                                audit_limit = st.number_input("顯示最近紀錄數", min_value=10, max_value=200, value=60, step=10, key="audit_limit")
                            try:
                                audit_log = requests.get(f"{API_URL}/admin/dependencies/audit-log", headers={"admin_token": admin_token}, params={"limit": audit_limit}, timeout=8)
                                if audit_log.status_code == 200:
                                    adata = audit_log.json()
                                    entries = adata.get('entries', [])
                                    if entries:
                                        import pandas as pd, datetime as dt
                                        rows = []
                                        for e in entries:
                                            ts = e.get('ts')
                                            mc = e.get('mismatch_count')
                                            rows.append({'time': dt.datetime.fromtimestamp(ts), 'mismatch': mc})
                                        df = pd.DataFrame(rows)
                                        st.line_chart(df.set_index('time')['mismatch'])
                                        st.caption(f"最近 {len(entries)} 次審計，最後時間: {entries[-1].get('ts')}")
                                    else:
                                        st.caption("無審計紀錄")
                                else:
                                    st.caption("無法取得審計紀錄")
                            except Exception as e:
                                st.caption(f"讀取審計紀錄失敗: {e}")
                        else:
                            st.warning("無法取得依賴狀態 (需要管理員權限)")
                    except Exception as e:
                        st.warning(f"依賴狀態讀取失敗: {e}")
            else:
                st.info("請輸入Token以查看管理功能。")

    # --- 向量資料庫維護分頁（僅在傳統RAG模式下顯示） ---
    if len(tabs) > 2:
        with tabs[2]:
            st.header("🗄️ 向量資料庫內容維護")
            admin_token_db = st.text_input("請輸入管理員Token", type="password", key="admin_token_db")
            
            if admin_token_db:
                # 模型選擇
                st.subheader("📋 選擇要維護的模型")
            
                try:
                    vector_models_resp = requests.get(f"{API_URL}/api/vector-models", timeout=10)
                    if vector_models_resp.status_code == 200:
                        vector_models = vector_models_resp.json()
                        
                        # 只顯示有數據且未在訓練的模型
                        available_models = [m for m in vector_models if m['has_data'] and not m['is_training']]
                        
                        if available_models:
                            model_options = {m['display_name']: m['folder_name'] for m in available_models}
                            selected_model_name = st.selectbox(
                                "選擇模型:",
                                options=list(model_options.keys()),
                                key="selected_model_for_content"
                            )
                            selected_model_folder = model_options[selected_model_name]
                            
                            st.markdown("---")
                            
                            # 內容管理選項
                            content_tabs = st.tabs(["📄 瀏覽文檔", "✏️ 編輯文檔", "➕ 新增文檔"])
                            
                            # 瀏覽文檔
                            with content_tabs[0]:
                                st.subheader("📄 瀏覽向量資料庫中的文檔")
                                
                                # 分頁控制
                                col1, col2, col3 = st.columns([1, 2, 1])
                                with col1:
                                    page = st.number_input("頁碼", min_value=1, value=1, key="doc_page")
                                with col2:
                                    page_size = st.selectbox("每頁顯示", [10, 20, 50], index=1, key="doc_page_size")
                                
                                # 獲取文檔列表
                                try:
                                    docs_resp = requests.get(
                                        f"{API_URL}/admin/vector-db/documents",
                                        headers={"admin_token": admin_token_db},
                                        params={
                                            "folder_name": selected_model_folder,
                                            "page": page,
                                            "page_size": page_size
                                        }
                                    )
                                    
                                    if docs_resp.status_code == 200:
                                        docs_data = docs_resp.json()
                                        documents = docs_data.get('documents', [])
                                        total = docs_data.get('total', 0)
                                        total_pages = docs_data.get('total_pages', 1)
                                        
                                        st.info(f"共找到 {total} 個文檔，第 {page}/{total_pages} 頁")
                                        
                                        for doc in documents:
                                            with st.expander(f"📄 {doc['file_name']} (chunk {doc['chunk_index']})", expanded=False):
                                                st.write(f"**文件路徑:** {doc['file_path']}")
                                                st.write(f"**文檔ID:** {doc['id']}")
                                                st.write("**內容預覽:**")
                                                st.text_area("內容預覽", value=doc['content'], height=100, disabled=True, key=f"preview_{doc['id']}", label_visibility="hidden")
                                                
                                                col1, col2 = st.columns(2)
                                                with col1:
                                                    if st.button("📝 編輯此文檔", key=f"edit_btn_{doc['id']}"):
                                                        st.session_state[f"editing_doc_{doc['id']}"] = True
                                                        st.session_state["edit_doc_id"] = doc['id']
                                                        st.rerun()
                                                
                                                with col2:
                                                    if st.button("🗑️ 刪除此文檔", key=f"del_btn_{doc['id']}", type="secondary"):
                                                        if st.session_state.get(f"confirm_del_{doc['id']}", False):
                                                            try:
                                                                del_resp = requests.delete(
                                                                    f"{API_URL}/admin/vector-db/document/{doc['id']}",
                                                                    headers={"admin_token": admin_token_db},
                                                                    params={"folder_name": selected_model_folder}
                                                                )
                                                                if del_resp.status_code == 200:
                                                                    st.success("✅ 文檔已刪除")
                                                                    st.session_state[f"confirm_del_{doc['id']}"] = False
                                                                    st.rerun()
                                                                else:
                                                                    st.error(f"刪除失敗: {del_resp.text}")
                                                            except Exception as e:
                                                                st.error(f"刪除失敗: {e}")
                                                        else:
                                                            st.warning("⚠️ 確定要刪除此文檔嗎？")
                                                            if st.button("確認刪除", key=f"confirm_del_btn_{doc['id']}", type="primary"):
                                                                st.session_state[f"confirm_del_{doc['id']}"] = True
                                                                st.rerun()
                                    else:
                                        st.error(f"獲取文檔列表失敗: {docs_resp.text}")
                                except Exception as e:
                                    st.error(f"獲取文檔列表失敗: {e}")
                            
                            # 編輯文檔
                            with content_tabs[1]:
                                st.subheader("✏️ 編輯文檔內容")
                                
                                # 檢查是否有要編輯的文檔
                                edit_doc_id = st.session_state.get("edit_doc_id")
                                if edit_doc_id:
                                    try:
                                        # 獲取文檔詳情
                                        doc_resp = requests.get(
                                            f"{API_URL}/admin/vector-db/document/{edit_doc_id}",
                                            headers={"admin_token": admin_token_db},
                                            params={"folder_name": selected_model_folder}
                                        )
                                        
                                        if doc_resp.status_code == 200:
                                            doc_data = doc_resp.json()
                                            
                                            st.write(f"**編輯文檔:** {doc_data['file_name']}")
                                            st.write(f"**文檔ID:** {doc_data['id']}")
                                            
                                            # 編輯內容
                                            new_content = st.text_area(
                                                "文檔內容:",
                                                value=doc_data['content'],
                                                height=300,
                                                key=f"edit_content_{edit_doc_id}"
                                            )
                                            
                                            col1, col2 = st.columns(2)
                                            with col1:
                                                if st.button("💾 保存修改", key="save_edit", type="primary"):
                                                    try:
                                                        update_resp = requests.put(
                                                            f"{API_URL}/admin/vector-db/document/{edit_doc_id}",
                                                            headers={"admin_token": admin_token_db},
                                                            params={"folder_name": selected_model_folder},
                                                            json={"content": new_content}
                                                        )
                                                        
                                                        if update_resp.status_code == 200:
                                                            st.success("✅ 文檔已更新")
                                                            st.session_state["edit_doc_id"] = None
                                                            st.rerun()
                                                        else:
                                                            st.error(f"更新失敗: {update_resp.text}")
                                                    except Exception as e:
                                                        st.error(f"更新失敗: {e}")
                                            
                                            with col2:
                                                if st.button("❌ 取消編輯", key="cancel_edit"):
                                                    st.session_state["edit_doc_id"] = None
                                                    st.rerun()
                                        else:
                                            st.error(f"獲取文檔詳情失敗: {doc_resp.text}")
                                            st.session_state["edit_doc_id"] = None
                                    except Exception as e:
                                        st.error(f"獲取文檔詳情失敗: {e}")
                                        st.session_state["edit_doc_id"] = None
                                else:
                                    st.info("請從「瀏覽文檔」頁面選擇要編輯的文檔")
                            # 新增文檔
                            with content_tabs[2]:
                                st.subheader("➕ 新增文檔到向量資料庫")
                                
                                with st.form("add_document_form"):
                                    file_name = st.text_input("文件名稱", placeholder="例如: 手動添加的文檔.txt", key="add_doc_filename")
                                    content = st.text_area("文檔內容", height=300, placeholder="請輸入要添加到向量資料庫的內容...", key="add_doc_content")
                                    
                                    # 可選的元數據
                                    st.write("**可選元數據:**")
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        file_path = st.text_input("文件路徑", placeholder="例如: /manual/custom_doc.txt", key="add_doc_filepath")
                                    with col2:
                                        chunk_index = st.number_input("塊索引", min_value=0, value=0, key="add_doc_chunk_index")
                                    
                                    submitted = st.form_submit_button("➕ 添加文檔", type="primary")
                                    
                                    if submitted:
                                        if not content.strip():
                                            st.error("請輸入文檔內容")
                                        else:
                                            try:
                                                metadata = {
                                                    "file_name": file_name or "手動添加的文檔",
                                                    "file_path": file_path or "manual_add",
                                                    "chunk_index": chunk_index
                                                }
                                                
                                                add_resp = requests.post(
                                                    f"{API_URL}/admin/vector-db/document",
                                                    headers={"admin_token": admin_token_db},
                                                    params={"folder_name": selected_model_folder},
                                                    json={
                                                        "content": content.strip(),
                                                        "metadata": metadata
                                                    }
                                                )
                                                
                                                if add_resp.status_code == 200:
                                                    st.success("✅ 文檔已成功添加到向量資料庫")
                                                    st.rerun()
                                                else:
                                                    st.error(f"添加失敗: {add_resp.text}")
                                            except Exception as e:
                                                st.error(f"添加失敗: {e}")
                        else:
                                    st.warning("沒有可用於內容維護的模型（需要有數據且未在訓練中）")
                                    
                                    # 模型狀態檢查完成
                    else:
                        st.error("無法獲取向量模型列表")
                except Exception as e:
                    st.error(f"獲取向量模型列表失敗: {e}")
            else:
                st.info("請輸入Token以查看向量資料庫維護功能。")
    
    # 顯示幫助模態框（如果需要）
    show_help_modal()


if __name__ == "__main__":
    import argparse
    
    # 解析命令行參數
    parser = argparse.ArgumentParser(description='啟動Streamlit前端界面')
    parser.add_argument('--port', type=int, default=STREAMLIT_PORT, 
                        help=f'Streamlit端口，默認為 {STREAMLIT_PORT}')
    args = parser.parse_args()
    
    print(f"啟動Streamlit前端界面，端口: {args.port}")

# Streamlit腳本執行時，直接調用main函數
main()
