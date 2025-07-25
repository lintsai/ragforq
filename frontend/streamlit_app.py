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

from config.config import APP_HOST, APP_PORT, STREAMLIT_PORT, API_BASE_URL, is_q_drive_accessible, Q_DRIVE_PATH, DISPLAY_DRIVE_NAME

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
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #424242;
        margin-top: 0;
    }
    .source-title {
        font-size: 1.2rem;
        font-weight: bold;
        color: #1E88E5;
        margin-top: 1rem;
    }
    .source-item {
        background-color: #f0f2f6;
        border-radius: 5px;
        padding: 10px;
        margin-bottom: 5px;
    }
    .footer {
        text-align: center;
        color: #9e9e9e;
        font-size: 0.8rem;
        margin-top: 3rem;
    }
    .stTextInput>div>div>input {
        font-size: 1.1rem;
    }
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

def get_answer(question: str, include_sources: bool = True, max_sources: Optional[int] = None, use_query_rewrite: bool = True, show_relevance: bool = True, selected_model: Optional[str] = None, language: str = "繁體中文") -> Dict[str, Any]:
    """獲取問題答案"""
    try:
        payload = {
            "question": question,
            "include_sources": include_sources,
            "max_sources": max_sources,
            "use_query_rewrite": use_query_rewrite,
            "show_relevance": show_relevance,
            "language": language  # 將語言作為獨立參數傳遞
        }
        
        if selected_model:
            payload["selected_model"] = selected_model
        
        response = requests.post(
            f"{API_BASE_URL}/ask",
            json=payload
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.session_state.last_error = f"獲取答案時發生錯誤: {str(e)}"
        raise

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
def update_chat_history(question, answer, sources=None, rewritten_question=None):
    """更新聊天歷史"""
    if len(st.session_state.chat_history) >= 10:  # 限制歷史記錄數量
        st.session_state.chat_history.pop(0)
    
    st.session_state.chat_history.append({
        "question": question,
        "answer": answer,
        "sources": sources,
        "rewritten_question": rewritten_question,
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
    
    # 標題
    st.markdown('<p class="main-header">Q槽文件智能助手</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">讓您的文檔知識觸手可及</p>', unsafe_allow_html=True)
    
    # 檢查API狀態
    api_status = check_api_status()
    
    if not api_status:
        st.error(f"無法連接到API服務，請確保API服務正在運行: {API_URL}")
        st.info("提示: 您可以通過運行 `python app.py` 啟動API服務")
        return
    
    # --- 分頁設計 ---
    tab_names = ["💬 智能問答", "🛠️ 管理員後台"]
    tabs = st.tabs(tab_names)

    # --- sidebar 保留管理入口 ---
    if 'admin_tab' not in st.session_state:
        st.session_state.admin_tab = 0

    def goto_admin():
        st.session_state.admin_tab = 1
        st.rerun()

    with st.sidebar:
        st.markdown("---")

    # --- 問答主頁 ---
    with tabs[0]:
        if st.session_state.get('admin_tab', 0) == 1:
            st.session_state.admin_tab = 0  # 自動切回主頁時重置
        
        # 模型選擇變數初始化
        selected_model_folder = None
        
        # 創建側邊欄
        with st.sidebar:
            st.markdown("### 關於")
            st.write("Q槽文件智能助手可以幫助您快速查找和了解公司內部文檔中的信息。")
            st.write("只需輸入您的問題，系統將自動搜索最相關的文檔並提供回答。")
            
            st.markdown("---")

            # 模型選擇 - 移到最上面
            st.markdown("### 🤖 模型設置")
            try:
                usable_models_response = requests.get(f"{API_URL}/api/usable-models", timeout=5)
                if usable_models_response.status_code == 200:
                    usable_models = usable_models_response.json()
                    if usable_models:
                        # 找到默認模型（第一個有數據且不在訓練中的模型）
                        default_model = None
                        for model in usable_models:
                            if model.get('has_data', False) and not model.get('is_training', False):
                                default_model = model['display_name']
                                break
                        
                        # 構建選項列表
                        if default_model:
                            model_options = [f"🌟 {default_model} (默認)"] + [model['display_name'] for model in usable_models if model['display_name'] != default_model]
                        else:
                            model_options = [model['display_name'] for model in usable_models]
                        
                        model_folder_map = {model['display_name']: model['folder_name'] for model in usable_models}
                        
                        selected_display_name = st.selectbox(
                            "選擇問答模型：",
                            options=model_options,
                            help="選擇用於問答的向量模型，帶🌟的是系統推薦的默認模型"
                        )
                        
                        # 獲取實際的文件夾名稱
                        if selected_display_name.startswith("🌟"):
                            # 移除星號和 "(默認)" 標記
                            actual_name = selected_display_name.replace("🌟 ", "").replace(" (默認)", "")
                            selected_model_folder = model_folder_map.get(actual_name)
                        else:
                            selected_model_folder = model_folder_map.get(selected_display_name)
                        
                        # 顯示當前選擇的模型狀態
                        current_model_info = None
                        for model in usable_models:
                            model_name = model['display_name']
                            if (selected_display_name.startswith("🌟") and model_name in selected_display_name) or model_name == selected_display_name:
                                current_model_info = model
                                break
                        
                        if current_model_info:
                            col1, col2 = st.columns(2)
                            with col1:
                                if current_model_info.get('has_data', False):
                                    st.success("✅ 有數據")
                                else:
                                    st.warning("⚠️ 無數據")
                            with col2:
                                if current_model_info.get('is_training', False):
                                    st.warning("⏳ 訓練中")
                                else:
                                    st.success("✅ 可用")
                    else:
                        st.warning("沒有可用的向量模型，將使用系統默認配置")
                        selected_model_folder = None
                else:
                    st.error("無法獲取可用模型列表，將使用系統默認配置")
                    selected_model_folder = None
            except Exception as e:
                st.error(f"獲取模型列表時出錯: {str(e)}，將使用系統默認配置")
                selected_model_folder = None
            
            st.markdown("---")
            
            # 顯示系統狀態
            st.markdown("### 系統狀態")
            status = st.session_state.api_status
            if status:
                st.success(f"API 服務狀態: {status.get('status', '未知')}")
                st.info(f"Q槽訪問狀態: {'可訪問' if status.get('q_drive_accessible') else '不可訪問'}")
                st.info(f"API 版本: {status.get('version', '未知')}")

            st.markdown("---")
            
            # 添加設置選項
            st.markdown("### 設置")
            
            # 語言選擇
            language_options = ["繁體中文", "简体中文", "English", "ไทย"]
            selected_language = st.selectbox(
                "🌐 回答語言：",
                options=language_options,
                index=language_options.index(st.session_state.selected_language),
                help="選擇AI回答時使用的語言"
            )
            st.session_state.selected_language = selected_language
            
            include_sources = st.checkbox("包含相關文件", value=True)
            max_sources = st.number_input("最大相關文件數", min_value=1, max_value=20, value=10)
            show_relevance = st.checkbox("顯示相關性理由", value=True, help="顯示為什麼這些文件與查詢相關")
            use_query_rewrite = st.checkbox("使用查詢優化", value=True, help="自動改寫查詢以獲得更準確的結果")

            # 添加清除歷史按鈕
            if st.button("清除歷史記錄", key="clear_history"):
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
        if question := st.chat_input("請輸入您的問題，例如：ITPortal是什麼？"):
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
                        selected_language
                    )

                    answer_text = result.get("answer", "無法獲取答案")
                    sources = result.get("sources", [])
                    rewritten_question = result.get("rewritten_query")

                    # 更新聊天歷史
                    update_chat_history(question, answer_text, sources, rewritten_question)

                    # 重新運行以更新界面
                    st.rerun()

                except Exception as e:
                    error_msg = f"處理問題時發生錯誤: {str(e)}"
                    update_chat_history(question, error_msg, [])
                    st.rerun()
        
        # 頁腳
        st.markdown(
            '<div class="footer">© 2025 公司名稱 - Q槽文件智能助手 v1.0.0</div>',
            unsafe_allow_html=True
        )

    # --- 管理員後台分頁 ---
    with tabs[1]:
        st.session_state.admin_tab = 1
        st.header("🛠️ 管理員後台")
        admin_token = st.text_input("管理員Token", type="password", key="admin_token_tab")
        if admin_token:
            st.success("已輸入Token，可操作管理功能")
            
            # 模型訓練管理
            st.subheader("📚 模型訓練管理")
            
            # 獲取 Ollama 模型列表
            try:
                ollama_models_resp = requests.get(f"{API_URL}/api/ollama/models", timeout=10)
                if ollama_models_resp.status_code == 200:
                    ollama_models = ollama_models_resp.json()
                    model_names = [model['name'] for model in ollama_models]
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        selected_ollama_model = st.selectbox(
                            "選擇 Ollama 語言模型：",
                            options=model_names,
                            help="用於問答的語言模型"
                        )
                    with col2:
                        selected_embedding_model = st.selectbox(
                            "選擇 Ollama 嵌入模型：",
                            options=model_names,
                            help="用於文本嵌入的模型"
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
                        help="選擇一個現有版本進行增量訓練，或建立一個帶有今天日期的新版本。"
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
                            
                            # 顯示狀態
                            st.markdown("### 當前選擇模型狀態")
                            if current_model_exists:
                                if current_model_training:
                                    st.warning("⏳ 該模型版本正在訓練中...")
                                elif current_model_has_data:
                                    st.success("✅ 該模型版本已有向量數據，可進行增量訓練或重新索引")
                                else:
                                    st.info("📝 該模型版本已創建但無數據，可進行初始訓練")
                            else:
                                st.info("🆕 該模型版本尚未創建，將創建新的向量資料夾進行初始訓練")
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
                                else:
                                    st.error(f"❌ 重新索引失敗: {resp.text}")
                            except Exception as e:
                                st.error(f"❌ API調用失敗: {e}")
                    
                else:
                    st.error("無法獲取 Ollama 模型列表")
            except Exception as e:
                st.error(f"獲取模型列表失敗: {e}")
            
            # 向量模型狀態
            st.markdown("---")
            st.subheader("📊 向量模型狀態")
            
            try:
                vector_models_resp = requests.get(f"{API_URL}/api/vector-models", timeout=10)
                if vector_models_resp.status_code == 200:
                    vector_models = vector_models_resp.json()
                    
                    if vector_models:
                        for model in vector_models:
                            with st.expander(f"📁 {model['display_name']}", expanded=False):
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    if model['has_data']:
                                        st.success("✅ 有數據")
                                    else:
                                        st.warning("⚠️ 無數據")
                                
                                with col2:
                                    if model['is_training']:
                                        st.warning("⏳ 訓練中")
                                    else:
                                        st.success("✅ 可用")
                                
                                with col3:
                                    if model['has_data'] and not model['is_training']:
                                        st.success("🟢 可問答")
                                    else:
                                        st.error("🔴 不可問答")
                                
                                if model['model_info']:
                                    st.write(f"**語言模型:** {model['model_info'].get('OLLAMA_MODEL', '未知')}")
                                    st.write(f"**嵌入模型:** {model['model_info'].get('OLLAMA_EMBEDDING_MODEL', '未知')}")
                                
                                st.write(f"**文件夾:** {model['folder_name']}")
                    else:
                        st.info("尚無向量模型")
                else:
                    st.error("無法獲取向量模型狀態")
            except Exception as e:
                st.error(f"獲取向量模型狀態失敗: {e}")
            

            
            # log下載鈕
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
                else:
                    status_text = progress_text = realtime_text = f"(監控API回應異常: {resp.status_code})"
            except Exception as e:
                status_text = progress_text = realtime_text = f"監控API錯誤: {e}"

            st.markdown("#### 狀態 Console")
            st.code(status_text, language="bash")
            st.markdown("#### 進度 Console")
            st.code(progress_text, language="bash")
            st.markdown("#### 實時監控 Console")
            st.code(realtime_text, language="bash")


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
