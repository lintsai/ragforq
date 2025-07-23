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

def get_answer(question: str, include_sources: bool = True, max_sources: Optional[int] = None, use_query_rewrite: bool = True, show_relevance: bool = True, selected_model: Optional[str] = None) -> Dict[str, Any]:
    """獲取問題答案"""
    try:
        payload = {
            "question": question,
            "include_sources": include_sources,
            "max_sources": max_sources,
            "use_query_rewrite": use_query_rewrite,
            "show_relevance": show_relevance
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
def update_chat_history(question, answer, sources=None):
    """更新聊天歷史"""
    if len(st.session_state.chat_history) >= 10:  # 限制歷史記錄數量
        st.session_state.chat_history.pop(0)
    
    st.session_state.chat_history.append({
        "question": question,
        "answer": answer,
        "sources": sources,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
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
        st._rerun()

    with st.sidebar:
        st.markdown("---")

    # --- 問答主頁 ---
    with tabs[0]:
        if st.session_state.get('admin_tab', 0) == 1:
            st.session_state.admin_tab = 0  # 自動切回主頁時重置
        # 創建側邊欄
        with st.sidebar:
            st.markdown("### 關於")
            st.write("Q槽文件智能助手可以幫助您快速查找和了解公司內部文檔中的信息。")
            st.write("只需輸入您的問題，系統將自動搜索最相關的文檔並提供回答。")
            
            st.markdown("### 使用說明")
            st.write("1. 在輸入框中輸入您的問題")
            st.write("2. 點擊'提問'按鈕或按回車鍵")
            st.write("3. 系統將搜索相關文檔並回答您的問題")
            st.write("4. 相關文件會顯示在回答下方")
            
            st.markdown("### 示例問題")
            example_questions = [
                "公司的年假政策是什麼？",
                "如何申請報銷差旅費？",
                "產品退貨流程是怎樣的？",
                "公司安全規定有哪些要點？"
            ]
            
            for q in example_questions:
                if st.button(q, key=f"example_{q}"):
                    handle_example_click(q)
            
            # 顯示系統狀態
            status = st.session_state.api_status
            if status:
                st.success(f"API 服務狀態: {status.get('status', '未知')}")
                st.info(f"Q槽訪問狀態: {'可訪問' if status.get('q_drive_accessible') else '不可訪問'}")
                st.info(f"API 版本: {status.get('version', '未知')}")
            
            # 添加設置選項
            st.header("設置")
            include_sources = st.checkbox("包含相關文件", value=True)
            max_sources = st.number_input("最大相關文件數", min_value=1, max_value=20, value=10)
            show_relevance = st.checkbox("顯示相關性理由", value=True, help="顯示為什麼這些文件與查詢相關")
            use_query_rewrite = st.checkbox("使用查詢優化", value=True, help="自動改寫查詢以獲得更準確的結果")

            # 添加清除歷史按鈕
            if st.button("清除歷史記錄", key="clear_history"):
                st.session_state.chat_history = []
                st.session_state.current_answer = None
                st._rerun()

        # 主要聊天界面
        st.header("💬 智能問答聊天")
        
        # 模型選擇 - 放在側邊欄
        with st.sidebar:
            st.markdown("### 模型設置")
            try:
                usable_models_response = requests.get(f"{API_URL}/api/usable-models", timeout=5)
                if usable_models_response.status_code == 200:
                    usable_models = usable_models_response.json()
                    if usable_models:
                        model_options = ["使用默認配置"] + [model['display_name'] for model in usable_models]
                        model_folder_map = {model['display_name']: model['folder_name'] for model in usable_models}
                        
                        selected_display_name = st.selectbox(
                            "選擇問答模型：",
                            options=model_options,
                            help="選擇用於問答的向量模型"
                        )
                        
                        # 獲取實際的文件夾名稱
                        if selected_display_name == "使用默認配置":
                            selected_model_folder = None
                        else:
                            selected_model_folder = model_folder_map.get(selected_display_name)
                    else:
                        st.warning("沒有可用的向量模型")
                        selected_model_folder = None
                else:
                    st.error("無法獲取可用模型列表")
                    selected_model_folder = None
            except Exception as e:
                st.error(f"獲取模型列表時出錯: {str(e)}")
                selected_model_folder = None
        
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
                        
                        # 文件附件樣式
                        files_html = "<div style='margin: 5px 0 15px 0;'>"
                        files_html += "<div style='font-size: 0.9em; color: #666; margin-bottom: 5px;'>📎 相關文件:</div>"
                        
                        for idx, (_, source) in enumerate(unique_files.items(), 1):
                            display_path = source["file_path"].replace(Q_DRIVE_PATH, DISPLAY_DRIVE_NAME)
                            score_display = f"{source['score']:.2f}" if source.get('score') is not None else "未知"
                            
                            files_html += f"""
                            <div style='background-color: #e3f2fd; border-left: 3px solid #2196f3; padding: 8px 12px; margin: 3px 0; border-radius: 4px; font-size: 0.85em;'>
                                <div style='font-weight: bold; color: #1976d2;'>{source['file_name']}</div>
                                <div style='color: #666; font-size: 0.8em;'>{display_path}</div>
                                <div style='color: #666; font-size: 0.8em;'>相關度: {score_display} | {source.get('location_info', '無位置信息')}</div>
                            </div>
                            """
                        
                        files_html += "</div>"
                        st.markdown(files_html, unsafe_allow_html=True)
                        
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
        
        # 問題輸入
        col1, col2 = st.columns([5, 1])
        
        with col1:
            question = st.text_input(
                "輸入您的問題...", 
                value="",
                placeholder="請輸入您的問題，例如：公司的年假政策是什麼？",
                key="chat_input",
                label_visibility="collapsed"
            )
        
        with col2:
            send_clicked = st.button("發送 📤", key="send_button", use_container_width=True)
        
        # 處理發送
        if (send_clicked or question) and question.strip():
            # 添加用戶問題到歷史
            with st.spinner("🤖 AI助手正在思考..."):
                try:
                    # 所有問題都通過正常的RAG流程處理
                    result = retry_with_backoff(
                        lambda: get_answer(question, include_sources, max_sources, use_query_rewrite, show_relevance, selected_model_folder)
                    )
                    
                    answer_text = result.get("answer", "無法獲取答案")
                    sources = result.get("sources", [])
                    
                    # 更新聊天歷史
                    update_chat_history(question, answer_text, sources)
                    
                    # 清空輸入框並重新運行
                    st.session_state.chat_input = ""
                    st._rerun()
                    
                except Exception as e:
                    error_msg = f"處理問題時發生錯誤: {str(e)}"
                    update_chat_history(question, error_msg, [])
                    st.session_state.chat_input = ""
                    st._rerun()
        
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
                    
                    # 檢查向量數據是否存在
                    try:
                        vector_models_resp = requests.get(f"{API_URL}/api/vector-models", timeout=5)
                        if vector_models_resp.status_code == 200:
                            vector_models = vector_models_resp.json()
                            
                            # 查找匹配的模型
                            current_model_exists = False
                            current_model_has_data = False
                            current_model_training = False
                            
                            for model in vector_models:
                                if (model['model_info'] and 
                                    model['model_info'].get('OLLAMA_MODEL') == selected_ollama_model and
                                    model['model_info'].get('OLLAMA_EMBEDDING_MODEL') == selected_embedding_model):
                                    current_model_exists = True
                                    current_model_has_data = model['has_data']
                                    current_model_training = model['is_training']
                                    break
                            
                            # 顯示狀態
                            st.markdown("### 當前選擇模型狀態")
                            if current_model_exists:
                                if current_model_training:
                                    st.warning("⏳ 該模型組合正在訓練中（資料夾內有.lock檔）...")
                                elif current_model_has_data:
                                    st.success("✅ 該模型組合已有向量數據，可進行增量訓練或重新索引")
                                else:
                                    st.info("📝 該模型組合已創建但無數據，可進行初始訓練")
                            else:
                                st.info("🆕 該模型組合尚未創建，將創建新的向量資料夾進行初始訓練")
                    except:
                        pass
                    
                    # 訓練按鈕
                    st.markdown("### 訓練操作")
                    btn_cols = st.columns(3)
                    
                    with btn_cols[0]:
                        if st.button("🚀 初始訓練", key="new_initial_training", 
                                   disabled=current_model_training if 'current_model_training' in locals() else False):
                            try:
                                resp = requests.post(
                                    f"{API_URL}/admin/training/initial",
                                    headers={"admin_token": admin_token},
                                    json={
                                        "ollama_model": selected_ollama_model,
                                        "ollama_embedding_model": selected_embedding_model
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
                                   disabled=current_model_training if 'current_model_training' in locals() else False):
                            try:
                                resp = requests.post(
                                    f"{API_URL}/admin/training/incremental",
                                    headers={"admin_token": admin_token},
                                    json={
                                        "ollama_model": selected_ollama_model,
                                        "ollama_embedding_model": selected_embedding_model
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
                                   disabled=current_model_training if 'current_model_training' in locals() else False):
                            try:
                                resp = requests.post(
                                    f"{API_URL}/admin/training/reindex",
                                    headers={"admin_token": admin_token},
                                    json={
                                        "ollama_model": selected_ollama_model,
                                        "ollama_embedding_model": selected_embedding_model
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
            with st.expander("📥 Log 下載"):
                try:
                    log_resp = requests.get(f"{API_URL}/admin/get_indexing_log", params={"log_type": "indexing"}, headers={"admin_token": admin_token}, timeout=10)
                    if log_resp.status_code == 200:
                        st.download_button("下載索引Log", data=log_resp.content, file_name="indexing.log", mime="text/plain")
                    else:
                        st.warning("尚無日誌可下載")
                except Exception as e:
                    st.error(f"下載Log失敗: {e}")
            
            # --- 監控當前狀態 ---
            st.markdown("---")
            st.subheader("📈 監控當前狀態")
            st_autorefresh(interval=60000, key="monitor_all_autorefresh")
            try:
                resp = requests.get(f"{API_URL}/admin/monitor_all", headers={"admin_token": admin_token}, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    status_text = data.get('status', '')
                    progress_text = data.get('progress', '')
                    realtime_text = data.get('realtime', '')
                else:
                    status_text = progress_text = realtime_text = "(監控API回應異常)"
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
