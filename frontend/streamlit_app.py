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

        # 主要問答界面
        st.header("💬 智能問答")
        
        # 模型選擇 - 每次都重新獲取最新的模型列表
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
                        help="選擇用於問答的向量模型，每次都會重新讀取最新的模型列表。資料夾內有.lock檔為訓練中不可使用。"
                    )
                    
                    # 獲取實際的文件夾名稱
                    if selected_display_name == "使用默認配置":
                        selected_model_folder = None
                    else:
                        selected_model_folder = model_folder_map.get(selected_display_name)
                else:
                    st.warning("沒有可用的向量模型，請先進行模型訓練")
                    selected_model_folder = None
            else:
                st.error("無法獲取可用模型列表")
                selected_model_folder = None
        except Exception as e:
            st.error(f"獲取模型列表時出錯: {str(e)}")
            selected_model_folder = None
        
        # 問題輸入
        question = st.text_input(
            "請輸入您的問題：", 
            value=st.session_state.current_question,
            help="您可以輸入問題或特殊命令，如 '列出文件' 來查看已索引的文件",
            key="question_input",
            on_change=handle_text_input_change
        )
        
        # 提問按鈕
        col1, col2 = st.columns([1, 5])
        with col1:
            search_clicked = st.button("提問", key="search_button")
            if search_clicked and question:
                # 如果點擊提問按鈕，直接設置搜索標誌
                st.session_state.current_question = question
                st.session_state.run_search = True
        
        # 檢查是否需要執行搜索
        run_search = 'run_search' in st.session_state and st.session_state.run_search
        
        # 處理特殊命令或問題
        if question and run_search:
            # 重置搜索標誌
            st.session_state.run_search = False
            
            # 清除之前的回答，創建新的容器顯示結果
            st.session_state.current_answer = None
            answer_container = st.container()
            
            with answer_container:
                # 處理特殊命令
                if any(keyword in question.strip().lower() for keyword in ["列出文件", "列出已索引文件", "文件列表", "show files", "list files", "索引文件", "已索引", "列出", "所有文件"]):
                    with st.spinner("正在獲取文件列表..."):
                        indexed_files = get_indexed_files()
                        if indexed_files:
                            # 直接顯示結果，不使用標準回答格式
                            st.markdown(f"## 已索引文件清單（共 {len(indexed_files)} 個文件）")
                            
                            # 按文件名對文件進行分組和去重
                            unique_files = {}
                            for file in indexed_files:
                                file_name = os.path.basename(file["file_path"])
                                if file_name not in unique_files:
                                    unique_files[file_name] = file
                            
                            # 創建表格顯示文件基本信息
                            file_data = []
                            for idx, (_, file) in enumerate(unique_files.items(), 1):
                                file_name = os.path.basename(file["file_path"])
                                file_data.append({
                                    "序號": idx,
                                    "文件名": file_name,
                                    "文件類型": file["file_type"],
                                    "大小 (KB)": f"{file['file_size']/1024:.2f}",
                                    "最後修改時間": file["last_modified"]
                                })
                            
                            # 顯示表格
                            st.table(file_data)
                            
                            # 下方顯示完整路徑信息，按文件名排序
                            st.markdown("### 文件詳細路徑")
                            for idx, (_, file) in enumerate(unique_files.items(), 1):
                                file_name = os.path.basename(file["file_path"])
                                # 將 Q_DRIVE_PATH 換成 DISPLAY_DRIVE_NAME
                                display_path = file["file_path"].replace(Q_DRIVE_PATH, DISPLAY_DRIVE_NAME)
                                st.write(f"{idx}. **{file_name}** - {display_path}")
                            
                            # 保存到歷史記錄
                            file_list_text = f"已找到 {len(indexed_files)} 個文件"
                            update_chat_history(question, file_list_text)
                        else:
                            st.warning("未找到已索引文件。請確保已運行索引程序或檢查日誌以獲取詳細信息。")
                            update_chat_history(question, "未找到已索引文件")
                # 處理正常問題
                else:
                    try:
                        with st.spinner("正在思考..."):
                            # 使用重試機制獲取答案
                            result = retry_with_backoff(
                                lambda: get_answer(question, include_sources, max_sources, use_query_rewrite, show_relevance, selected_model_folder)
                            )
                            
                            # 如果啟用了查詢優化，顯示優化後的查詢
                            if use_query_rewrite and "rewritten_query" in result:
                                rewritten_query = result["rewritten_query"]
                                if rewritten_query != question:
                                    with st.expander("查看優化後的查詢"):
                                        st.markdown("**原始查詢:**")
                                        st.info(question)
                                        st.markdown("**優化後查詢:**")
                                        st.success(rewritten_query)
                            
                            # 直接顯示答案，不添加標題
                            answer_text = result.get("answer", "無法獲取答案")
                            st.write(answer_text)
                            
                            # 保存當前回答
                            st.session_state.current_answer = {
                                "answer": answer_text,
                                "sources": result.get("sources", [])
                            }
                            
                            # 顯示來源文檔
                            if include_sources and result.get("sources"):
                                # 使用集合去重
                                unique_files = {}
                                for source in result["sources"]:
                                    file_path = source["file_path"]
                                    if file_path not in unique_files:
                                        unique_files[file_path] = source
                                
                                # 顯示相關文件標題
                                st.markdown("## 相關文件")
                                
                                # 創建相關文件表格（去重後）
                                source_data = []
                                for idx, (_, source) in enumerate(unique_files.items(), 1):
                                    # 安全地處理分數格式化
                                    if 'score' in source and source['score'] is not None:
                                        score_display = f"{source['score']:.2f}"
                                    else:
                                        score_display = "未知"
                                        
                                    source_data.append({
                                        "序號": idx,
                                        "文件名": source['file_name'],
                                        "位置": source.get('location_info', '無位置信息'),
                                        "相關度": score_display
                                    })
                                
                                # 顯示相關文件表格
                                st.table(source_data)
                                
                                # 顯示詳細相關文件信息
                                st.markdown("### 文件詳細信息")
                                for idx, (_, source) in enumerate(unique_files.items(), 1):
                                    with st.expander(f"**文件 {idx}: {source['file_name']}**", expanded=False):
                                        display_path = source["file_path"].replace(Q_DRIVE_PATH, DISPLAY_DRIVE_NAME)
                                        st.write(f"文件路徑: {display_path}")
                                        
                                        if source.get("location_info"):
                                            st.write(f"位置信息: {source['location_info']}")
                                        
                                        # 安全地處理分數顯示
                                        if source.get("score") is not None:
                                            st.write(f"相關度分數: {source['score']:.4f}")
                                        else:
                                            st.write("相關度分數: 未知")
                                        
                                        # 顯示相關性理由（如果有）
                                        if show_relevance and source.get("relevance_reason"):
                                            st.markdown("**相關性理由:**")
                                            st.success(source["relevance_reason"])
                                        
                                        if source.get("content"):
                                            st.markdown("**相關內容:**")
                                            st.info(source["content"])
                            
                            # 更新聊天歷史
                            update_chat_history(question, answer_text, result.get("sources", []))
                    except Exception as e:
                        st.error(f"處理問題時發生錯誤: {str(e)}")
                        if st.button("重試"):
                            st._rerun()
        
        # 顯示聊天歷史
        if st.session_state.chat_history:
            st.markdown("---")
            st.markdown("## 歷史問答")
            
            for i, chat in enumerate(reversed(st.session_state.chat_history)):
                with st.expander(f"問題: {chat['question']} ({chat['timestamp']})", expanded=False):
                    st.markdown("**回答:**")
                    st.write(chat["answer"])
                    
                    # 顯示來源（如果有）
                    if "sources" in chat and chat["sources"]:
                        st.markdown("**相關文件:**")
                        for idx, source in enumerate(chat["sources"], 1):
                            display_path = source["file_path"].replace(Q_DRIVE_PATH, DISPLAY_DRIVE_NAME)
                            st.write(f"{idx}. {source['file_name']} - {display_path}")
        
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