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

def get_answer(question: str, include_sources: bool = True, max_sources: Optional[int] = None, use_query_rewrite: bool = True, show_relevance: bool = True) -> Dict[str, Any]:
    """獲取問題答案"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/ask",
            json={
                "question": question,
                "include_sources": include_sources,
                "max_sources": max_sources,
                "use_query_rewrite": use_query_rewrite,
                "show_relevance": show_relevance
            }
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
            st.experimental_rerun()

    # 主要問答界面
    st.header("💬 智能問答")
    
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
                            lambda: get_answer(question, include_sources, max_sources, use_query_rewrite, show_relevance)
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
                        st.experimental_rerun()
    
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