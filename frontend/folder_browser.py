"""
文件夾瀏覽器組件
提供多層級文件夾導航和選擇功能
"""

import streamlit as st
import requests
from typing import Optional, Dict, Any

class FolderBrowser:
    """文件夾瀏覽器類"""
    
    def __init__(self, api_url: str):
        self.api_url = api_url
        
        # 初始化 session state
        if 'folder_browser_path' not in st.session_state:
            st.session_state.folder_browser_path = ""
        if 'selected_folder_path' not in st.session_state:
            st.session_state.selected_folder_path = None
    
    def render(self) -> Optional[str]:
        """
        渲染文件夾瀏覽器界面
        
        Returns:
            選中的文件夾路徑，如果沒有選中則返回 None
        """
        try:
            # 獲取當前路徑的文件夾數據
            folders_data = self._get_folder_data(st.session_state.folder_browser_path)
            if not folders_data:
                st.error("無法獲取文件夾數據")
                return None
            
            # 渲染導航欄
            self._render_navigation(folders_data)
            
            # 渲染統計信息
            self._render_statistics(folders_data)
            
            # 渲染文件夾列表
            selected_path = self._render_folder_list(folders_data)
            
            return selected_path
            
        except Exception as e:
            st.error(f"文件夾瀏覽器錯誤: {str(e)}")
            return None
    
    def _get_folder_data(self, path: str) -> Optional[Dict[str, Any]]:
        """獲取指定路徑的文件夾數據"""
        try:
            url = f"{self.api_url}/api/folders"
            if path:
                url += f"?path={path}"
            
            response = requests.get(url, timeout=15)
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"API 錯誤: {response.status_code}")
                return None
                
        except Exception as e:
            st.error(f"獲取文件夾數據失敗: {str(e)}")
            return None
    
    def _render_navigation(self, folders_data: Dict[str, Any]):
        """渲染導航欄"""
        st.markdown("**📍 當前位置:**")
        
        # 構建路徑顯示
        if folders_data.get("path_parts"):
            path_display = " > ".join([part["name"] for part in folders_data["path_parts"]])
            st.text(f"🏠 根目錄 > {path_display}")
        elif st.session_state.folder_browser_path:
            st.text(f"🏠 根目錄 > {st.session_state.folder_browser_path}")
        else:
            st.text("🏠 根目錄")
        
        # 導航按鈕
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            if folders_data.get("parent_path") is not None:
                if st.button("⬆️ 上級", key="nav_back", help="返回上級目錄"):
                    st.session_state.folder_browser_path = folders_data["parent_path"]
                    st.rerun()
        
        with col2:
            if st.button("🏠 根目錄", key="nav_home", help="返回根目錄"):
                st.session_state.folder_browser_path = ""
                st.rerun()
    
    def _render_statistics(self, folders_data: Dict[str, Any]):
        """渲染統計信息"""
        st.markdown("**📊 目錄統計:**")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="📁 文件夾",
                value=folders_data["total_folders"],
                help="當前目錄下的文件夾數量"
            )
        
        with col2:
            st.metric(
                label="📄 文件",
                value=folders_data["files_count"],
                help="當前目錄下的支持文件數量"
            )
        
        with col3:
            size_mb = folders_data.get('size_mb', 0)
            if size_mb >= 1024:
                size_display = f"{size_mb/1024:.1f} GB"
            else:
                size_display = f"{size_mb:.1f} MB"
            
            st.metric(
                label="💾 大小",
                value=size_display,
                help="當前目錄下所有文件的總大小"
            )
        
        with col4:
            # 計算平均文件大小
            if folders_data["files_count"] > 0:
                avg_size = size_mb / folders_data["files_count"]
                avg_display = f"{avg_size:.1f} MB"
            else:
                avg_display = "0 MB"
            
            st.metric(
                label="📊 平均",
                value=avg_display,
                help="平均每個文件的大小"
            )
    
    def _render_folder_list(self, folders_data: Dict[str, Any]) -> Optional[str]:
        """渲染文件夾列表並處理選擇"""
        
        # 當前目錄選項
        current_path = st.session_state.folder_browser_path
        current_files = folders_data["files_count"]
        current_size = folders_data.get('size_mb', 0)
        
        st.markdown("**📂 選擇範圍:**")
        
        # 當前目錄選項
        if current_files > 0:
            col1, col2 = st.columns([3, 1])
            with col1:
                size_display = f"{current_size:.1f} MB" if current_size < 1024 else f"{current_size/1024:.1f} GB"
                st.info(f"📍 當前目錄：{current_files} 個文件，{size_display}")
            with col2:
                if st.button("✅ 選擇", key="select_current", help="選擇當前目錄作為搜索範圍"):
                    st.session_state.selected_folder_path = current_path
                    st.success(f"🎯 已選擇：{current_path if current_path else '根目錄'}")
                    return current_path
        
        # 子文件夾列表
        if folders_data["folders"]:
            st.markdown("**📁 子文件夾:**")
            
            for i, folder in enumerate(folders_data["folders"]):
                with st.container():
                    col1, col2, col3 = st.columns([2, 1, 1])
                    
                    with col1:
                        # 文件夾信息
                        size_display = f"{folder.get('size_mb', 0):.1f} MB" if folder.get('size_mb', 0) < 1024 else f"{folder.get('size_mb', 0)/1024:.1f} GB"
                        st.write(f"📁 **{folder['name']}**")
                        st.caption(f"📄 {folder['files_count']} 個文件 • 💾 {size_display}")
                    
                    with col2:
                        if st.button("📂 進入", key=f"enter_{i}", help=f"進入 {folder['name']} 文件夾"):
                            st.session_state.folder_browser_path = folder["path"]
                            st.rerun()
                    
                    with col3:
                        if folder['files_count'] > 0:
                            if st.button("✅ 選擇", key=f"select_{i}", help=f"選擇 {folder['name']} 作為搜索範圍"):
                                st.session_state.selected_folder_path = folder["path"]
                                st.success(f"🎯 已選擇：{folder['name']}")
                                return folder["path"]
                        else:
                            st.button("❌ 無文件", key=f"empty_{i}", disabled=True, help="此文件夾沒有支持的文件")
                    
                    st.divider()
        else:
            if current_files == 0:
                st.warning("📂 此目錄沒有文件夾或支持的文件")
            else:
                st.info("📂 此目錄沒有子文件夾")
        
        return None
    
    def get_selected_path(self) -> Optional[str]:
        """獲取當前選中的路徑"""
        return st.session_state.get('selected_folder_path')
    
    def clear_selection(self):
        """清除選擇"""
        st.session_state.selected_folder_path = None
    
    def reset_browser(self):
        """重置瀏覽器到根目錄"""
        st.session_state.folder_browser_path = ""
        st.session_state.selected_folder_path = None