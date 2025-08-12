"""
前端幫助系統
"""

import streamlit as st
from typing import Dict, Any

def render_help_sidebar():
    """渲染幫助側邊欄"""
    with st.sidebar:
        st.markdown("## 📚 幫助中心")
        
        help_topics = {
            "🚀 快速開始": "quick_start",
            "🎯 平台選擇": "platform_selection", 
            "🤖 模型選擇": "model_selection",
            "⚙️ 推理引擎": "inference_engine",
            "🔧 RAG 模式": "rag_modes",
            "❓ 常見問題": "faq",
            "🔍 故障排除": "troubleshooting"
        }
        
        selected_topic = st.selectbox(
            "選擇幫助主題:",
            options=list(help_topics.keys()),
            key="help_topic"
        )
        
        if st.button("顯示幫助", key="show_help_btn"):
            # 初始化 session state 如果不存在
            if 'show_help' not in st.session_state:
                st.session_state.show_help = False
            if 'current_help_topic' not in st.session_state:
                st.session_state.current_help_topic = None
            
            st.session_state.show_help = True
            st.session_state.current_help_topic = help_topics[selected_topic]
            st.rerun()

def render_help_content(topic: str):
    """渲染幫助內容"""
    help_contents = {
        "quick_start": render_quick_start_help,
        "platform_selection": render_platform_selection_help,
        "model_selection": render_model_selection_help,
        "inference_engine": render_inference_engine_help,
        "rag_modes": render_rag_modes_help,
        "faq": render_faq_help,
        "troubleshooting": render_troubleshooting_help
    }
    
    if topic in help_contents:
        help_contents[topic]()
    else:
        st.error(f"找不到幫助主題: {topic}")

def render_quick_start_help():
    """快速開始幫助"""
    st.markdown("# 🚀 快速開始指南")
    
    st.markdown("""
    ## 歡迎使用 Q槽文件智能助手！
    
    ### 第一次使用？請按照以下步驟：
    
    #### 1️⃣ 選擇 AI 平台
    - **Ollama**: 本地推理，隱私保護，適合大多數場景
    - **Hugging Face**: 豐富模型，雲端推理，適合實驗研究
    
    #### 2️⃣ 選擇模型
    - **語言模型**: 用於生成回答的 AI 模型
    - **嵌入模型**: 用於文本向量化的模型
    
    #### 3️⃣ 選擇 RAG 模式
    - **傳統 RAG**: 使用預建向量資料庫，快速響應
    - **動態 RAG**: 即時檢索處理，無需預建資料庫
    
    #### 4️⃣ 開始使用
    - 在問答界面輸入您的問題
    - 系統會自動搜索相關文檔並生成回答
    
    ### 💡 小貼士
    - 首次使用可能需要下載模型，請耐心等待
    - 建議先使用較小的模型進行測試
    - 如遇問題，請查看故障排除部分
    """)

def render_platform_selection_help():
    """平台選擇幫助"""
    st.markdown("# 🎯 AI 平台選擇指南")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ## 🏠 Ollama 平台
        
        ### ✅ 優點
        - **隱私保護**: 完全本地運行
        - **無網路依賴**: 離線可用
        - **易於管理**: 簡單的模型管理
        - **資源效率**: 針對本地硬體優化
        
        ### ⚠️ 注意事項
        - 需要預先下載模型
        - 模型選擇相對有限
        - 需要足夠的本地儲存空間
        
        ### 🎯 適合場景
        - 企業內部部署
        - 隱私敏感應用
        - 網路受限環境
        - 穩定的生產環境
        """)
    
    with col2:
        st.markdown("""
        ## ☁️ Hugging Face 平台
        
        ### ✅ 優點
        - **豐富模型**: 大量開源模型可選
        - **最新技術**: 快速獲得最新模型
        - **高性能**: 支援 vLLM 推理引擎
        - **社群支援**: 活躍的開發者社群
        
        ### ⚠️ 注意事項
        - 需要網路連接下載模型
        - 首次使用需要較長準備時間
        - 對硬體要求較高
        
        ### 🎯 適合場景
        - 研究和實驗
        - 需要最新模型
        - 有充足硬體資源
        - 開發和測試環境
        """)
    
    st.markdown("""
    ## 🤔 如何選擇？
    
    | 需求 | 推薦平台 | 原因 |
    |------|----------|------|
    | 企業生產環境 | Ollama | 穩定、隱私、易管理 |
    | 研究實驗 | Hugging Face | 模型豐富、技術先進 |
    | 首次體驗 | Ollama | 設置簡單、快速上手 |
    | 高性能需求 | Hugging Face | 支援 vLLM 高性能推理 |
    """)

def render_model_selection_help():
    """模型選擇幫助"""
    st.markdown("# 🤖 模型選擇指南")
    
    st.markdown("""
    ## 🧠 語言模型 (Language Model)
    
    語言模型負責理解問題並生成回答。
    
    ### 📊 模型大小對比
    
    | 模型大小 | 參數量 | 記憶體需求 | 回答品質 | 響應速度 |
    |----------|--------|------------|----------|----------|
    | 小型 | < 1B | 4GB | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
    | 中型 | 1B-7B | 8-16GB | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
    | 大型 | 7B-20B | 16-40GB | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
    | 超大型 | > 20B | 40GB+ | ⭐⭐⭐⭐⭐ | ⭐⭐ |
    
    ### 🎯 推薦選擇
    - **開發測試**: Qwen/Qwen2-0.5B-Instruct
    - **生產環境**: openai/gpt-oss-20b（建議 vLLM）
    - **多語言/中文優先**: Qwen 系列（建議 0.5B/1.5B 規模；若 2.5 失敗請改用 2.0）
    """)
    
    st.markdown("""
    ## 🔤 嵌入模型 (Embedding Model)
    
    嵌入模型負責將文本轉換為向量，用於文檔檢索。
    
    ### 🌐 多語言支援
    
    | 模型 | 中文 | 英文 | 其他語言 | 大小 |
    |------|------|------|----------|------|
    | Multilingual MPNet | ✅ | ✅ | ✅ | 1.1GB |
    | Multilingual MiniLM | ✅ | ✅ | ✅ | 278MB |
    
    ### 🎯 推薦選擇
    - **推薦（多語言）**: Multilingual MiniLM
    - **高精度多語言**: Multilingual MPNet
    """)

def render_inference_engine_help():
    """推理引擎幫助"""
    st.markdown("# ⚙️ 推理引擎選擇指南")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ## 🔧 Transformers
        
        ### ✅ 優點
        - **穩定可靠**: 成熟的推理框架
        - **兼容性好**: 支援所有模型
        - **易於調試**: 豐富的調試工具
        - **記憶體友好**: 較低的記憶體需求
        
        ### ⚠️ 缺點
        - 推理速度較慢
        - 並發性能有限
        
        ### 🎯 適合場景
        - 開發和測試
        - 記憶體受限環境
        - 單用戶使用
        - 模型兼容性要求高
        """)
    
    with col2:
        st.markdown("""
        ## ⚡ vLLM
        
        ### ✅ 優點
        - **高性能**: 顯著提升推理速度
        - **高並發**: 支援多用戶同時使用
        - **記憶體優化**: 高效的記憶體管理
        - **生產就緒**: 專為生產環境設計
        
        ### ⚠️ 缺點
        - 對 GPU 記憶體要求較高
        - 模型支援有限
        - 設置較複雜
        
        ### 🎯 適合場景
        - 生產環境
        - 多用戶服務
        - 高性能需求
        - 充足的 GPU 資源
        """)
    
    st.markdown("""
    ## 📊 性能對比
    
    | 指標 | Transformers | vLLM | 提升幅度 |
    |------|--------------|------|----------|
    | 推理速度 | 基準 | 2-5x | 200-500% |
    | 並發能力 | 基準 | 10x+ | 1000%+ |
    | 記憶體效率 | 基準 | 1.5-2x | 50-100% |
    | GPU 利用率 | 60-70% | 90%+ | 30%+ |
    
    ## 🤔 如何選擇？
    
    - **開發階段**: 選擇 Transformers
    - **生產部署**: 選擇 vLLM（如果硬體支援）
    - **記憶體不足**: 選擇 Transformers
    - **高並發需求**: 選擇 vLLM
    """)

def render_rag_modes_help():
    """RAG 模式幫助"""
    st.markdown("# 🔧 RAG 模式選擇指南")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ## 📚 傳統 RAG
        
        ### 🔄 工作流程
        1. 預先索引所有文檔
        2. 建立向量資料庫
        3. 用戶提問時快速檢索
        4. 生成回答
        
        ### ✅ 優點
        - **快速響應**: 毫秒級檢索
        - **穩定性能**: 可預測的響應時間
        - **適合大量文檔**: 支援數萬個文件
        - **資源效率**: 查詢時資源消耗低
        
        ### ⚠️ 缺點
        - 需要預先建立索引
        - 新增文檔需要重新索引
        - 佔用儲存空間
        
        ### 🎯 適合場景
        - 文檔集合相對穩定
        - 需要快速響應
        - 大量文檔檢索
        - 生產環境
        """)
    
    with col2:
        st.markdown("""
        ## ⚡ 動態 RAG
        
        ### 🔄 工作流程
        1. 用戶提問
        2. 即時搜索相關文件
        3. 動態處理和向量化
        4. 生成回答
        
        ### ✅ 優點
        - **無需預處理**: 即開即用
        - **動態更新**: 自動處理新文檔
        - **靈活性高**: 適應變化的內容
        - **節省儲存**: 不需要預建索引
        
        ### ⚠️ 缺點
        - 響應時間較長
        - 計算資源消耗大
        - 性能不穩定
        
        ### 🎯 適合場景
        - 文檔經常變動
        - 小規模文檔集合
        - 實驗和測試
        - 即時性要求不高
        """)
    
    st.markdown("""
    ## 📊 詳細對比
    
    | 特性 | 傳統 RAG | 動態 RAG |
    |------|----------|----------|
    | 響應時間 | < 1秒 | 5-30秒 |
    | 文檔數量 | 數萬+ | 數百 |
    | 儲存需求 | 高 | 低 |
    | 計算需求 | 低 | 高 |
    | 設置複雜度 | 中 | 低 |
    | 內容更新 | 需重建索引 | 自動 |
    
    ## 🤔 選擇建議
    
    - **首次體驗**: 選擇動態 RAG
    - **生產環境**: 選擇傳統 RAG
    - **文檔 < 1000**: 動態 RAG
    - **文檔 > 1000**: 傳統 RAG
    - **內容經常變動**: 動態 RAG
    - **需要快速響應**: 傳統 RAG
    """)

def render_faq_help():
    """常見問題幫助"""
    st.markdown("# ❓ 常見問題")
    
    faqs = [
        {
            "question": "🤔 為什麼模型下載很慢？",
            "answer": """
            **可能原因和解決方案**：
            - **網路問題**: 檢查網路連接，嘗試使用代理
            - **伺服器繁忙**: 稍後重試，或選擇較小的模型
            - **防火牆限制**: 檢查企業防火牆設置
            - **磁盤空間不足**: 確保有足夠的儲存空間
            
            **加速下載**：
            - 設置 Hugging Face Token
            - 使用鏡像站點
            - 選擇較小的模型進行測試
            """
        },
        {
            "question": "💾 GPU 記憶體不足怎麼辦？",
            "answer": """
            **解決方案**：
            1. **選擇較小模型**: 使用參數量更少的模型
            2. **降低批處理大小**: 在設置中調整批處理大小
            3. **使用 CPU**: 切換到 CPU 推理（較慢但可用）
            4. **調整 GPU 記憶體使用率**: 降低 vLLM 記憶體使用率
            5. **關閉其他程序**: 釋放 GPU 記憶體
            
            **記憶體需求參考**：
            - 小型模型 (< 1B): 4GB
            - 中型模型 (1-7B): 8-16GB  
            - 大型模型 (7-20B): 16-40GB
            """
        },
        {
            "question": "🔄 如何更新文檔索引？",
            "answer": """
            **傳統 RAG**：
            1. 進入管理員後台
            2. 選擇對應的模型
            3. 點擊「增量訓練」或「重新索引」
            4. 等待索引完成
            
            **動態 RAG**：
            - 無需手動更新，系統會自動處理新文檔
            
            **注意事項**：
            - 索引過程中模型無法使用
            - 大量文檔索引可能需要較長時間
            - 建議在低峰時段進行索引更新
            """
        },
        {
            "question": "🌐 支援哪些語言？",
            "answer": """
            **系統支援的語言**：
            - 🇹🇼 繁體中文
            - 🇨🇳 简体中文  
            - 🇺🇸 English
            - 🇹🇭 ไทย (泰文)
            - 🌍 Dynamic (自動檢測)
            
            **語言選擇建議**：
            - 根據您的文檔主要語言選擇
            - Dynamic 模式會自動適應問題語言
            - 多語言文檔建議使用 Dynamic 模式
            """
        },
        {
            "question": "⚡ 如何提升問答速度？",
            "answer": """
            **優化建議**：
            1. **使用 vLLM 推理引擎** (Hugging Face 平台)
            2. **選擇較小的模型** 進行測試
            3. **使用傳統 RAG** 而非動態 RAG
            4. **調整批處理大小** 適應硬體
            5. **使用 GPU 推理** 而非 CPU
            6. **關閉不必要的功能** 如來源顯示
            
            **硬體建議**：
            - 使用 SSD 儲存
            - 充足的 RAM (32GB+)
            - 高性能 GPU (RTX 3080+)
            """
        }
    ]
    
    for faq in faqs:
        with st.expander(faq["question"]):
            st.markdown(faq["answer"])

def render_troubleshooting_help():
    """故障排除幫助"""
    st.markdown("# 🔍 故障排除指南")
    
    st.markdown("""
    ## 🚨 常見錯誤及解決方案
    """)
    
    issues = [
        {
            "title": "🔌 API 服務無法啟動",
            "symptoms": [
                "瀏覽器無法訪問 http://localhost:8000",
                "前端顯示連接錯誤",
                "服務啟動失敗"
            ],
            "solutions": [
                "檢查端口 8000 是否被佔用",
                "確認 Python 環境和依賴包正確安裝",
                "查看 logs/app.log 日誌文件",
                "嘗試重新啟動服務",
                "檢查防火牆設置"
            ]
        },
        {
            "title": "🤖 模型載入失敗",
            "symptoms": [
                "模型下載中斷",
                "載入模型時出錯",
                "記憶體不足錯誤"
            ],
            "solutions": [
                "檢查網路連接和代理設置",
                "確保有足夠的磁盤空間",
                "選擇較小的模型進行測試",
                "清理模型緩存重新下載",
                "檢查 GPU 記憶體是否充足"
            ]
        },
        {
            "title": "📄 文檔索引問題",
            "symptoms": [
                "索引過程中斷",
                "找不到文檔",
                "索引速度很慢"
            ],
            "solutions": [
                "檢查 Q 槽路徑設置是否正確",
                "確認文檔格式受支援",
                "檢查文件權限",
                "調整批處理大小",
                "查看索引日誌"
            ]
        },
        {
            "title": "💬 問答品質問題",
            "symptoms": [
                "回答不相關",
                "找不到相關文檔",
                "回答品質差"
            ],
            "solutions": [
                "嘗試重新表述問題",
                "檢查文檔是否已正確索引",
                "調整相似度閾值",
                "使用更大的語言模型",
                "檢查文檔內容品質"
            ]
        }
    ]
    
    for issue in issues:
        with st.expander(issue["title"]):
            st.markdown("**症狀**:")
            for symptom in issue["symptoms"]:
                st.markdown(f"- {symptom}")
            
            st.markdown("**解決方案**:")
            for solution in issue["solutions"]:
                st.markdown(f"- {solution}")
    
    st.markdown("""
    ## 🛠️ 診斷工具
    
    ### 系統檢查
    ```bash
    # 檢查整體系統狀態
    python tests/test_complete_system.py --check
    
    # 檢查 Hugging Face 環境
    python scripts/check_hf_environment.py
    
    # 檢查環境配置
    python scripts/check_env_config.py
    ```
    
    ### 日誌查看
    - **應用日誌**: `logs/app.log`
    - **索引日誌**: `logs/indexing.log`
    - **錯誤日誌**: `logs/error.log`
    
    ### 重置系統
    ```bash
    # 清理模型緩存
    rm -rf ./models/cache/*
    
    # 重置用戶配置
    rm -f config/user_setup.json
    
    # 重新啟動服務
    python scripts/quick_start.py
    ```
    
    ## 📞 獲取幫助
    
    如果問題仍未解決，請：
    1. 收集相關日誌文件
    2. 記錄錯誤訊息和操作步驟
    3. 提供系統環境信息
    4. 聯繫技術支援團隊
    """)

def show_help_modal():
    """顯示幫助模態框"""
    if st.session_state.get('show_help', False):
        with st.container():
            col1, col2, col3 = st.columns([1, 8, 1])
            
            with col2:
                with st.container():
                    # 關閉按鈕
                    if st.button("❌ 關閉幫助", key="close_help"):
                        st.session_state.show_help = False
                        st.rerun()
                    
                    # 顯示幫助內容
                    topic = st.session_state.get('current_help_topic', 'quick_start')
                    render_help_content(topic)
