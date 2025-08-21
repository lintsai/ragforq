import os
import sys
import logging
import json
from typing import List, Dict, Any, Optional
from collections import deque
from pathlib import Path
from fastapi import FastAPI, HTTPException, Body, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import subprocess
from fastapi.responses import FileResponse, PlainTextResponse

# 添加項目根目錄到路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.monitor_indexing import get_status_text, get_progress_text, get_monitor_text, get_indexing_status

from config.config import APP_HOST, APP_PORT, is_q_drive_accessible, ADMIN_TOKEN, SELECTED_PLATFORM, LOGS_DIR, BACKUPS_DIR
from rag_engine.rag_engine_factory import get_rag_engine_for_language, get_supported_languages, validate_language
from indexer.document_indexer import DocumentIndexer
from langchain_core.documents import Document
from utils.ollama_utils import ollama_utils
from utils.vector_db_manager import vector_db_manager
from utils.platform_manager import get_platform_manager
from utils.setup_flow_manager import get_setup_flow_manager

# 設置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 讓 Docker 情境下（直接以 uvicorn 啟動）也會輸出到 logs/app.log
def _configure_file_logging():
    try:
        # 確保日誌目錄存在
        from config.config import LOGS_DIR  # 已在上方導入，但保留以利靜態掃描
        logs_dir_path = Path(LOGS_DIR)
        logs_dir_path.mkdir(parents=True, exist_ok=True)

        log_file = logs_dir_path / "app.log"

        # 檢查是否已經有指向 app.log 的處理器，避免重複寫入
        root_logger = logging.getLogger()
        for h in root_logger.handlers:
            if isinstance(h, logging.FileHandler):
                try:
                    if hasattr(h, 'baseFilename') and str(h.baseFilename).endswith(str(log_file)):
                        return
                except Exception:
                    pass

        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        # 掛到 root，讓所有子 logger（含 uvicorn）也能寫入
        root_logger.addHandler(file_handler)

        # 明確掛到 uvicorn 相關 logger，以防其自有設定不經過 root
        for name in ["uvicorn", "uvicorn.error", "uvicorn.access"]:
            uv_logger = logging.getLogger(name)
            uv_logger.propagate = True
            uv_logger.addHandler(file_handler)

        root_logger.setLevel(logging.INFO)
    except Exception as e:
        # 僅記錄，不阻斷服務啟動
        logger.warning(f"設定檔案日誌處理器失敗: {e}")

_configure_file_logging()

# 創建FastAPI應用
app = FastAPI(
    title="Q槽文件智能助手 API",
    description="提供Q槽文件檢索和問答功能的API",
    version="1.0.0"
)

# === 依賴審計排程器 (Task 6 自動化) ===
_AUDIT_THREAD_STARTED = False
def _start_dependency_audit_scheduler():
    """啟動背景執行緒定期寫入依賴審計 (避免過多 I/O, 預設每 6 小時)。

    透過環境變數 DEP_AUDIT_INTERVAL_MIN 覆寫，最小 30 分鐘。
    僅在主進程啟動一次，避免 uvicorn --reload 產生多重執行；簡易判斷使用 os.getpid().
    """
    global _AUDIT_THREAD_STARTED
    if _AUDIT_THREAD_STARTED:
        return
    _AUDIT_THREAD_STARTED = True
    import threading, time as _time, os as _os
    interval_min = int(os.getenv('DEP_AUDIT_INTERVAL_MIN', '360') or '360')
    if interval_min < 30:
        interval_min = 30

    def _worker():
        while True:
            try:
                # 延遲首輪 2 分鐘，避免啟動競態
                _time.sleep(120)
                from importlib import metadata as importlib_metadata  # noqa
                py_deps = _load_pyproject_deps(Path('pyproject.toml'))
                req_versions = _load_requirements_versions(Path('requirements.txt'))
                items: List[DependencyStatusItem] = []
                for pkg in CORE_DEP_PACKAGES:
                    py_ver=None
                    for k,v in py_deps.items():
                        if k.lower()==pkg:
                            if isinstance(v, dict):
                                py_ver=v.get('version')
                            else:
                                py_ver=v
                            break
                    req_ver = req_versions.get(pkg)
                    try:
                        inst_ver = importlib_metadata.version(pkg)
                    except Exception:
                        inst_ver = None
                    status='aligned'
                    want = req_ver or py_ver
                    if not (py_ver or req_ver):
                        status='missing'
                    elif inst_ver and want and inst_ver != want:
                        status='mismatch'
                    elif not inst_ver:
                        status='missing'
                    items.append(DependencyStatusItem(package=pkg, pyproject=py_ver, requirements=req_ver, installed=inst_ver, status=status))
                _dependency_audit_write(items)
                logger.debug('[DependencyAuditScheduler] 寫入依賴審計完成')
            except Exception as e:
                logger.debug(f"[DependencyAuditScheduler] 失敗: {e}")
            _time.sleep(interval_min * 60)

    try:
        t = threading.Thread(target=_worker, name='DepAuditScheduler', daemon=True)
        t.start()
    except Exception as e:
        logger.warning(f"啟動依賴審計排程失敗: {e}")

_start_dependency_audit_scheduler()

# 添加CORS中間件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允許所有來源，可根據需要限制
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 定義請求和回應模型
class QuestionRequest(BaseModel):
    question: str
    include_sources: bool = True
    max_sources: Optional[int] = None
    use_query_rewrite: bool = True
    language: str = "繁體中文"  # 回答語言
    selected_model: Optional[str] = None  # 選擇的模型文件夾名稱（支援版本）
    use_dynamic_rag: bool = False  # 是否使用Dynamic RAG
    ollama_model: Optional[str] = None  # Dynamic RAG需要的語言模型
    ollama_embedding_model: Optional[str] = None  # Dynamic RAG需要的嵌入模型
    show_relevance: bool = True
    platform: Optional[str] = None # 新增：動態RAG使用的平台
    folder_path: Optional[str] = None  # 新增：指定搜索的文件夾路徑

class SourceInfo(BaseModel):
    file_name: str
    file_path: str
    location_info: Optional[str] = None
    relevance_reason: Optional[str] = None
    score: Optional[float] = None

class QuestionResponse(BaseModel):
    answer: str
    sources: List[SourceInfo] = []
    rewritten_query: Optional[str] = None
    language: Optional[str] = None # 新增語言欄位
    file_count_warning: Optional[str] = None # 文件數量警告
    scope_info: Optional[Dict[str, Any]] = None  # 動態RAG範圍資訊（檔案估算、警告等）

class ScopeInfoRequest(BaseModel):
    """請求動態RAG檔案範圍與估算資訊"""
    language: str = "繁體中文"
    ollama_model: str
    ollama_embedding_model: str
    platform: Optional[str] = None
    folder_path: Optional[str] = None

class ScopeInfoResponse(BaseModel):
    """回傳動態RAG範圍資訊與是否建議阻擋"""
    scope_info: Dict[str, Any]
    file_count_warning: Optional[str] = None
    block_recommended: bool = False
    blocking_reason: Optional[str] = None

class StatusResponse(BaseModel):
    status: str
    q_drive_accessible: bool
    version: str = "1.0.0"
    runtime_state: Optional[Dict[str, Any]] = None  # 新增：運行態索引與模型狀態

class FileInfo(BaseModel):
    file_path: str
    last_modified: str
    file_size: int
    file_type: str

class ModelInfo(BaseModel):
    name: str
    size: int
    modified: str

class VectorModelInfo(BaseModel):
    folder_name: str
    display_name: str
    version: Optional[str]  # 版本信息
    model_info: Optional[Dict[str, str]]
    is_training: bool
    has_data: bool

class TrainingRequest(BaseModel):
    ollama_model: str
    ollama_embedding_model: str
    version: Optional[str] = None  # 版本標識，如日期 "20250722"

class PlatformSelectionRequest(BaseModel):
    platform_type: str

class ModelSelectionRequest(BaseModel):
    language_model: str
    embedding_model: str
    inference_engine: str = "transformers"

class RAGModeSelectionRequest(BaseModel):
    rag_mode: str

# 初始化RAG引擎緩存
rag_engines = {}

def get_rag_engine(selected_model: Optional[str] = None, language: str = "繁體中文", 
                   use_dynamic_rag: bool = False, ollama_model: str = None, 
                   ollama_embedding_model: str = None, platform: Optional[str] = None,
                   folder_path: Optional[str] = None):
    """
    獲取或初始化指定語言的RAG引擎，支持智能模型選擇和訓練狀態檢查
    
    Args:
        selected_model: 選擇的模型文件夾名稱，如果為 None 則自動選擇最佳可用模型
        language: 目標語言
        use_dynamic_rag: 是否使用Dynamic RAG
        ollama_model: Ollama語言模型（Dynamic RAG需要）
        ollama_embedding_model: Ollama嵌入模型（Dynamic RAG需要）
    """
    global rag_engines
    
    # 驗證語言支持
    if not validate_language(language):
        logger.warning(f"不支持的語言: {language}，默認使用繁體中文")
        language = "繁體中文"

    # 資料夾路徑安全處理（僅在 dynamic 模式使用）
    if use_dynamic_rag and folder_path:
        try:
            from config.config import Q_DRIVE_PATH
            base_root = Path(Q_DRIVE_PATH).resolve()
            candidate = (base_root / folder_path).resolve()
            if not str(candidate).startswith(str(base_root)):
                logger.warning(f"[FolderScopeSecurity] 檢測到越界的 folder_path: {folder_path} -> {candidate}，改用根目錄")
                folder_path = None
            else:
                # 將路徑儲存為相對於 base_root 的標準化形式，避免含有盤符
                try:
                    folder_path = str(candidate.relative_to(base_root)).replace('\\', '/')
                except Exception:
                    folder_path = None
        except Exception as e:
            logger.warning(f"[FolderScopeSecurity] 處理 folder_path 失敗 '{folder_path}': {e}，改用根目錄")
            folder_path = None
    
    # 創建緩存鍵，包含語言和平台信息
    if use_dynamic_rag:
        # Dynamic RAG 引擎需要根據平台、語言及 folder_path 進行緩存，避免不同範圍混用
        current_platform = platform or "ollama"  # 如果未提供平台，默認為 ollama
        # 將 folder_path 正規化，避免 None 與空字串衝突
        folder_key = (folder_path.strip().replace('\\', '/')) if folder_path else "__root__"
        cache_key = f"dynamic_{current_platform}_{language}_{ollama_model}_{ollama_embedding_model}_{folder_key}"
    else:
        # 對於非動態RAG，需要檢測平台來生成正確的緩存鍵
        if selected_model:
            from config.config import detect_platform_from_model
            # 這裡需要獲取實際的模型名稱來檢測平台
            cache_key = f"{selected_model}_{language}"
        else:
            cache_key = f"auto_selected_{language}"
    
    try:
        if cache_key not in rag_engines:
            if use_dynamic_rag:
                # 使用Dynamic RAG
                logger.info(f"初始化Dynamic RAG引擎 - 語言: {language}, 模型: {ollama_model}, 嵌入: {ollama_embedding_model}")
                
                if not ollama_model or not ollama_embedding_model:
                    raise HTTPException(status_code=400, detail="Dynamic RAG需要指定ollama_model和ollama_embedding_model")
                
                # 使用工廠創建Dynamic RAG引擎，傳入正確的用戶語言
                # 我們需要告訴工廠這是 Dynamic RAG，但同時傳入用戶的實際語言
                # 使用特殊的語言標識符來表示 Dynamic RAG + 用戶語言
                dynamic_language_key = f"Dynamic_{language}"
                current_platform = platform or "ollama"
                rag_engines[cache_key] = get_rag_engine_for_language(
                    dynamic_language_key, None, ollama_model, ollama_embedding_model, current_platform, folder_path=folder_path
                )
                
            elif selected_model:
                # 使用指定的模型
                logger.info(f"初始化{language}RAG引擎 - 指定模型: {selected_model}")
                
                # 獲取模型信息
                models = vector_db_manager.list_available_models()
                model_info = None
                model_path = None
                
                for model in models:
                    if model['folder_name'] == selected_model:
                        model_info = model['model_info']
                        model_path = model['folder_path']
                        break
                
                if not model_info or not model_path:
                    raise HTTPException(status_code=404, detail=f"找不到模型: {selected_model}")
                
                if not vector_db_manager.has_vector_data(Path(model_path)):
                    raise HTTPException(status_code=400, detail=f"模型 {selected_model} 沒有向量數據")
                
                # 檢查訓練狀態和鎖定有效性
                if vector_db_manager.is_training(Path(model_path)):
                    is_valid, reason = vector_db_manager.is_lock_valid(Path(model_path))
                    if is_valid:
                        raise HTTPException(status_code=423, detail=f"模型 {selected_model} 正在訓練中: {reason}")
                    else:
                        # 鎖定無效，自動清理
                        logger.warning(f"檢測到無效鎖定，自動清理: {reason}")
                        vector_db_manager.remove_lock_file(Path(model_path))
                
                # 創建文檔索引器
                document_indexer = DocumentIndexer(
                    vector_db_path=model_path,
                    ollama_embedding_model=model_info['OLLAMA_EMBEDDING_MODEL']
                )
                
                # 智能檢測平台
                from config.config import detect_platform_from_model
                platform = detect_platform_from_model(model_info['OLLAMA_MODEL'])
                
                # 使用工廠創建對應語言的RAG引擎
                rag_engines[cache_key] = get_rag_engine_for_language(
                    language=language,
                    document_indexer=document_indexer,
                    ollama_model=model_info['OLLAMA_MODEL'],
                    platform=platform
                )
            else:
                # 自動選擇最佳可用模型
                logger.info(f"自動選擇最佳可用模型用於{language}...")
                
                # 首先嘗試獲取可用的向量模型
                usable_models = vector_db_manager.get_usable_models()
                
                if usable_models:
                    # 使用第一個可用的向量模型
                    best_model = usable_models[0]
                    model_info = best_model['model_info']
                    model_path = best_model['folder_path']
                    
                    logger.info(f"自動選擇向量模型: {best_model['display_name']} 用於{language}")
                    
                    # 創建文檔索引器
                    document_indexer = DocumentIndexer(
                        vector_db_path=model_path,
                        ollama_embedding_model=model_info['OLLAMA_EMBEDDING_MODEL']
                    )
                    
                    # 智能檢測平台
                    from config.config import detect_platform_from_model
                    platform = detect_platform_from_model(model_info['OLLAMA_MODEL'])
                    
                    # 使用工廠創建對應語言的RAG引擎
                    rag_engines[cache_key] = get_rag_engine_for_language(
                        language=language,
                        document_indexer=document_indexer,
                        ollama_model=model_info['OLLAMA_MODEL'],
                        platform=platform
                    )
                else:
                    # 回退到默認配置
                    logger.info(f"沒有可用的向量模型，使用默認配置用於{language}...")
                    
                    # 獲取可用的 Ollama 模型
                    available_models = ollama_utils.get_model_names()
                    if not available_models:
                        raise HTTPException(status_code=503, detail="沒有可用的 Ollama 模型，請確保 Ollama 服務正在運行並已下載模型")
                    
                    # 選擇默認模型
                    default_embedding_model = None
                    default_language_model = None
                    
                    # 優先選擇常見的嵌入模型
                    for model in available_models:
                        if 'embed' in model.lower():
                            default_embedding_model = model
                            break
                    
                    # 選擇語言模型
                    for model in available_models:
                        if 'embed' not in model.lower():
                            default_language_model = model
                            break
                    
                    # 如果沒有找到合適的模型，使用第一個可用模型
                    if not default_embedding_model:
                        default_embedding_model = available_models[0]
                    if not default_language_model:
                        default_language_model = available_models[0]
                    
                    logger.info(f"使用默認模型用於{language} - 語言模型: {default_language_model}, 嵌入模型: {default_embedding_model}")
                    
                    # 使用默認模型創建文檔索引器
                    document_indexer = DocumentIndexer(ollama_embedding_model=default_embedding_model)
                    
                    # 使用工廠創建對應語言的RAG引擎（默認使用 Ollama）
                    rag_engines[cache_key] = get_rag_engine_for_language(
                        language=language,
                        document_indexer=document_indexer,
                        ollama_model=default_language_model,
                        platform="ollama"
                    )
        
        return rag_engines[cache_key]
    
    except HTTPException:
        # 重新拋出 HTTP 異常
        raise
    except Exception as e:
        logger.error(f"初始化{language} RAG 引擎時出錯: {str(e)}")
        raise HTTPException(status_code=500, detail=f"初始化{language} RAG 引擎失敗: {str(e)}")

# 管理API - 權限驗證

async def check_admin(request: Request):
    token = request.headers.get("admin_token")
    if token != ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="未授權: token錯誤")

# 定義API路由
@app.get("/", response_model=StatusResponse)
async def get_status():
    """獲取API狀態"""
    q_drive_accessible = is_q_drive_accessible()
    runtime_state = None
    try:
        from utils.state_manager import load_state
        runtime_state = load_state()
    except Exception:
        runtime_state = None
    return {
        "status": "運行中",
        "q_drive_accessible": q_drive_accessible,
        "version": "1.0.0",
        "runtime_state": runtime_state
    }

@app.get("/api/runtime-state")
async def get_runtime_state():
    """單獨獲取 runtime_state（提供前端輪詢顯示索引狀態）。"""
    try:
        from utils.state_manager import load_state
        return load_state()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"無法讀取 runtime_state: {e}")

@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """
    提問API
    
    根據用戶問題搜索相關文檔並生成回答
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="問題不能為空")
    
    try:
        # 確保問題文本正確編碼
        question = request.question
        if isinstance(question, bytes):
            question = question.decode('utf-8')
        
        # 記錄接收到的問題以便調試
        logger.info(f"接收到問題: {repr(question)}")
        
        # 調試：記錄請求參數
        logger.info(f"請求參數調試 - use_dynamic_rag: {request.use_dynamic_rag}, ollama_model: {request.ollama_model}, ollama_embedding_model: {request.ollama_embedding_model}")
        
        # 文件數量警告
        # file_count_warning = None
        # if request.use_dynamic_rag and request.folder_path:
        #     validation_result = _get_folder_validation_results(request.folder_path)
        #     if validation_result.get("warning_level") == "high":
        #         file_count_warning = validation_result.get("suggestion")

        # 獲取RAG引擎
        engine = get_rag_engine(
            selected_model=request.selected_model,
            language=request.language,
            use_dynamic_rag=request.use_dynamic_rag,
            ollama_model=request.ollama_model if request.use_dynamic_rag else None,
            ollama_embedding_model=request.ollama_embedding_model if request.use_dynamic_rag else None,
            platform=request.platform if request.use_dynamic_rag else None,
            folder_path=request.folder_path
        )
        
        # 檢查動態檔案數警告（前面尚未阻擋且有範圍時，保留提示用於回傳）
        scope_info = None
        file_count_warning = None
        if request.use_dynamic_rag:
            if hasattr(engine, 'get_scope_info'):
                try:
                    scope_info = engine.get_scope_info()
                except Exception:
                    scope_info = None
            if hasattr(engine, 'get_file_count_warning'):
                file_count_warning = engine.get_file_count_warning()

        # 若為動態RAG且範圍過大未限制 => 依新版信心與估算策略阻擋
        if request.use_dynamic_rag and hasattr(engine, 'get_scope_info'):
            try:
                local_scope_info = scope_info or engine.get_scope_info()
                if local_scope_info:
                    warning_level = local_scope_info.get('file_count_warning_level')
                    est_count = local_scope_info.get('estimated_file_count') or 0
                    folder_limited = local_scope_info.get('folder_limited')
                    est_conf = local_scope_info.get('estimation_confidence')
                    sec_rate_flag = local_scope_info.get('security_rate_flag')
                    block = False
                    if not folder_limited:
                        if warning_level == 'high':
                            block = True
                        elif est_count > 9000:
                            block = True
                        elif 7500 <= est_count <= 9000 and est_conf in ('high','medium'):
                            block = True
                    # 安全速率自動阻擋：critical 強制；elevated 若同時達高文件風險也阻擋
                    if sec_rate_flag == 'critical':
                        block = True
                    elif sec_rate_flag == 'elevated' and not folder_limited and warning_level in ('high',) and not block:
                        block = True
                    if block:
                        block_msg = (
                            f"檢測到當前搜索範圍包含大量文件 (估算約 {est_count} 個，信心: {est_conf or '未知'})。\n"
                            "為確保性能與回答品質，本次查詢已被暫停。\n\n"
                            "請執行：\n"
                            "1. 勾選『限制搜索範圍』\n"
                            "2. 選擇較小或更聚焦的資料夾後再提問。\n\n"
                            "如確需大範圍檢索，請分批次縮小範圍進行。"
                        )
                        if sec_rate_flag == 'critical':
                            block_msg += "\n\n[安全自動阻擋] 最近出現大量可疑路徑事件，請確認路徑選擇或聯繫系統管理員。"
                        elif sec_rate_flag == 'elevated':
                            block_msg += "\n\n[安全提示] 最近路徑安全事件增加，建議縮小範圍並再次嘗試。"
                        return QuestionResponse(
                            answer=block_msg,
                            sources=[],
                            rewritten_query=None,
                            language=engine.get_language(),
                            file_count_warning=(local_scope_info.get('file_count_warning') or file_count_warning),
                            scope_info=local_scope_info
                        )
            except Exception as _e:
                logger.warning(f"取得動態範圍資訊失敗，跳過阻擋判斷: {_e}")

        # 根據是否使用查詢改寫選擇函數取得回答與文件
        if request.use_query_rewrite:
            answer, sources, documents, rewritten_query = engine.get_answer_with_query_rewrite(request.question)
        else:
            answer, sources, documents = engine.get_answer_with_sources(request.question)
            rewritten_query = None

        source_info_list = []
        if request.include_sources and documents:
            # 建立文件相關度映射 - 每個文件保留最高相關度的段落
            file_relevance_map = {}
            
            for doc in documents:
                metadata = doc.metadata
                file_path = metadata.get("file_path", "")
                score = metadata.get("score", float('inf'))  # 默認最低相關度
                
                if file_path:
                    # 如果文件還沒記錄，或者當前段落相關度更高（score更小）
                    if file_path not in file_relevance_map or score < file_relevance_map[file_path]['score']:
                        file_relevance_map[file_path] = {
                            'doc': doc,
                            'score': score,
                            'metadata': metadata
                        }
            
            # 按相關度排序文件（score越小越相關）
            sorted_files = sorted(file_relevance_map.items(), key=lambda x: x[1]['score'])
            
            # 限制來源數量，默認最多5筆
            max_sources = request.max_sources if request.max_sources is not None else 5
            
            # 處理最相關的文件
            for file_path, file_info in sorted_files[:max_sources]:
                doc = file_info['doc']
                metadata = file_info['metadata']
                score = file_info['score']
                
                # 提取位置信息
                location_info = None
                if "page_number" in metadata:
                    location_info = f"頁碼: {metadata['page_number']}"
                elif "block_number" in metadata:
                    location_info = f"塊: {metadata['block_number']}"
                elif "sheet_name" in metadata:
                    location_info = f"工作表: {metadata['sheet_name']}"
                
                # 暫時存儲，稍後批量生成相關性理由
                relevance_reason = None
                
                source_info = SourceInfo(
                    file_name=metadata.get("file_name", os.path.basename(file_path)),
                    file_path=file_path,
                    location_info=location_info,
                    relevance_reason=relevance_reason,
                    score=score if score != float('inf') else None
                )
                source_info_list.append(source_info)
            
            # 生成相關性理由
            if request.show_relevance and source_info_list:
                try:
                    # 統一使用批量生成方法
                    docs_for_relevance = []
                    for source_info in source_info_list:
                        file_path = source_info.file_path
                        if file_path in file_relevance_map:
                            docs_for_relevance.append(file_relevance_map[file_path]['doc'])

                    if docs_for_relevance:
                        batch_reasons = engine.generate_batch_relevance_reasons(request.question, docs_for_relevance)
                        
                        # 將生成的理由分配給對應的來源
                        for i, source_info in enumerate(source_info_list):
                            if i < len(batch_reasons):
                                source_info.relevance_reason = batch_reasons[i]
                            else:
                                # Fallback in case of mismatch
                                source_info.relevance_reason = f"相關文檔 {i+1}"
                
                except Exception as e:
                    logger.error(f"生成相關性理由時出錯: {str(e)}")
                    # 如果生成失敗，使用簡單的分數說明
                    for i, source_info in enumerate(source_info_list):
                        if source_info.score is not None:
                            if source_info.score < 0.5:
                                source_info.relevance_reason = "高度相關文檔"
                            elif source_info.score < 1.0:
                                source_info.relevance_reason = "相關文檔"
                            elif source_info.score < 1.5:
                                source_info.relevance_reason = "部分相關文檔"
                            else:
                                source_info.relevance_reason = "可能相關文檔"
                        else:
                            source_info.relevance_reason = f"相關文檔 {i+1}"
        
        return QuestionResponse(
            answer=answer,
            sources=source_info_list,
            rewritten_query=rewritten_query,
            language=engine.get_language(),
            file_count_warning=file_count_warning,
            scope_info=scope_info
        )
        
    except Exception as e:
        logger.error(f"處理問題時出錯: {str(e)}")
        raise HTTPException(status_code=500, detail=f"處理問題時出錯: {str(e)}")

@app.post("/api/dynamic/scope-info", response_model=ScopeInfoResponse)
async def get_dynamic_scope_info(scope_req: ScopeInfoRequest):
    """取得 Dynamic RAG 當前模型/範圍的檔案估算、警告與是否應阻擋查詢。

    提供前端在尚未送出問題前即可提示使用者縮小搜索範圍。
    """
    try:
        engine = get_rag_engine(
            selected_model=None,
            language=scope_req.language,
            use_dynamic_rag=True,
            ollama_model=scope_req.ollama_model,
            ollama_embedding_model=scope_req.ollama_embedding_model,
            platform=scope_req.platform,
            folder_path=scope_req.folder_path
        )

        scope_info = None
        file_count_warning = None
        if hasattr(engine, 'get_scope_info'):
            try:
                scope_info = engine.get_scope_info()
            except Exception as e:
                logger.warning(f"取得 scope_info 失敗: {e}")
        if hasattr(engine, 'get_file_count_warning'):
            try:
                file_count_warning = engine.get_file_count_warning()
            except Exception:
                pass

        if not scope_info:
            scope_info = {
                "folder_limited": bool(scope_req.folder_path),
                "estimated_file_count": None,
                "file_count_warning_level": None,
                "file_count_warning": file_count_warning,
            }

        warning_level = scope_info.get('file_count_warning_level')
        est_count = scope_info.get('estimated_file_count') or 0
        folder_limited = scope_info.get('folder_limited')
        est_conf = scope_info.get('estimation_confidence')

        block_recommended = False
        blocking_reason = None
        # 阻擋策略：
        # 1. 高警告且未限制 => 一律阻擋
        # 2. 估算超過 9000（高風險帶寬）且未限制 => 阻擋（無論信心）
        # 3. 估算 7500~9000 且信心為 high/medium 且未限制 => 阻擋
        if not folder_limited:
            if warning_level == 'high':
                block_recommended = True
            elif est_count > 9000:
                block_recommended = True
            elif 7500 <= est_count <= 9000 and est_conf in ('high','medium'):
                block_recommended = True
        # 安全速率自動阻擋
        sec_rate_flag = scope_info.get('security_rate_flag')
        if sec_rate_flag == 'critical':
            block_recommended = True
            blocking_reason = (blocking_reason or '') + "\n[安全自動阻擋] 檢測到大量可疑路徑事件，請檢查使用者選擇的路徑或系統日誌。"
        elif sec_rate_flag == 'elevated' and not block_recommended:
            blocking_reason = (blocking_reason or '') + "\n[安全提示] 最近路徑安全事件增加，建議確認搜尋範圍。"

        if block_recommended and not blocking_reason:
            blocking_reason = (
                f"檢測到當前搜索範圍包含大量文件 (估算約 {est_count} 個，信心: {est_conf or '未知'})。\n"
                "為確保性能與品質，請勾選『限制搜索範圍』並選擇較小的資料夾後再提問。"
            )

        return ScopeInfoResponse(
            scope_info=scope_info,
            file_count_warning=file_count_warning or scope_info.get('file_count_warning'),
            block_recommended=block_recommended,
            blocking_reason=blocking_reason
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"取得動態範圍資訊失敗: {e}")
        raise HTTPException(status_code=500, detail=f"取得動態範圍資訊失敗: {e}")


@app.get("/api/dynamic/estimation-stats")
async def get_dynamic_estimation_stats(limit: int = Query(200, ge=10, le=2000), include_samples: bool = Query(True)):
    """取得文件數量估算的近期統計，用於校準與觀察估算器表現。

    來源: logs/estimation_audit.log 由 dynamic_rag_base 寫入的行。
    每行格式: ts=... path=... est=... actual=... err_pct=... conf=... sampled=... total_dirs=... method=... truncated=...
    """
    try:
        from config.config import LOGS_DIR
        log_path = Path(LOGS_DIR) / 'estimation_audit.log'
        if not log_path.exists():
            return {
                "total_samples": 0,
                "limit": limit,
                "mean_error_pct": None,
                "mae_pct": None,
                "mape_pct": None,
                "mean_signed_error_pct": None,
                "overestimate_rate": None,
                "underestimate_rate": None,
                "confidence_stats": {},
                "recent_samples": [] if include_samples else None
            }

        # 高效讀取檔案尾部最後 N 行
        lines = deque(maxlen=limit)
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if line.strip():
                    lines.append(line.strip())

        samples = []
        for line in lines:
            parts = line.split()
            record = {}
            for p in parts:
                if '=' not in p:
                    continue
                k, v = p.split('=', 1)
                record[k] = v
            try:
                est = int(record.get('est')) if record.get('est') and record.get('est').isdigit() else None
                actual = int(record.get('actual')) if record.get('actual') and record.get('actual').isdigit() else None
                err_pct_raw = record.get('err_pct')
                try:
                    err_pct = float(err_pct_raw) if err_pct_raw not in (None, 'None') else None
                except ValueError:
                    err_pct = None
                samples.append({
                    'ts': int(record.get('ts')) if record.get('ts') and record.get('ts').isdigit() else None,
                    'est': est,
                    'actual': actual,
                    'err_pct': err_pct,
                    'conf': record.get('conf'),
                    'method': record.get('method'),
                    'truncated': record.get('truncated') in ('True', 'true', '1'),
                })
            except Exception:
                continue

        valid = [s for s in samples if s.get('err_pct') is not None]
        total = len(valid)
        if total == 0:
            return {
                "total_samples": 0,
                "limit": limit,
                "mean_error_pct": None,
                "mae_pct": None,
                "mape_pct": None,
                "mean_signed_error_pct": None,
                "overestimate_rate": None,
                "underestimate_rate": None,
                "confidence_stats": {},
                "recent_samples": samples if include_samples else None
            }

        # 統計計算
        import math
        signed_errors = [s['err_pct'] for s in valid]
        abs_errors = [abs(e) for e in signed_errors]
        mean_signed = sum(signed_errors) / total
        mae = sum(abs_errors) / total
        # MAPE: 使用 actual 值再計算一次 (避免使用已記錄 err_pct 因其已是百分比)
        ape_values = []
        for s in valid:
            est = s['est']
            actual = s['actual']
            if est is not None and actual and actual > 0:
                ape_values.append(abs(est - actual) / actual * 100.0)
        mape = sum(ape_values) / len(ape_values) if ape_values else None
        mean_error_pct = sum(signed_errors) / total  # 與 mean_signed 同義，保留兩個欄位語意清晰

        over_rate = sum(1 for e in signed_errors if e > 0) / total
        under_rate = sum(1 for e in signed_errors if e < 0) / total

        # 信心分組統計
        conf_groups: Dict[str, Dict[str, Any]] = {}
        for s in valid:
            c = s.get('conf') or 'unknown'
            conf_groups.setdefault(c, {"count": 0, "sum_signed": 0.0, "sum_abs": 0.0})
            conf_groups[c]["count"] += 1
            conf_groups[c]["sum_signed"] += s['err_pct']
            conf_groups[c]["sum_abs"] += abs(s['err_pct'])
        confidence_stats = {}
        for c, g in conf_groups.items():
            count = g['count']
            confidence_stats[c] = {
                "count": count,
                "mean_error_pct": round(g['sum_signed'] / count, 2) if count else None,
                "mae_pct": round(g['sum_abs'] / count, 2) if count else None,
                "bias_direction": 'over' if g['sum_signed'] / count > 0 else ('under' if g['sum_signed'] / count < 0 else 'neutral')
            }

        response = {
            "total_samples": total,
            "limit": limit,
            "mean_error_pct": round(mean_error_pct, 2),
            "mae_pct": round(mae, 2),
            "mape_pct": round(mape, 2) if mape is not None else None,
            "mean_signed_error_pct": round(mean_signed, 2),
            "overestimate_rate": round(over_rate, 3),
            "underestimate_rate": round(under_rate, 3),
            "confidence_stats": confidence_stats,
            "recent_samples": samples if include_samples else None
        }
        return response
    except Exception as e:
        logger.error(f"取得估算統計失敗: {e}")
        raise HTTPException(status_code=500, detail=f"取得估算統計失敗: {e}")

@app.get("/api/dynamic/security-events")
async def get_dynamic_security_events(limit: int = Query(50, ge=1, le=200)):
    """取得最近的動態RAG資料夾路徑安全事件（越界 / 解析失敗）。"""
    try:
        from rag_engine.dynamic_rag_base import SECURITY_EVENTS
        events = SECURITY_EVENTS[-limit:]
        return {"count": len(events), "limit": limit, "events": events}
    except Exception as e:
        logger.error(f"取得安全事件失敗: {e}")
        raise HTTPException(status_code=500, detail=f"取得安全事件失敗: {e}")

@app.get("/api/dynamic/security-alerts")
async def get_dynamic_security_alerts(limit: int = Query(50, ge=1, le=500)):
    """讀取 security_alerts.log 最近的警報行 (語義：速率達 elevated/critical 時寫入)。"""
    try:
        from config.config import LOGS_DIR
        log_path = Path(LOGS_DIR) / 'security_alerts.log'
        if not log_path.exists():
            return {"count": 0, "limit": limit, "alerts": []}
        lines = []
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if line.strip():
                    lines.append(line.strip())
        sliced = lines[-limit:]
        # 解析成結構 (ts=xxx level=yyy ...)
        parsed = []
        for ln in sliced:
            parts = ln.split()
            rec = {"raw": ln}
            for p in parts:
                if '=' in p:
                    k,v = p.split('=',1)
                    rec[k]=v
            parsed.append(rec)
        return {"count": len(parsed), "limit": limit, "alerts": parsed}
    except Exception as e:
        logger.error(f"讀取 security_alerts 失敗: {e}")
        raise HTTPException(status_code=500, detail=f"讀取 security_alerts 失敗: {e}")




@app.get("/admin/logs")
async def list_logs(request: Request):
    """列出所有可用的日誌文件"""
    await check_admin(request)
    log_dir = Path(LOGS_DIR)
    if not log_dir.exists():
        return []
    
    log_files = [f.name for f in log_dir.iterdir() if f.is_file() and f.name.endswith(".log")]
    return sorted(log_files, reverse=True)

@app.get("/admin/download_log")
async def download_log(request: Request, filename: str = Query(...)):
    """下載指定的日誌文件"""
    await check_admin(request)
    
    # 安全性檢查：防止路徑遍歷
    if '/' in filename or '\\' in filename:
        raise HTTPException(status_code=400, detail="無效的文件名")
        
    log_file = Path(LOGS_DIR) / filename
    
    if not log_file.exists() or not log_file.is_file():
        return PlainTextResponse("(尚無日誌)", status_code=404)
        
    return FileResponse(str(log_file), filename=log_file.name, media_type='text/plain')




@app.get("/admin/monitor_status")
async def monitor_status(request: Request):
    await check_admin(request)
    return PlainTextResponse(get_status_text())

@app.get("/admin/monitor_progress")
async def monitor_progress(request: Request):
    await check_admin(request)
    return PlainTextResponse(get_progress_text())

@app.get("/admin/monitor_realtime")
async def monitor_realtime(request: Request):
    await check_admin(request)
    return PlainTextResponse(get_monitor_text(once=True))

@app.get("/admin/monitor_all")
async def monitor_all(request: Request, model_folder_name: Optional[str] = Query(None)):
    await check_admin(request)
    indexing_status = get_indexing_status(model_folder_name)
    return {
        "status": get_status_text(model_folder_name),
        "progress": get_progress_text(model_folder_name),
        "realtime": get_monitor_text(model_folder_name, once=True),
        "training_model": indexing_status.get('training_model_name')
    }

# 新增：內部使用的API
@app.get("/api/internal/get-model-folder-name")
async def get_model_folder_name_endpoint(ollama_model: str, ollama_embedding_model: str, version: Optional[str] = None):
    """根據模型組件生成標準化的文件夾名稱"""
    folder_name = vector_db_manager.get_model_folder_name(ollama_model, ollama_embedding_model, version)
    return {"folder_name": folder_name}

# 新增：模型管理相關 API
@app.get("/api/ollama/models", response_model=List[ModelInfo])
async def get_ollama_models():
    """獲取 Ollama 可用模型列表"""
    try:
        models = ollama_utils.get_available_models()
        return [ModelInfo(
            name=model['name'],
            size=model['size'],
            modified=model['modified']
        ) for model in models]
    except Exception as e:
        logger.error(f"獲取 Ollama 模型列表失敗: {str(e)}")
        raise HTTPException(status_code=500, detail=f"獲取模型列表失敗: {str(e)}")

@app.get("/api/ollama/models/categorized")
async def get_ollama_models_categorized():
    """獲取分類的 Ollama 模型列表（用於 Dynamic RAG）"""
    try:
        models = ollama_utils.get_available_models()
        
        # 分類模型
        language_models = []
        embedding_models = []
        
        for model in models:
            model_name = model['name'].lower()
            
            # 根據模型名稱判斷類型
            if any(embed_keyword in model_name for embed_keyword in ['embed', 'embedding', 'nomic']):
                embedding_models.append(model['name'])
            else:
                language_models.append(model['name'])
        
        return {
            "language_models": language_models,
            "embedding_models": embedding_models
        }
    except Exception as e:
        logger.error(f"獲取分類 Ollama 模型列表失敗: {str(e)}")
        raise HTTPException(status_code=500, detail=f"獲取模型列表失敗: {str(e)}")

@app.get("/api/vector-models", response_model=List[VectorModelInfo])
async def get_vector_models():
    """獲取向量數據庫模型列表"""
    try:
        models = vector_db_manager.list_available_models()
        response_models = []
        for model in models:
            version = model.get('version')
            display_name = model['display_name']
            
            # 如果沒有版本號，則在顯示名稱中明確標示
            if not version or version == "current":
                version = "current"
                if "(當前版本)" not in display_name:
                    display_name += " (當前版本)"

            response_models.append(VectorModelInfo(
                folder_name=model['folder_name'],
                display_name=display_name,
                version=version,
                model_info=model['model_info'],
                is_training=model['is_training'],
                has_data=model['has_data']
            ))
        return response_models
    except Exception as e:
        logger.error(f"獲取向量模型列表失敗: {str(e)}")
        raise HTTPException(status_code=500, detail=f"獲取向量模型列表失敗: {str(e)}")

@app.get("/api/usable-models", response_model=List[VectorModelInfo])
async def get_usable_models():
    """獲取可用於問答的模型列表（有數據且未在訓練中）"""
    try:
        models = vector_db_manager.get_usable_models()
        response_models = []
        for model in models:
            version = model.get('version')
            display_name = model['display_name']
            
            # 如果沒有版本號，則在顯示名稱中明確標示
            if not version or version == "current":
                version = "current"
                if "(當前版本)" not in display_name:
                    display_name += " (當前版本)"

            response_models.append(VectorModelInfo(
                folder_name=model['folder_name'],
                display_name=display_name,
                version=version,
                model_info=model['model_info'],
                is_training=model['is_training'],
                has_data=model['has_data']
            ))
        return response_models
    except Exception as e:
        logger.error(f"獲取可用模型列表失敗: {str(e)}")
        raise HTTPException(status_code=500, detail=f"獲取可用模型列表失敗: {str(e)}")

def _get_folder_validation_results(folder_path: str) -> dict:
    """
    驗證文件夾路徑並估算文件數量，返回一個包含結果的字典。
    """
    try:
        from config.config import Q_DRIVE_PATH, SUPPORTED_FILE_TYPES
        
        if not folder_path:
            return {"exists": False, "message": "未提供文件夾路徑"}

        folder_path_clean = folder_path.strip('/\\')
        # 安全路徑解析與越界檢測
        base_root = Path(Q_DRIVE_PATH).resolve()
        candidate = (base_root / folder_path_clean).resolve()
        if not str(candidate).startswith(str(base_root)):
            return {"exists": False, "message": "指定的文件夾越界 (非法路徑)"}
        full_path = str(candidate)
        
        if not (os.path.exists(full_path) and os.path.isdir(full_path)):
            return {"exists": False, "message": "指定的文件夾不存在"}

        estimated_count = 0
        actual_count = 0
        sampled_dirs = 0
        max_sample_dirs = 10
        max_quick_count = 500

        try:
            first_level_files = [f for f in os.listdir(full_path) 
                               if os.path.isfile(os.path.join(full_path, f)) 
                               and os.path.splitext(f)[1].lower() in SUPPORTED_FILE_TYPES]
            
            sample_total = 0
            total_dirs_count = 0
            
            for root, dirs, files in os.walk(full_path):
                depth = root.replace(full_path, '').count(os.sep)
                if depth > 3:
                    dirs[:] = []
                    continue
                
                total_dirs_count += 1
                
                if sampled_dirs < max_sample_dirs:
                    supported_files = [f for f in files if os.path.splitext(f)[1].lower() in SUPPORTED_FILE_TYPES]
                    sample_total += len(supported_files)
                    sampled_dirs += 1
                    
                    if sampled_dirs <= 3:
                        actual_count += len(supported_files)
                
                if sampled_dirs >= 3 and sample_total <= max_quick_count:
                    for f in files:
                        if os.path.splitext(f)[1].lower() in SUPPORTED_FILE_TYPES:
                            actual_count += 1
                
                dirs[:] = [d for d in dirs if not d.startswith('.') and d.lower() not in ['system', 'temp', 'tmp']]
            
            if actual_count > 0 and sample_total <= max_quick_count:
                file_count = actual_count
                count_type = "精確"
            elif sampled_dirs > 0:
                avg_files_per_dir = sample_total / sampled_dirs
                estimated_count = int(avg_files_per_dir * total_dirs_count)
                file_count = estimated_count
                count_type = "估算"
            else:
                file_count = len(first_level_files)
                count_type = "根目錄"
            
        except Exception as e:
            logger.warning(f"計算文件數量時出錯: {str(e)}")
            file_count = 0
            count_type = "未知"
        
        suggestion = ""
        warning_level = "low"
        if file_count > 5000:
            suggestion = f"檢測到處理文件數量過多 ({file_count} 個)，可能影響處理速度。建議縮小搜索範圍。"
            warning_level = "high"
        elif file_count > 1000:
            suggestion = "文件數量適中，搜索效果應該不錯"
            warning_level = "medium"
        else:
            suggestion = "文件數量較少，搜索效果會很好"
            warning_level = "low"

        return {
            "exists": True,
            "full_path": full_path,
            "file_count": file_count,
            "count_type": count_type,
            "warning_level": warning_level,
            "suggestion": suggestion,
            "message": f"文件夾存在，{count_type}包含約 {file_count} 個支持的文件"
        }

    except Exception as e:
        logger.error(f"驗證文件夾路徑失敗: {str(e)}")
        return {"exists": False, "message": f"驗證過程中發生錯誤: {e}"}

@app.get("/api/validate-folder")
async def validate_folder_path(folder_path: str = Query(...)):
    """驗證文件夾路徑是否存在並快速估算文件數量"""
    try:
        return _get_folder_validation_results(folder_path)
    except Exception as e:
        logger.error(f"驗證文件夾路徑失敗: {str(e)}")
        raise HTTPException(status_code=500, detail=f"驗證文件夾路徑失敗: {str(e)}")


@app.get("/api/model-versions", response_model=List[Dict[str, Any]])
async def get_model_versions(ollama_model: str = Query(...), ollama_embedding_model: str = Query(...)):
    """獲取指定模型組合的所有版本"""
    try:
        versions = vector_db_manager.get_model_versions(ollama_model, ollama_embedding_model)
        return versions
    except Exception as e:
        logger.error(f"獲取模型版本列表失敗: {str(e)}")
        raise HTTPException(status_code=500, detail=f"獲取模型版本列表失敗: {str(e)}")

@app.post("/admin/training/initial")
async def start_initial_training(request: Request, training_request: TrainingRequest):
    """開始初始訓練"""
    await check_admin(request)
    try:
        # 獲取模型文件夾路徑
        folder_name = vector_db_manager.get_model_folder_name(
            training_request.ollama_model,
            training_request.ollama_embedding_model,
            training_request.version
        )
        model_path = vector_db_manager.base_path / folder_name
        model_path.mkdir(parents=True, exist_ok=True)

        # 立即創建鎖定文件以反映狀態
        process_info = {"type": "initial", "status": "starting", "api_pid": os.getpid()}
        vector_db_manager.create_lock_file(model_path, process_info)

        # 先將模型資訊寫入暫存檔案
        training_info = training_request.dict()
        with open("temp_training_info.json", "w") as f:
            json.dump(training_info, f)
        
        cmd = [sys.executable, "scripts/model_training_manager.py", "initial"]
        process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        return {"status": "初始訓練已開始", "message": f"訓練進程已啟動 (PID: {process.pid})"}
        
    except Exception as e:
        logger.error(f"開始初始訓練失敗: {str(e)}")
        if 'model_path' in locals() and vector_db_manager.is_training(model_path):
            vector_db_manager.remove_lock_file(model_path)
        raise HTTPException(status_code=500, detail=f"開始訓練失敗: {str(e)}")

@app.post("/admin/training/incremental")
async def start_incremental_training(request: Request, training_request: TrainingRequest):
    """開始增量訓練"""
    await check_admin(request)
    try:
        # 獲取模型文件夾路徑
        folder_name = vector_db_manager.get_model_folder_name(
            training_request.ollama_model,
            training_request.ollama_embedding_model,
            training_request.version
        )
        model_path = vector_db_manager.base_path / folder_name

        # 立即創建鎖定文件
        process_info = {"type": "incremental", "status": "starting", "api_pid": os.getpid()}
        vector_db_manager.create_lock_file(model_path, process_info)

        training_info = training_request.dict()
        with open("temp_training_info.json", "w") as f:
            json.dump(training_info, f)
        
        cmd = [sys.executable, "scripts/model_training_manager.py", "incremental"]
        process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        return {"status": "增量訓練已開始", "message": f"訓練進程已啟動 (PID: {process.pid})"}
        
    except Exception as e:
        logger.error(f"開始增量訓練失敗: {str(e)}")
        if 'model_path' in locals() and vector_db_manager.is_training(model_path):
            vector_db_manager.remove_lock_file(model_path)
        raise HTTPException(status_code=500, detail=f"開始增量訓練失敗: {str(e)}")

@app.post("/admin/training/reindex")
async def start_reindex_training(request: Request, training_request: TrainingRequest):
    """開始重新索引"""
    await check_admin(request)
    try:
        # 獲取模型文件夾路徑
        folder_name = vector_db_manager.get_model_folder_name(
            training_request.ollama_model,
            training_request.ollama_embedding_model,
            training_request.version
        )
        model_path = vector_db_manager.base_path / folder_name

        # 立即創建鎖定文件
        process_info = {"type": "reindex", "status": "starting", "api_pid": os.getpid()}
        vector_db_manager.create_lock_file(model_path, process_info)

        training_info = training_request.dict()
        with open("temp_training_info.json", "w") as f:
            json.dump(training_info, f)
        
        cmd = [sys.executable, "scripts/model_training_manager.py", "reindex"]
        process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        return {"status": "重新索引已開始", "message": f"訓練進程已啟動 (PID: {process.pid})"}
        
    except Exception as e:
        logger.error(f"開始重新索引失敗: {str(e)}")
        if 'model_path' in locals() and vector_db_manager.is_training(model_path):
            vector_db_manager.remove_lock_file(model_path)
        raise HTTPException(status_code=500, detail=f"開始重新索引失敗: {str(e)}")

@app.get("/admin/lock-status")
async def get_lock_status(request: Request):
    """獲取所有模型的鎖定狀態"""
    await check_admin(request)
    try:
        models = vector_db_manager.list_available_models()
        lock_status = []
        
        for model in models:
            model_path = Path(model['folder_path'])
            is_locked = vector_db_manager.is_training(model_path)
            
            if is_locked:
                is_valid, reason = vector_db_manager.is_lock_valid(model_path)
                lock_info = vector_db_manager.get_lock_info(model_path)
            else:
                is_valid, reason = True, "沒有鎖定"
                lock_info = None
            
            lock_status.append({
                "model_name": model['display_name'],
                "folder_name": model['folder_name'],
                "is_locked": is_locked,
                "is_lock_valid": is_valid,
                "lock_reason": reason,
                "lock_info": lock_info,
                "has_data": model['has_data'],
                "can_use": model['has_data'] and (not is_locked or not is_valid)
            })
        
        return lock_status
    except Exception as e:
        logger.error(f"獲取鎖定狀態失敗: {str(e)}")
        raise HTTPException(status_code=500, detail=f"獲取鎖定狀態失敗: {str(e)}")

@app.post("/admin/force-unlock")
async def force_unlock_model(request: Request, folder_name: str = Body(...), reason: str = Body("管理員手動解鎖")):
    """強制解鎖指定模型"""
    await check_admin(request)
    try:
        models = vector_db_manager.list_available_models()
        target_model = None
        
        for model in models:
            if model['folder_name'] == folder_name:
                target_model = model
                break
        
        if not target_model:
            raise HTTPException(status_code=404, detail=f"找不到模型: {folder_name}")
        
        model_path = Path(target_model['folder_path'])
        
        if not vector_db_manager.is_training(model_path):
            return {"status": "success", "message": "模型沒有被鎖定"}
        
        success = vector_db_manager.force_unlock_model(model_path, reason)
        
        if success:
            # 清理對應的RAG引擎緩存
            if folder_name in rag_engines:
                del rag_engines[folder_name]
            if 'auto_selected' in rag_engines:
                del rag_engines['auto_selected']
            
            return {"status": "success", "message": f"成功解鎖模型: {target_model['display_name']}"}
        else:
            return {"status": "error", "message": "解鎖失敗"}
            
    except Exception as e:
        logger.error(f"強制解鎖失敗: {str(e)}")
        raise HTTPException(status_code=500, detail=f"強制解鎖失敗: {str(e)}")

@app.post("/admin/cleanup-invalid-locks")
async def cleanup_invalid_locks(request: Request):
    """清理所有無效的鎖定文件"""
    await check_admin(request)
    try:
        from utils.training_lock_manager import training_lock_manager
        results = training_lock_manager.cleanup_invalid_locks(vector_db_manager.base_path)
        
        # 清理RAG引擎緩存
        global rag_engines
        rag_engines.clear()
        
        return {"status": "success", "results": results}
    except Exception as e:
        logger.error(f"清理無效鎖定失敗: {str(e)}")
        raise HTTPException(status_code=500, detail=f"清理失敗: {str(e)}")

# === 新的設置流程 API ===

@app.get("/api/setup/status")
async def get_setup_status():
    """獲取設置狀態"""
    setup_manager = get_setup_flow_manager()
    return {
        "setup_completed": setup_manager.is_setup_completed(),
        "current_step": setup_manager.get_current_step(),
        "progress": setup_manager.get_setup_progress(),
        "configuration": setup_manager.get_current_configuration()
    }

@app.get("/api/setup/platforms")
async def get_available_platforms():
    """獲取可用平台列表"""
    setup_manager = get_setup_flow_manager()
    return setup_manager.get_platform_selection_data()

@app.post("/api/setup/platform")
async def set_platform(request: PlatformSelectionRequest):
    """設置平台選擇"""
    setup_manager = get_setup_flow_manager()
    return setup_manager.set_platform(request.platform_type)

@app.get("/api/setup/models")
async def get_available_models():
    """獲取可用模型列表"""
    setup_manager = get_setup_flow_manager()
    return setup_manager.get_model_selection_data()

@app.post("/api/setup/models")
async def set_models(request: ModelSelectionRequest):
    """設置模型選擇"""
    setup_manager = get_setup_flow_manager()
    return setup_manager.set_models(request.language_model, request.embedding_model, request.inference_engine)

@app.get("/api/setup/rag-modes")
async def get_rag_modes():
    """獲取 RAG 模式選項"""
    setup_manager = get_setup_flow_manager()
    return setup_manager.get_rag_mode_selection_data()

@app.post("/api/setup/rag-mode")
async def set_rag_mode(request: RAGModeSelectionRequest):
    """設置 RAG 模式"""
    setup_manager = get_setup_flow_manager()
    return setup_manager.set_rag_mode(request.rag_mode)

@app.get("/api/setup/review")
async def get_configuration_review():
    """獲取配置審查數據"""
    setup_manager = get_setup_flow_manager()
    return setup_manager.get_configuration_review_data()

@app.post("/api/setup/complete")
async def complete_setup():
    """完成設置"""
    setup_manager = get_setup_flow_manager()
    return setup_manager.complete_setup()

# === 文件夾選擇 API ===
@app.get("/api/folders")
async def get_folder_list(path: Optional[str] = Query(None)):
    """獲取指定路徑下的文件夾列表，支持多層級導航和詳細統計"""
    try:
        from config.config import Q_DRIVE_PATH, get_supported_file_extensions
        
        # 如果沒有指定路徑，使用 Q 槽根目錄
        if not path:
            base_path = Path(Q_DRIVE_PATH)
        else:
            # 確保路徑在 Q 槽範圍內（安全檢查）
            requested_path = Path(Q_DRIVE_PATH) / path.lstrip('/')
            if not str(requested_path).startswith(str(Path(Q_DRIVE_PATH))):
                raise HTTPException(status_code=400, detail="路徑不在允許範圍內")
            base_path = requested_path
        
        if not base_path.exists():
            raise HTTPException(status_code=404, detail="路徑不存在")
        
        folders = []
        files_count = 0
        total_size = 0
        supported_extensions = get_supported_file_extensions()
        
        def calculate_folder_stats_shallow(folder_path: Path):
            """計算文件夾統計信息（淺層掃描，避免大文件夾超時）"""
            # 從環境變量獲取配置
            max_files = int(os.getenv("FOLDER_SCAN_MAX_FILES", "50"))
            max_depth = int(os.getenv("FOLDER_SCAN_MAX_DEPTH", "1"))
            
            file_count = 0
            total_size = 0
            
            def scan_directory(dir_path: Path, current_depth: int = 0):
                nonlocal file_count, total_size
                
                if current_depth > max_depth or file_count >= max_files:
                    return
                
                try:
                    # 只掃描當前目錄，不遞歸太深
                    for item in dir_path.iterdir():
                        if file_count >= max_files:
                            break
                            
                        if item.is_file() and item.suffix.lower() in supported_extensions:
                            file_count += 1
                            try:
                                total_size += item.stat().st_size
                            except (OSError, PermissionError):
                                pass
                        elif item.is_dir() and current_depth < max_depth:
                            # 只遞歸到指定深度
                            scan_directory(item, current_depth + 1)
                            
                except (PermissionError, OSError):
                    pass
            
            scan_directory(folder_path)
            return file_count, total_size
        
        def get_immediate_folder_info(folder_path: Path):
            """獲取文件夾的直接信息（不遞歸，最快速）"""
            # 從環境變量獲取配置
            max_files = int(os.getenv("FOLDER_SCAN_MAX_FILES", "50"))
            
            file_count = 0
            has_subfolders = False
            
            try:
                for item in folder_path.iterdir():
                    if item.is_file() and item.suffix.lower() in supported_extensions:
                        file_count += 1
                        if file_count >= max_files:  # 使用配置的最大文件數
                            break
                    elif item.is_dir():
                        has_subfolders = True
                        if file_count >= max_files // 2 and has_subfolders:  # 有文件又有子文件夾，快速返回
                            break
            except (PermissionError, OSError):
                pass
            
            return file_count, has_subfolders
        
        try:
            # 獲取文件夾列表和統計（使用快速掃描避免超時）
            for item in base_path.iterdir():
                if item.is_dir():
                    # 使用快速掃描獲取文件夾信息
                    immediate_files, has_subfolders = get_immediate_folder_info(item)
                    
                    relative_path = str(item.relative_to(Path(Q_DRIVE_PATH)))
                    
                    # 根據文件夾大小決定掃描策略
                    if immediate_files > 20 or has_subfolders:
                        # 大文件夾：使用估算值，避免精確計算導致超時
                        display_count = f"{immediate_files}+" if immediate_files >= 50 else str(immediate_files)
                        folders.append({
                            "name": item.name,
                            "path": relative_path,
                            "files_count": immediate_files,
                            "files_count_display": display_count,
                            "total_size": 0,  # 大文件夾不計算總大小
                            "size_mb": 0,
                            "is_folder": True,
                            "is_large_folder": True,
                            "has_subfolders": has_subfolders
                        })
                    else:
                        # 小文件夾：使用精確計算
                        folder_files_count, folder_size = calculate_folder_stats_shallow(item)
                        folders.append({
                            "name": item.name,
                            "path": relative_path,
                            "files_count": folder_files_count,
                            "files_count_display": str(folder_files_count),
                            "total_size": folder_size,
                            "size_mb": round(folder_size / (1024 * 1024), 2),
                            "is_folder": True,
                            "is_large_folder": False,
                            "has_subfolders": has_subfolders
                        })
                        
                elif item.is_file() and item.suffix.lower() in supported_extensions:
                    files_count += 1
                    try:
                        total_size += item.stat().st_size
                    except (OSError, PermissionError):
                        pass
        
        except (PermissionError, OSError) as e:
            logger.warning(f"無法訪問路徑 {base_path}: {e}")
            raise HTTPException(status_code=403, detail="無權限訪問該路徑")
        
        # 按文件夾名稱排序
        folders.sort(key=lambda x: x["name"].lower())
        
        # 構建路徑導航
        path_parts = []
        if base_path != Path(Q_DRIVE_PATH):
            current_path = base_path.relative_to(Path(Q_DRIVE_PATH))
            parts = current_path.parts
            for i, part in enumerate(parts):
                path_parts.append({
                    "name": part,
                    "path": str(Path(*parts[:i+1])) if i > 0 else part
                })
        
        return {
            "current_path": str(base_path.relative_to(Path(Q_DRIVE_PATH))) if base_path != Path(Q_DRIVE_PATH) else "",
            "parent_path": str(base_path.parent.relative_to(Path(Q_DRIVE_PATH))) if base_path.parent != base_path and base_path != Path(Q_DRIVE_PATH) else None,
            "path_parts": path_parts,
            "folders": folders,
            "files_count": files_count,
            "total_size": total_size,
            "size_mb": round(total_size / (1024 * 1024), 2),
            "total_folders": len(folders)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"獲取文件夾列表失敗: {str(e)}")
        raise HTTPException(status_code=500, detail=f"獲取文件夾列表失敗: {str(e)}")

@app.post("/api/setup/reset")
async def reset_setup():
    """重置設置"""
    setup_manager = get_setup_flow_manager()
    return setup_manager.reset_setup()

# 重複的路由定義已移除 - 原始定義在第780行

# 向量資料庫維護 API
@app.get("/admin/vector-db/info")
async def get_vector_db_info(request: Request, folder_name: str = Query(...)):
    """獲取指定向量資料庫的詳細信息"""
    await check_admin(request)
    try:
        models = vector_db_manager.list_available_models()
        target_model = None
        
        for model in models:
            if model['folder_name'] == folder_name:
                target_model = model
                break
        
        if not target_model:
            raise HTTPException(status_code=404, detail=f"找不到模型: {folder_name}")
        
        model_path = Path(target_model['folder_path'])
        
        # 收集詳細信息
        info = {
            "folder_name": folder_name,
            "folder_path": str(model_path),
            "model_info": target_model['model_info'],
            "has_data": target_model['has_data'],
            "is_training": target_model['is_training'],
            "version": target_model.get('version'),
            "display_name": target_model['display_name']
        }
        
        # 文件系統信息
        if model_path.exists():
            import os
            folder_size = sum(f.stat().st_size for f in model_path.rglob('*') if f.is_file())
            file_count = len([f for f in model_path.rglob('*') if f.is_file()])
            
            info["filesystem"] = {
                "folder_size_bytes": folder_size,
                "folder_size_mb": round(folder_size / (1024 * 1024), 2),
                "file_count": file_count,
                "created_time": os.path.getctime(model_path),
                "modified_time": os.path.getmtime(model_path)
            }
            
            # 檢查關鍵文件
            key_files = {
                "index.faiss": (model_path / "index.faiss").exists(),
                "index.pkl": (model_path / "index.pkl").exists(),
                ".model": (model_path / ".model").exists(),
                ".lock": (model_path / ".lock").exists()
            }
            info["key_files"] = key_files
            
            # 如果有向量數據，嘗試獲取更多信息
            if target_model['has_data']:
                try:
                    info["vector_stats"] = {
                        "status": "有向量數據",
                        "note": "詳細統計需要載入向量數據庫"
                    }
                except Exception as e:
                    info["vector_stats"] = {
                        "status": "無法獲取向量統計",
                        "error": str(e)
                    }
        
        return info
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"獲取向量資料庫信息失敗: {str(e)}")
        raise HTTPException(status_code=500, detail=f"獲取信息失敗: {str(e)}")

@app.post("/admin/vector-db/backup")
async def backup_vector_db(request: Request, backup_request: dict = Body(...)):
    """備份指定的向量資料庫"""
    await check_admin(request)
    try:
        folder_name = backup_request.get("folder_name")
        if not folder_name:
            raise HTTPException(status_code=400, detail="缺少 folder_name 參數")

        models = vector_db_manager.list_available_models()
        target_model = None
        for m in models:
            if m['folder_name'] == folder_name:
                target_model = m
                break
        if not target_model:
            raise HTTPException(status_code=404, detail=f"找不到模型: {folder_name}")
        model_path = Path(target_model['folder_path'])
        if not model_path.exists():
            raise HTTPException(status_code=404, detail=f"模型路徑不存在: {model_path}")
        # 創建備份
        import shutil
        import datetime
        
        backup_dir = Path(BACKUPS_DIR)
        backup_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{folder_name}_backup_{timestamp}"
        backup_path = backup_dir / backup_name
        
        # 複製整個資料夾
        shutil.copytree(model_path, backup_path)
        
        # 創建備份信息文件
        backup_info = {
            "original_folder": folder_name,
            "original_path": str(model_path),
            "backup_time": datetime.datetime.now().isoformat(),
            "model_info": target_model['model_info'],
            "backup_size_bytes": sum(f.stat().st_size for f in backup_path.rglob('*') if f.is_file())
        }
        
        with open(backup_path / ".backup_info", 'w', encoding='utf-8') as f:
            json.dump(backup_info, f, ensure_ascii=False, indent=2)
        
        logger.info(f"成功備份向量資料庫: {folder_name} -> {backup_path}")
        
        return {
            "status": "success",
            "backup_path": str(backup_path),
            "backup_name": backup_name,
            "backup_size_mb": round(backup_info["backup_size_bytes"] / (1024 * 1024), 2)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"備份向量資料庫失敗: {str(e)}")
        raise HTTPException(status_code=500, detail=f"備份失敗: {str(e)}")

@app.delete("/admin/vector-db/delete")
async def delete_vector_db(request: Request, delete_request: dict = Body(...)):
    """刪除指定的向量資料庫"""
    await check_admin(request)
    try:
        folder_name = delete_request.get("folder_name")
        if not folder_name:
            raise HTTPException(status_code=400, detail="缺少 folder_name 參數")
        
        models = vector_db_manager.list_available_models()
        target_model = None
        
        for model in models:
            if model['folder_name'] == folder_name:
                target_model = model
                break
        
        if not target_model:
            raise HTTPException(status_code=404, detail=f"找不到模型: {folder_name}")
        
        # 檢查是否正在訓練
        if target_model['is_training']:
            is_valid, reason = vector_db_manager.is_lock_valid(Path(target_model['folder_path']))
            if is_valid:
                raise HTTPException(status_code=423, detail=f"模型正在訓練中，無法刪除: {reason}")
        
        model_path = Path(target_model['folder_path'])
        
        if not model_path.exists():
            return {"status": "success", "message": "模型資料夾已不存在"}
        
        # 刪除資料夾
        import shutil
        shutil.rmtree(model_path)
        
        # 清理相關的RAG引擎緩存
        global rag_engines
        keys_to_remove = [key for key in rag_engines.keys() if folder_name in key]
        for key in keys_to_remove:
            del rag_engines[key]
        
        logger.info(f"成功刪除向量資料庫: {folder_name}")
        
        return {
            "status": "success",
            "message": f"成功刪除模型: {target_model['display_name']}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"刪除向量資料庫失敗: {str(e)}")
        raise HTTPException(status_code=500, detail=f"刪除失敗: {str(e)}")

@app.post("/admin/vector-db/cleanup-empty")
async def cleanup_empty_folders(request: Request):
    """清理空的向量資料庫資料夾"""
    await check_admin(request)
    try:
        import shutil
        
        cleaned_count = 0
        cleaned_folders = []
        
        if vector_db_manager.base_path.exists():
            for folder in vector_db_manager.base_path.iterdir():
                if folder.is_dir():
                    # 檢查是否為空資料夾或只有 .model 文件但沒有實際數據
                    has_vector_data = vector_db_manager.has_vector_data(folder)
                    is_training = vector_db_manager.is_training(folder)
                    
                    # 如果沒有向量數據且沒有在訓練，檢查是否為空資料夾
                    if not has_vector_data and not is_training:
                        files = list(folder.rglob('*'))
                        # 只有 .model 文件或完全空的資料夾
                        if len(files) == 0 or (len(files) == 1 and files[0].name == '.model'):
                            try:
                                shutil.rmtree(folder)
                                cleaned_folders.append(folder.name)
                                cleaned_count += 1
                                logger.info(f"清理空資料夾: {folder.name}")
                            except Exception as e:
                                logger.error(f"清理資料夾失敗 {folder.name}: {str(e)}")
        
        return {
            "status": "success",
            "cleaned_count": cleaned_count,
            "cleaned_folders": cleaned_folders
        }
        
    except Exception as e:
        logger.error(f"清理空資料夾失敗: {str(e)}")
        raise HTTPException(status_code=500, detail=f"清理失敗: {str(e)}")

@app.get("/admin/vector-db/stats")
async def get_vector_db_stats(request: Request):
    """獲取向量資料庫統計信息"""
    await check_admin(request)
    try:
        models = vector_db_manager.list_available_models()
        
        total_size = 0
        total_files = 0
        stats_by_status = {
            "with_data": 0,
            "training": 0,
            "empty": 0,
            "usable": 0
        }
        
        model_details = []
        
        for model in models:
            model_path = Path(model['folder_path'])
            
            if model_path.exists():
                # 計算資料夾大小和文件數
                folder_size = sum(f.stat().st_size for f in model_path.rglob('*') if f.is_file())
                file_count = len([f for f in model_path.rglob('*') if f.is_file()])
                
                total_size += folder_size
                total_files += file_count
                
                # 統計狀態
                if model['has_data']:
                    stats_by_status["with_data"] += 1
                    if not model['is_training']:
                        stats_by_status["usable"] += 1
                else:
                    stats_by_status["empty"] += 1
                
                if model['is_training']:
                    stats_by_status["training"] += 1
                
                model_details.append({
                    "name": model['display_name'],
                    "folder_name": model['folder_name'],
                    "size_mb": round(folder_size / (1024 * 1024), 2),
                    "file_count": file_count,
                    "has_data": model['has_data'],
                    "is_training": model['is_training']
                })
        
        return {
            "total_models": len(models),
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "total_files": total_files,
            "stats_by_status": stats_by_status,
            "model_details": sorted(model_details, key=lambda x: x['size_mb'], reverse=True)
        }
        
    except Exception as e:
        logger.error(f"獲取統計信息失敗: {str(e)}")
        raise HTTPException(status_code=500, detail=f"獲取統計失敗: {str(e)}")

@app.get("/admin/vector-db/integrity-check")
async def check_vector_db_integrity(request: Request):
    """檢查向量資料庫完整性"""
    await check_admin(request)
    try:
        models = vector_db_manager.list_available_models()
        issues = []
        all_valid = True
        
        for model in models:
            model_path = Path(model['folder_path'])
            model_issues = []
            
            # 檢查資料夾是否存在
            if not model_path.exists():
                model_issues.append("資料夾不存在")
                all_valid = False
            else:
                # 檢查 .model 文件
                if not (model_path / ".model").exists():
                    model_issues.append("缺少 .model 文件")
                    all_valid = False
                else:
                    # 檢查 .model 文件內容
                    try:
                        model_info = vector_db_manager.get_model_info(model_path)
                        if not model_info:
                            model_issues.append(".model 文件無法讀取")
                            all_valid = False
                        elif not model_info.get('OLLAMA_MODEL') or not model_info.get('OLLAMA_EMBEDDING_MODEL'):
                            model_issues.append(".model 文件缺少必要信息")
                            all_valid = False
                    except Exception as e:
                        model_issues.append(f".model 文件錯誤: {str(e)}")
                        all_valid = False
                
                # 如果聲稱有數據，檢查向量文件
                if model['has_data']:
                    if not (model_path / "index.faiss").exists():
                        model_issues.append("缺少 index.faiss 文件")
                        all_valid = False
                    if not (model_path / "index.pkl").exists():
                        model_issues.append("缺少 index.pkl 文件")
                        all_valid = False
                
                # 檢查鎖定文件的有效性
                if model['is_training']:
                    is_valid, reason = vector_db_manager.is_lock_valid(model_path)
                    if not is_valid:
                        model_issues.append(f"無效的訓練鎖定: {reason}")
                        all_valid = False
            
            if model_issues:
                issues.append({
                    "model_name": model['display_name'],
                    "folder_name": model['folder_name'],
                    "issues": model_issues
                })
        
        return {
            "all_valid": all_valid,
            "total_models_checked": len(models),
            "models_with_issues": len(issues),
            "issues": issues
        }
        
    except Exception as e:
        logger.error(f"完整性檢查失敗: {str(e)}")
        raise HTTPException(status_code=500, detail=f"完整性檢查失敗: {str(e)}")

# 向量資料庫內容維護 API
@app.get("/admin/vector-db/documents")
async def get_vector_documents(request: Request, folder_name: str = Query(...), page: int = Query(1), page_size: int = Query(20)):
    """獲取向量資料庫中的文檔列表"""
    await check_admin(request)
    try:
        models = vector_db_manager.list_available_models()
        target_model = None
        
        for model in models:
            if model['folder_name'] == folder_name:
                target_model = model
                break
        
        if not target_model:
            raise HTTPException(status_code=404, detail=f"找不到模型: {folder_name}")
        
        if not target_model['has_data']:
            return {"documents": [], "total": 0, "page": page, "page_size": page_size}
        
        # 創建文檔索引器來訪問向量資料庫
        from indexer.document_indexer import DocumentIndexer
        indexer = DocumentIndexer(
            vector_db_path=target_model['folder_path'],
            ollama_embedding_model=target_model['model_info']['OLLAMA_EMBEDDING_MODEL']
        )
        
        # 獲取所有文檔
        vector_store = indexer.get_vector_store()
        if not vector_store:
            return {"documents": [], "total": 0, "page": page, "page_size": page_size}
        
        all_docs = vector_store.docstore._dict
        documents = []
        
        for doc_id, doc in all_docs.items():
            documents.append({
                "id": doc_id,
                "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                "full_content": doc.page_content,
                "metadata": doc.metadata,
                "file_path": doc.metadata.get("file_path", "未知"),
                "file_name": doc.metadata.get("file_name", "未知"),
                "chunk_index": doc.metadata.get("chunk_index", 0)
            })
        
        # 分頁
        total = len(documents)
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        paginated_docs = documents[start_idx:end_idx]
        
        return {
            "documents": paginated_docs,
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": (total + page_size - 1) // page_size
        }
        
    except Exception as e:
        logger.error(f"獲取向量文檔失敗: {str(e)}")
        raise HTTPException(status_code=500, detail=f"獲取文檔失敗: {str(e)}")

@app.get("/admin/vector-db/document/{doc_id}")
async def get_vector_document(request: Request, folder_name: str = Query(...), doc_id: str = ""):
    """獲取特定文檔的詳細內容"""
    await check_admin(request)
    try:
        models = vector_db_manager.list_available_models()
        target_model = None
        
        for model in models:
            if model['folder_name'] == folder_name:
                target_model = model
                break
        
        if not target_model:
            raise HTTPException(status_code=404, detail=f"找不到模型: {folder_name}")
        
        if not target_model['has_data']:
            raise HTTPException(status_code=404, detail="模型沒有數據")
        
        # 創建文檔索引器來訪問向量資料庫
        from indexer.document_indexer import DocumentIndexer
        indexer = DocumentIndexer(
            vector_db_path=target_model['folder_path'],
            ollama_embedding_model=target_model['model_info']['OLLAMA_EMBEDDING_MODEL']
        )
        
        vector_store = indexer.get_vector_store()
        if not vector_store:
            raise HTTPException(status_code=404, detail="向量存儲不存在")
        
        all_docs = vector_store.docstore._dict
        if doc_id not in all_docs:
            raise HTTPException(status_code=404, detail=f"找不到文檔: {doc_id}")
        
        doc = all_docs[doc_id]
        return {
            "id": doc_id,
            "content": doc.page_content,
            "metadata": doc.metadata,
            "file_path": doc.metadata.get("file_path", "未知"),
            "file_name": doc.metadata.get("file_name", "未知"),
            "chunk_index": doc.metadata.get("chunk_index", 0)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"獲取文檔詳情失敗: {str(e)}")
        raise HTTPException(status_code=500, detail=f"獲取文檔詳情失敗: {str(e)}")

@app.put("/admin/vector-db/document/{doc_id}")
async def update_vector_document(request: Request, folder_name: str = Query(...), doc_id: str = "", update_data: dict = Body(...)):
    """更新向量資料庫中的文檔內容"""
    await check_admin(request)
    try:
        models = vector_db_manager.list_available_models()
        target_model = None
        
        for model in models:
            if model['folder_name'] == folder_name:
                target_model = model
                break
        
        if not target_model:
            raise HTTPException(status_code=404, detail=f"找不到模型: {folder_name}")
        
        if not target_model['has_data']:
            raise HTTPException(status_code=404, detail="模型沒有數據")
        
        # 檢查是否正在訓練
        if target_model['is_training']:
            raise HTTPException(status_code=423, detail="模型正在訓練中，無法編輯")
        
        new_content = update_data.get("content")
        if not new_content:
            raise HTTPException(status_code=400, detail="缺少內容")
        
        # 創建文檔索引器來訪問向量資料庫
        from indexer.document_indexer import DocumentIndexer
        indexer = DocumentIndexer(
            vector_db_path=target_model['folder_path'],
            ollama_embedding_model=target_model['model_info']['OLLAMA_EMBEDDING_MODEL']
        )
        
        vector_store = indexer.get_vector_store()
        if not vector_store:
            raise HTTPException(status_code=404, detail="向量存儲不存在")
        
        all_docs = vector_store.docstore._dict
        if doc_id not in all_docs:
            raise HTTPException(status_code=404, detail=f"找不到文檔: {doc_id}")
        
        # 更新文檔內容
        old_doc = all_docs[doc_id]
        updated_doc = Document(
            page_content=new_content,
            metadata=old_doc.metadata
        )
        
        # 重新計算嵌入向量
        new_embedding = indexer.embeddings.embed_documents([new_content])[0]
        
        # 更新向量存儲
        vector_store.docstore._dict[doc_id] = updated_doc
        
        # 更新FAISS索引中的向量
        import numpy as np
        vector_id = int(doc_id.split('_')[-1]) if '_' in doc_id else 0
        vector_store.index.remove_ids(np.array([vector_id]))
        vector_store.index.add(np.array([new_embedding]).astype('float32'))
        
        # 保存更新後的向量存儲
        indexer._save_vector_store()
        
        logger.info(f"成功更新文檔 {doc_id} 的內容")
        
        return {
            "status": "success",
            "message": f"成功更新文檔內容",
            "document": {
                "id": doc_id,
                "content": new_content,
                "metadata": updated_doc.metadata
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"更新文檔失敗: {str(e)}")
        raise HTTPException(status_code=500, detail=f"更新文檔失敗: {str(e)}")

@app.delete("/admin/vector-db/document/{doc_id}")
async def delete_vector_document(request: Request, folder_name: str = Query(...), doc_id: str = ""):
    """刪除向量資料庫中的特定文檔"""
    await check_admin(request)
    try:
        models = vector_db_manager.list_available_models()
        target_model = None
        
        for model in models:
            if model['folder_name'] == folder_name:
                target_model = model
                break
        
        if not target_model:
            raise HTTPException(status_code=404, detail=f"找不到模型: {folder_name}")
        
        if not target_model['has_data']:
            raise HTTPException(status_code=404, detail="模型沒有數據")
        
        # 檢查是否正在訓練
        if target_model['is_training']:
            raise HTTPException(status_code=423, detail="模型正在訓練中，無法刪除")
        
        # 創建文檔索引器來訪問向量資料庫
        from indexer.document_indexer import DocumentIndexer
        indexer = DocumentIndexer(
            vector_db_path=target_model['folder_path'],
            ollama_embedding_model=target_model['model_info']['OLLAMA_EMBEDDING_MODEL']
        )
        
        vector_store = indexer.get_vector_store()
        if not vector_store:
            raise HTTPException(status_code=404, detail="向量存儲不存在")
        
        all_docs = vector_store.docstore._dict
        if doc_id not in all_docs:
            raise HTTPException(status_code=404, detail=f"找不到文檔: {doc_id}")
        
        # 刪除文檔
        del all_docs[doc_id]
        
        # 從FAISS索引中移除對應的向量
        import numpy as np
        try:
            vector_id = int(doc_id.split('_')[-1]) if '_' in doc_id else 0
            vector_store.index.remove_ids(np.array([vector_id]))
        except Exception as e:
            logger.warning(f"無法從FAISS索引中移除向量 {vector_id}: {str(e)}")
        
        # 保存更新後的向量存儲
        indexer._save_vector_store()
        
        logger.info(f"成功刪除文檔 {doc_id}")
        
        return {
            "status": "success",
            "message": f"成功刪除文檔 {doc_id}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"刪除文檔失敗: {str(e)}")
        raise HTTPException(status_code=500, detail=f"刪除文檔失敗: {str(e)}")

@app.post("/admin/vector-db/document")
async def add_vector_document(request: Request, folder_name: str = Query(...), doc_data: dict = Body(...)):
    """向向量資料庫添加新文檔"""
    await check_admin(request)
    try:
        models = vector_db_manager.list_available_models()
        target_model = None
        
        for model in models:
            if model['folder_name'] == folder_name:
                target_model = model
                break
        
        if not target_model:
            raise HTTPException(status_code=404, detail=f"找不到模型: {folder_name}")
        
        # 檢查是否正在訓練
        if target_model['is_training']:
            raise HTTPException(status_code=423, detail="模型正在訓練中，無法添加")
        
        content = doc_data.get("content")
        metadata = doc_data.get("metadata", {})
        
        if not content:
            raise HTTPException(status_code=400, detail="缺少文檔內容")
        
        # 創建文檔索引器來訪問向量資料庫
        from indexer.document_indexer import DocumentIndexer
        indexer = DocumentIndexer(
            vector_db_path=target_model['folder_path'],
            ollama_embedding_model=target_model['model_info']['OLLAMA_EMBEDDING_MODEL']
        )
        
        # 創建新文檔
        new_doc = Document(
            page_content=content,
            metadata={
                "file_name": metadata.get("file_name", "手動添加"),
                "file_path": metadata.get("file_path", "manual_add"),
                "chunk_index": metadata.get("chunk_index", 0),
                "added_manually": True,
                **metadata
            }
        )
        
        # 添加到向量存儲
        vector_store = indexer.get_vector_store()
        if vector_store:
            # 使用現有的向量存儲添加文檔
            vector_store.add_documents([new_doc])
        else:
            # 創建新的向量存儲
            vector_store = indexer.embeddings.from_documents([new_doc], indexer.embeddings)
            indexer.vector_store = vector_store
        
        # 保存向量存儲
        indexer._save_vector_store()
        
        logger.info(f"成功添加新文檔到模型 {folder_name}")
        
        return {
            "status": "success",
            "message": "成功添加新文檔",
            "document": {
                "content": content,
                "metadata": new_doc.metadata
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"添加文檔失敗: {str(e)}")
        raise HTTPException(status_code=500, detail=f"添加文檔失敗: {str(e)}")

@app.get("/health")
async def health_check():
    """健康檢查端點"""
    return {"status": "healthy", "timestamp": "2025-08-12", "api_version": "1.0.0"}

# 移除了多餘的模型狀態檢測和下載端點

@app.get("/api/supported-languages")
async def get_supported_languages_endpoint():
    """獲取支持的語言列表"""
    try:
        languages = get_supported_languages()
        return {"supported_languages": languages}
    except Exception as e:
        logger.error(f"獲取支持語言列表失敗: {str(e)}")
        raise HTTPException(status_code=500, detail=f"獲取支持語言列表失敗: {str(e)}")

@app.get("/files", response_model=List[FileInfo])
async def list_files():
    """
    列出所有已索引的文件
    
    返回系統中已索引的文件列表及其相關信息
    """
    try:
        # 嘗試使用默認RAG引擎獲取文件列表
        engine = get_rag_engine()
        indexed_files = engine.document_indexer.list_indexed_files()
        return indexed_files
    except HTTPException:
        # 重新拋出 HTTP 異常
        raise
    except Exception as e:
        logger.error(f"獲取文件列表時出錯: {str(e)}")
        raise HTTPException(status_code=500, detail=f"獲取文件列表時出錯: {str(e)}")

# 配置更新端點
class ConfigUpdateRequest(BaseModel):
    platform: str
    language_model: Optional[str] = None
    embedding_model: Optional[str] = None
    inference_engine: Optional[str] = "transformers"
    rag_mode: Optional[str] = "traditional"
    language: Optional[str] = "dynamic"
    setup_completed: bool = True

# === 依賴與版本鎖定 / 健康檢查 ===
CORE_DEP_PACKAGES = [
    'fastapi','uvicorn','langchain','langchain-community','langchain-core','langchain-huggingface',
    'langchain-ollama','langchain-text-splitters','transformers','torch','sentence-transformers',
    'numpy','pydantic','faiss-cpu','streamlit'
]

class DependencyPinResult(BaseModel):
    package: str
    old: Optional[str]
    new: str

class DependencyPinResponse(BaseModel):
    changes: List[DependencyPinResult]
    message: str
    executed: bool

class DependencyStatusItem(BaseModel):
    package: str
    pyproject: Optional[str]
    requirements: Optional[str]
    installed: Optional[str]
    status: str  # aligned | mismatch | missing

class DependencyStatusResponse(BaseModel):
    items: List[DependencyStatusItem]
    generated_at: int
    mismatch_count: int | None = None

class DependencyLockExportResponse(BaseModel):
    success: bool
    lock_ran: bool
    export_ran: bool
    message: str
    changed_requirements: List[str] | None = None  # diff style lines
    elapsed_seconds: float | None = None
    last_audit_ts: Optional[int] = None

class DependencyAuditEntry(BaseModel):
    ts: int
    mismatch_count: int
    items: Optional[List[str]] = None  # package statuses summary

class DependencyAuditLogResponse(BaseModel):
    count: int
    entries: List[DependencyAuditEntry]
    limit: int
    latest_ts: Optional[int] = None

def _load_requirements_versions(req_path: Path) -> dict:
    import re
    versions = {}
    if not req_path.exists():
        return versions
    pat = re.compile(r'^(?P<name>[A-Za-z0-9_.-]+)==(?P<ver>[^=]+)$')
    for line in req_path.read_text(encoding='utf-8', errors='ignore').splitlines():
        line=line.strip()
        if not line or line.startswith('#'): continue
        m = pat.match(line)
        if m:
            versions[m.group('name').lower()] = m.group('ver')
    return versions

def _load_pyproject_deps(py_path: Path) -> dict:
    import re
    try:
        try:
            import tomllib  # type: ignore  # py311+
        except ModuleNotFoundError:  # pragma: no cover
            import tomli as tomllib  # type: ignore
        data = tomllib.loads(py_path.read_text(encoding='utf-8'))
        return data.get('tool',{}).get('poetry',{}).get('dependencies',{}) or {}
    except Exception:
        # fallback 簡單解析
        deps = {}
        text = py_path.read_text(encoding='utf-8')
        in_dep=False
        for line in text.splitlines():
            s=line.strip()
            if s.startswith('[tool.poetry.dependencies]'):
                in_dep=True; continue
            if in_dep and s.startswith('['): break
            if in_dep and '=' in s and not s.startswith('#'):
                m=re.match(r'([A-Za-z0-9_.-]+)\s*=\s*"([^"]+)"', s)
                if m:
                    deps[m.group(1)] = m.group(2)
        return deps

def _dependency_audit_write(result_items: List['DependencyStatusItem']):
    """將當前依賴狀態寫入審計日誌 (logs/dependency_audit.log)。

    行格式: ts=<epoch> mismatch_count=<int> item=<pkg>:<status>:<py>:<req>:<inst> ...
    僅保留最近 500 行（滾動截斷）。
    """
    try:
        from config.config import LOGS_DIR
        os.makedirs(LOGS_DIR, exist_ok=True)
        path = Path(LOGS_DIR) / 'dependency_audit.log'
        mismatch = sum(1 for it in result_items if it.status in ('mismatch','missing'))
        parts = [f"ts={int(__import__('time').time())}", f"mismatch_count={mismatch}"]
        for it in result_items:
            # 簡要表示，避免過長
            parts.append(f"item={it.package}:{it.status}:{it.pyproject or '-'}:{it.requirements or '-'}:{it.installed or '-'}")
        line = ' '.join(parts) + '\n'
        # 追加
        with open(path, 'a', encoding='utf-8') as f:
            f.write(line)
        # 截斷
        try:
            with open(path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            if len(lines) > 500:
                with open(path, 'w', encoding='utf-8') as f:
                    f.writelines(lines[-500:])
        except Exception:
            pass
    except Exception as e:
        logger.debug(f"寫入依賴審計失敗: {e}")

@app.post("/admin/dependencies/pin", response_model=DependencyPinResponse)
async def admin_pin_core_dependencies(request: Request):
    await check_admin(request)
    try:
        py_path = Path('pyproject.toml')
        req_path = Path('requirements.txt')
        if not py_path.exists():
            raise HTTPException(status_code=400, detail='pyproject.toml 不存在')
        req_versions = _load_requirements_versions(req_path)
        deps = _load_pyproject_deps(py_path)
        text = py_path.read_text(encoding='utf-8')
        changes = []
        import re
        def _replace_line(pkg: str, new_ver: str, original: str) -> str:
            pattern = re.compile(rf'^(\s*{re.escape(pkg)}\s*=\s*)"[^"]+"', re.MULTILINE)
            return pattern.sub(lambda m: f"{m.group(1)}\"{new_ver}\"", original)
        for pkg in CORE_DEP_PACKAGES:
            # 查找 pyproject 中現有版本表示
            match_key = None
            for k in deps.keys():
                if k.lower()==pkg:
                    match_key = k; break
            if not match_key: continue
            req_ver = req_versions.get(pkg)
            if not req_ver: continue
            cur_val = deps[match_key]
            if isinstance(cur_val, dict):
                cur_ver = cur_val.get('version')
                if cur_ver and cur_ver!=req_ver:
                    cur_val['version']=req_ver
                    changes.append((pkg, cur_ver, req_ver))
            else:
                if cur_val!=req_ver:
                    text = _replace_line(match_key, req_ver, text)
                    changes.append((pkg, cur_val, req_ver))
        if changes:
            py_path.write_text(text, encoding='utf-8')
            logger.info(f"依賴鎖定更新: {changes}")
        return DependencyPinResponse(
            changes=[DependencyPinResult(package=pkg, old=old, new=new) for pkg,old,new in changes],
            message = '已更新並建議重新執行 poetry lock --no-update' if changes else '無需更新，已對齊',
            executed = True
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"執行依賴鎖定失敗: {e}")
        raise HTTPException(status_code=500, detail=f"執行依賴鎖定失敗: {e}")

@app.get("/admin/dependencies/status", response_model=DependencyStatusResponse)
async def admin_dependency_status(request: Request):
    await check_admin(request)
    try:
        from importlib import metadata as importlib_metadata
        py_deps = _load_pyproject_deps(Path('pyproject.toml'))
        req_versions = _load_requirements_versions(Path('requirements.txt'))
        items = []
        mismatch = 0
        for pkg in CORE_DEP_PACKAGES:
            py_ver = None
            for k,v in py_deps.items():
                if k.lower()==pkg:
                    if isinstance(v, dict):
                        py_ver = v.get('version')
                    else:
                        py_ver = v
                    break
            req_ver = req_versions.get(pkg)
            try:
                inst_ver = importlib_metadata.version(pkg)
            except Exception:
                inst_ver = None
            status = 'aligned'
            # 判斷狀態
            want = req_ver or py_ver
            if not (py_ver or req_ver):
                status = 'missing'
            elif inst_ver and want and inst_ver != want:
                status = 'mismatch'
            elif not inst_ver:
                status = 'missing'
            if status in ('mismatch','missing'):
                mismatch += 1
            items.append(DependencyStatusItem(
                package=pkg,
                pyproject=py_ver,
                requirements=req_ver,
                installed=inst_ver,
                status=status
            ))
        resp = DependencyStatusResponse(items=items, generated_at=int(__import__('time').time()), mismatch_count=mismatch)
        # 可選即時審計: 若 query 參數 audit=true 則寫入審計
        try:
            if request.query_params.get('audit','false').lower() in ('1','true','yes'):
                _dependency_audit_write(items)
        except Exception:
            pass
        return resp
    except Exception as e:
        logger.error(f"取得依賴狀態失敗: {e}")
        raise HTTPException(status_code=500, detail=f"取得依賴狀態失敗: {e}")

@app.post("/admin/dependencies/lock-export", response_model=DependencyLockExportResponse)
async def admin_dependency_lock_export(request: Request):
    """執行 poetry lock --no-update 以及 export requirements.txt，並返回差異摘要。

    流程：
      1. 讀取現有 requirements.txt 保存快照
      2. 執行 lock (若可行)
      3. 執行 export (覆寫 requirements.txt)
      4. 產生差異行 (新增/修改/刪除)
    """
    await check_admin(request)
    import time as _time, shutil, subprocess, difflib
    start = _time.time()
    req_path = Path('requirements.txt')
    before_lines = []
    if req_path.exists():
        before_lines = req_path.read_text(encoding='utf-8', errors='ignore').splitlines()
    # 決定 poetry 命令
    poetry_cmd = shutil.which('poetry') or None
    lock_ran = False
    export_ran = False
    messages = []
    try:
        if poetry_cmd:
            # lock
            try:
                cp = subprocess.run([poetry_cmd, 'lock', '--no-update'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=300)
                lock_ran = cp.returncode == 0
                messages.append(f"lock rc={cp.returncode}")
                if cp.returncode != 0:
                    messages.append(cp.stderr[:400])
            except Exception as e:
                messages.append(f"lock 失敗: {e}")
            # export
            try:
                cp2 = subprocess.run([poetry_cmd, 'export', '-f', 'requirements.txt', '--output', 'requirements.txt', '--without-hashes'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=180)
                export_ran = cp2.returncode == 0
                messages.append(f"export rc={cp2.returncode}")
                if cp2.returncode != 0:
                    messages.append(cp2.stderr[:400])
            except Exception as e:
                messages.append(f"export 失敗: {e}")
        else:
            messages.append('找不到 poetry 可執行檔 (未執行 lock/export)')
        # 差異
        after_lines = []
        if req_path.exists():
            after_lines = req_path.read_text(encoding='utf-8', errors='ignore').splitlines()
        diff = list(difflib.unified_diff(before_lines, after_lines, fromfile='before', tofile='after', lineterm=''))
        # 過濾 header 只保留變動行
        changed = [l for l in diff if l.startswith(('+','-')) and not l.startswith(('+++','---'))][:200]
        elapsed = round(_time.time() - start, 2)
        # 讀取最新審計時間
        last_audit_ts = None
        try:
            from config.config import LOGS_DIR
            ap = Path(LOGS_DIR)/'dependency_audit.log'
            if ap.exists():
                with open(ap,'r',encoding='utf-8',errors='ignore') as f:
                    for line in reversed(f.readlines()):
                        if line.strip().startswith('ts='):
                            try:
                                last_audit_ts=int(line.split()[0].split('=')[1])
                            except Exception:
                                pass
                            break
        except Exception:
            pass
        return DependencyLockExportResponse(
            success = export_ran or lock_ran,
            lock_ran = lock_ran,
            export_ran = export_ran,
            message=' | '.join(messages),
            changed_requirements=changed or None,
            elapsed_seconds=elapsed,
            last_audit_ts=last_audit_ts
        )
    except Exception as e:
        logger.error(f"執行 lock/export 失敗: {e}")
        raise HTTPException(status_code=500, detail=f"執行 lock/export 失敗: {e}")

@app.post("/api/config/update")
async def update_config(config: ConfigUpdateRequest):
    """更新系統配置"""
    try:
        # 保存配置到文件
        config_file = Path("config/user_setup.json")
        config_file.parent.mkdir(exist_ok=True)
        
        config_data = config.dict()
        
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"配置已更新: {config_data}")
        
        return {
            "success": True,
            "message": "配置更新成功",
            "config": config_data
        }
        
    except Exception as e:
        logger.error(f"更新配置失敗: {str(e)}")
        raise HTTPException(status_code=500, detail=f"更新配置失敗: {str(e)}")

@app.get("/api/config/current")
async def get_current_config():
    """獲取當前配置"""
    try:
        config_file = Path("config/user_setup.json")
        
        if config_file.exists():
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
        else:
            config = {
                "platform": "huggingface",
                "language_model": None,
                "embedding_model": None,
                "inference_engine": "transformers",
                "rag_mode": "traditional",
                "language": "dynamic",
                "setup_completed": False
            }
        
        return {
            "success": True,
            "config": config
        }
        
    except Exception as e:
        logger.error(f"獲取配置失敗: {str(e)}")
        raise HTTPException(status_code=500, detail=f"獲取配置失敗: {str(e)}")

# === 依賴審計與安全度量附加端點 ===

@app.post("/admin/dependencies/audit-run", response_model=DependencyAuditEntry)
async def admin_dependency_audit_run(request: Request):
    """立即執行依賴審計並寫入日誌，返回本次結果摘要。"""
    await check_admin(request)
    try:
        # 重用狀態邏輯
        from importlib import metadata as importlib_metadata
        py_deps = _load_pyproject_deps(Path('pyproject.toml'))
        req_versions = _load_requirements_versions(Path('requirements.txt'))
        items: List[DependencyStatusItem] = []
        for pkg in CORE_DEP_PACKAGES:
            py_ver=None
            for k,v in py_deps.items():
                if k.lower()==pkg:
                    if isinstance(v, dict):
                        py_ver=v.get('version')
                    else:
                        py_ver=v
                    break
            req_ver = req_versions.get(pkg)
            try:
                inst_ver = importlib_metadata.version(pkg)
            except Exception:
                inst_ver = None
            status='aligned'
            want = req_ver or py_ver
            if not (py_ver or req_ver):
                status='missing'
            elif inst_ver and want and inst_ver != want:
                status='mismatch'
            elif not inst_ver:
                status='missing'
            items.append(DependencyStatusItem(package=pkg, pyproject=py_ver, requirements=req_ver, installed=inst_ver, status=status))
        _dependency_audit_write(items)
        mismatch = sum(1 for it in items if it.status in ('mismatch','missing'))
        return DependencyAuditEntry(ts=int(__import__('time').time()), mismatch_count=mismatch, items=[f"{it.package}:{it.status}" for it in items])
    except Exception as e:
        logger.error(f"依賴審計失敗: {e}")
        raise HTTPException(status_code=500, detail=f"依賴審計失敗: {e}")

@app.get("/admin/dependencies/audit-log", response_model=DependencyAuditLogResponse)
async def admin_dependency_audit_log(request: Request, limit: int = Query(100, ge=10, le=500)):
    """取得最近依賴審計結果，供前端趨勢顯示。"""
    await check_admin(request)
    try:
        from config.config import LOGS_DIR
        path = Path(LOGS_DIR)/'dependency_audit.log'
        if not path.exists():
            return DependencyAuditLogResponse(count=0, entries=[], limit=limit, latest_ts=None)
        lines=[]
        with open(path,'r',encoding='utf-8',errors='ignore') as f:
            for ln in f:
                if ln.strip():
                    lines.append(ln.strip())
        entries=[]
        for ln in lines[-limit:]:
            parts=ln.split()
            ts_val=None; mismatch=0; pkgs=[]
            for p in parts:
                if p.startswith('ts='):
                    try: ts_val=int(p.split('=',1)[1])
                    except: ts_val=None
                elif p.startswith('mismatch_count='):
                    try: mismatch=int(p.split('=',1)[1])
                    except: mismatch=0
                elif p.startswith('item='):
                    pkgs.append(p.split('=',1)[1])
            if ts_val:
                entries.append(DependencyAuditEntry(ts=ts_val, mismatch_count=mismatch, items=pkgs))
        latest_ts = entries[-1].ts if entries else None
        return DependencyAuditLogResponse(count=len(entries), entries=entries, limit=limit, latest_ts=latest_ts)
    except Exception as e:
        logger.error(f"讀取依賴審計日誌失敗: {e}")
        raise HTTPException(status_code=500, detail=f"讀取依賴審計日誌失敗: {e}")

@app.get('/api/dynamic/security-metrics')
async def get_security_metrics(window_minutes: int = Query(60, ge=5, le=720)):
    """安全事件統計: 回傳最近 window_minutes 內每分鐘計數、總數、速率旗標與最新事件。"""
    try:
        from rag_engine.dynamic_rag_base import SECURITY_EVENTS
        now_ts = int(__import__('time').time())
        window_seconds = window_minutes*60
        buckets = {}
        total=0
        recent_10m=0
        for ev in reversed(SECURITY_EVENTS):
            delta = now_ts - ev['ts']
            if delta > window_seconds:
                break
            minute_key = (ev['ts']//60)*60  # 對齊分鐘
            buckets.setdefault(minute_key,0)
            buckets[minute_key]+=1
            total+=1
            if delta <= 600:
                recent_10m+=1
        level=None
        if recent_10m >= 15:
            level='critical'
        elif recent_10m >= 8:
            level='elevated'
        series = [{'minute_ts': k, 'count': v} for k, v in sorted(buckets.items())]
        return {'total':total,'recent_10m':recent_10m,'level':level,'window_minutes':window_minutes,'series':series}
    except Exception as e:
        logger.error(f"取得安全指標失敗: {e}")
        raise HTTPException(status_code=500, detail=f"取得安全指標失敗: {e}")

@app.get('/api/dynamic/file-check')
async def dynamic_file_check(folder_path: Optional[str] = None):
    """檢查動態RAG檔案數量，返回是否應該阻擋處理"""
    try:
        from rag_engine.dynamic_rag_base import SmartFileRetriever
        
        retriever = SmartFileRetriever(folder_path=folder_path)
        scope_info = retriever._quick_estimate_file_count(str(retriever.folder_path))
        
        estimated_count = scope_info.get('estimated_total', 0)
        warning_level = "none"
        should_block = False
        warning_message = None
        
        # 使用與動態RAG相同的閾值邏輯
        high_cut = 60000
        fast_cut = 40000
        medium_cut = 20000
        low_cut = 10000
        
        if estimated_count > high_cut:
            warning_level = "critical_blocked"
            should_block = True
            warning_message = f"檢測到極大量文件 (估算約 {estimated_count} 個)，系統已停止處理以確保穩定性。請選擇特定資料夾限制搜索範圍後重試。"
        elif estimated_count > fast_cut:
            warning_level = "high"
            warning_message = f"檢測到大量文件 (估算約 {estimated_count} 個)，強烈建議選擇特定資料夾範圍以提高搜索精度和速度。"
        elif estimated_count > medium_cut:
            warning_level = "medium"
            warning_message = f"檢測到較多文件 (估算約 {estimated_count} 個)，若搜索結果不理想可考慮限制搜索範圍。"
        elif estimated_count > low_cut:
            warning_level = "low"
            warning_message = f"檢測到文件數量較多 (估算約 {estimated_count} 個)，可考慮限制搜索範圍以獲得更精確的結果。"
            
        return {
            "estimated_file_count": estimated_count,
            "warning_level": warning_level,
            "should_block": should_block,
            "warning_message": warning_message,
            "folder_limited": retriever.folder_path != retriever.base_path,
            "effective_folder": str(retriever.folder_path),
            "confidence": scope_info.get('confidence', 'low'),
            "method": scope_info.get('method', 'unknown')
        }
        
    except Exception as e:
        logger.error(f"動態檔案檢查失敗: {str(e)}")
        return {
            "estimated_file_count": 0,
            "warning_level": "error",
            "should_block": True,
            "warning_message": f"估算過程發生錯誤: {str(e)}",
            "folder_limited": False,
            "effective_folder": folder_path or "/"
        }


@app.post('/api/dynamic/quick-estimate')
async def dynamic_quick_estimate(request_data: dict = Body(...)):
    """快速文件數量估算端點 - 專門用於前端即時顯示
    
    接收參數:
    - folder_path: 可選的資料夾路徑
    - quick_mode: 是否使用快速模式 (默認true)
    """
    try:
        folder_path = request_data.get('folder_path')
        quick_mode = request_data.get('quick_mode', True)
        
        from rag_engine.dynamic_rag_base import SmartFileRetriever
        
        retriever = SmartFileRetriever(folder_path=folder_path)
        
        # 使用快速估算模式
        if quick_mode:
            # 使用較少的採樣進行快速估算
            scope_info = retriever._quick_estimate_file_count(
                str(retriever.folder_path),
                max_sample_dirs=30  # 進一步減少採樣數量提高速度
            )
        else:
            # 完整估算
            scope_info = retriever._quick_estimate_file_count(str(retriever.folder_path))
        
        estimated_count = scope_info.get('estimated_total', 0)
        confidence = scope_info.get('confidence', 'low')
        method = scope_info.get('method', 'sampling')
        
        # 計算警告等級
        warning_level = "none"
        warning_message = None
        should_block = False
        
        # 閾值檢查
        if estimated_count > 60000:
            warning_level = "critical"
            should_block = True
            warning_message = f"檔案數量過多 (約 {estimated_count:,} 個)，請縮小搜索範圍"
        elif estimated_count > 40000:
            warning_level = "high"
            warning_message = f"檔案數量較多 (約 {estimated_count:,} 個)，建議限制搜索範圍"
        elif estimated_count > 20000:
            warning_level = "medium"
            warning_message = f"檔案數量中等 (約 {estimated_count:,} 個)，可考慮限制範圍"
        elif estimated_count > 10000:
            warning_level = "low"
            warning_message = f"檔案數量較多 (約 {estimated_count:,} 個)"
        else:
            warning_level = "safe"
            warning_message = f"檔案數量適中 (約 {estimated_count:,} 個)"
        
        # 建構回應
        response_data = {
            "estimated_file_count": estimated_count,
            "warning_level": warning_level,
            "warning_message": warning_message,
            "should_block": should_block,
            "folder_limited": bool(folder_path),
            "effective_folder": str(retriever.folder_path),
            "confidence": confidence,
            "method": method,
            "quick_mode": quick_mode,
            "estimation_details": {
                "sampled_dirs": scope_info.get('sampled_dirs', 0),
                "total_dirs": scope_info.get('total_dirs', 0),
                "mean_files_per_dir": scope_info.get('mean_files_per_dir', 0),
                "confidence_interval_width": scope_info.get('confidence_interval_width'),
                "max_depth_reached": scope_info.get('max_depth_reached', 0)
            }
        }
        
        return response_data
        
    except Exception as e:
        logger.error(f"快速檔案估算失敗: {str(e)}")
        return {
            "estimated_file_count": 0,
            "warning_level": "error",
            "warning_message": f"估算失敗: {str(e)}",
            "should_block": True,
            "folder_limited": False,
            "effective_folder": folder_path or "/",
            "confidence": "unknown",
            "method": "error",
            "quick_mode": quick_mode,
            "estimation_details": {}
        }


# 背景估算任務管理
import asyncio
import uuid
from typing import Dict
from datetime import datetime
import threading

# 針對背景任務的執行緒鎖，避免多執行緒同時讀寫 background_tasks
_bg_tasks_lock = threading.Lock()

# 存儲背景任務的字典
background_tasks: Dict[str, Dict] = {}

@app.post('/api/dynamic/background-estimate')
async def start_background_estimate(request_data: dict = Body(...)):
    """啟動背景文件估算任務 (非阻塞, 以 session 隔離)

    Dedupe 僅在相同 session_id + 相同 canonical_key 下重用；不同瀏覽器分頁或不同使用者可同時獨立估算。
    前端需在呼叫時傳遞 session_id (UUID)。
    """
    folder_path = request_data.get('folder_path') or None
    session_id = request_data.get('session_id') or 'global'
    canonical_key = (folder_path or '__root__')
    try:
        with _bg_tasks_lock:
            for task in background_tasks.values():
                if (task.get('canonical_key') == canonical_key and
                    task.get('session_id') == session_id and
                    task.get('status') in ("running", "starting")):
                    return {
                        "estimation_id": task['id'],
                        "status": task['status'],
                        "message": "同一 session 下已有進行中的估算任務，重用現有ID"
                    }

            estimation_id = str(uuid.uuid4())
            background_tasks[estimation_id] = {
                "id": estimation_id,
                "session_id": session_id,
                "folder_path": folder_path,
                "canonical_key": canonical_key,
                "status": "starting",
                "progress": 0,
                "result": None,
                "error": None,
                "started_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            }

        loop = asyncio.get_running_loop()
        loop.run_in_executor(None, run_background_estimation_sync, estimation_id, folder_path)

        return {"estimation_id": estimation_id, "status": "started", "message": "背景估算已啟動"}
    except Exception as e:
        logger.error(f"啟動背景估算失敗: {str(e)}")
        return {"error": f"啟動背景估算失敗: {str(e)}", "status": "error"}

@app.get('/api/dynamic/background-estimate/{estimation_id}')
async def get_background_estimate_status(estimation_id: str):
    """查詢背景估算任務狀態"""
    
    if estimation_id not in background_tasks:
        raise HTTPException(status_code=404, detail="估算任務不存在")
    
    task = background_tasks[estimation_id]
    task["updated_at"] = datetime.now().isoformat()
    
    return task

@app.delete('/api/dynamic/background-estimate/{estimation_id}')
async def cancel_background_estimate(estimation_id: str):
    """取消背景估算任務"""
    
    if estimation_id not in background_tasks:
        raise HTTPException(status_code=404, detail="估算任務不存在")
    
    task = background_tasks[estimation_id]
    task["status"] = "cancelled"
    task["updated_at"] = datetime.now().isoformat()
    
    # 清理任務（延遲5分鐘後清理）
    async def cleanup_task():
        await asyncio.sleep(300)  # 5分鐘
        if estimation_id in background_tasks:
            del background_tasks[estimation_id]
    
    asyncio.create_task(cleanup_task())
    
    return {"message": "任務已取消", "estimation_id": estimation_id}

def run_background_estimation_sync(estimation_id: str, folder_path: str):
    """(同步版本) 在執行緒中執行重型估算，避免阻塞事件迴圈"""
    try:
        with _bg_tasks_lock:
            task = background_tasks.get(estimation_id)
            if not task or task.get('status') == 'cancelled':
                return
            task['status'] = 'running'
            task['progress'] = 5
            task['updated_at'] = datetime.now().isoformat()

        from rag_engine.dynamic_rag_base import SmartFileRetriever
        retriever = SmartFileRetriever(folder_path=folder_path)

        # 中期進度
        with _bg_tasks_lock:
            task = background_tasks.get(estimation_id)
            if not task or task.get('status') == 'cancelled':
                return
            task['progress'] = 25
            task['updated_at'] = datetime.now().isoformat()

        # 執行估算 (可能較慢)
        scope_info = retriever._quick_estimate_file_count(str(retriever.folder_path))

        with _bg_tasks_lock:
            task = background_tasks.get(estimation_id)
            if not task or task.get('status') == 'cancelled':
                return
            task['progress'] = 70
            task['updated_at'] = datetime.now().isoformat()

        estimated_count = scope_info.get('estimated_total', 0)
        confidence = scope_info.get('confidence', 'low')
        method = scope_info.get('method', 'sampling')

        # 警告等級
        if estimated_count > 60000:
            warning_level, should_block, warning_message = 'critical', True, f"檔案數量過多 (約 {estimated_count:,} 個)，請縮小搜索範圍"
        elif estimated_count > 40000:
            warning_level, should_block, warning_message = 'high', False, f"檔案數量較多 (約 {estimated_count:,} 個)，建議限制搜索範圍"
        elif estimated_count > 20000:
            warning_level, should_block, warning_message = 'medium', False, f"檔案數量中等 (約 {estimated_count:,} 個)，可考慮限制範圍"
        elif estimated_count > 10000:
            warning_level, should_block, warning_message = 'low', False, f"檔案數量較多 (約 {estimated_count:,} 個)"
        else:
            warning_level, should_block, warning_message = 'safe', False, f"檔案數量適中 (約 {estimated_count:,} 個)"

        with _bg_tasks_lock:
            task = background_tasks.get(estimation_id)
            if not task or task.get('status') == 'cancelled':
                return
            task['progress'] = 100
            task['status'] = 'completed'
            task['result'] = {
                'estimated_file_count': estimated_count,
                'warning_level': warning_level,
                'warning_message': warning_message,
                'should_block': should_block,
                'folder_limited': bool(folder_path),
                'effective_folder': str(retriever.folder_path),
                'confidence': confidence,
                'method': method,
                'estimation_details': {
                    'sampled_dirs': scope_info.get('sampled_dirs', 0),
                    'total_dirs': scope_info.get('total_dirs_seen', 0),
                    'mean_files_per_dir': scope_info.get('mean_per_dir', 0),
                    'confidence_interval_width': scope_info.get('ci_width'),
                    'stdev': scope_info.get('stdev', 0)
                }
            }
            task['updated_at'] = datetime.now().isoformat()

        # 延遲清理
        def _delayed_cleanup():
            import time as _t
            _t.sleep(1800)
            with _bg_tasks_lock:
                if estimation_id in background_tasks:
                    del background_tasks[estimation_id]
        threading.Thread(target=_delayed_cleanup, daemon=True).start()

    except Exception as e:
        logger.error(f"背景估算任務失敗: {e}")
        with _bg_tasks_lock:
            if estimation_id in background_tasks:
                background_tasks[estimation_id]['status'] = 'error'
                background_tasks[estimation_id]['error'] = str(e)
                background_tasks[estimation_id]['progress'] = 0

@app.post('/api/dynamic/folder-stats')
async def get_folder_detailed_stats(request_data: dict = Body(...)):
    """獲取資料夾詳細統計信息（文件數量和大小）
    
    這是一個較慢的操作，建議在背景執行
    """
    try:
        folder_path = request_data.get('folder_path', '/')
        max_depth = request_data.get('max_depth', 3)  # 限制深度避免過慢
        
        from pathlib import Path
        import os
        
        if folder_path == '/' or not folder_path:
            from config.config import Q_DRIVE_PATH
            scan_path = Path(Q_DRIVE_PATH)
        else:
            from config.config import Q_DRIVE_PATH
            scan_path = Path(Q_DRIVE_PATH) / folder_path.lstrip('/')
        
        if not scan_path.exists() or not scan_path.is_dir():
            return {
                "error": "資料夾不存在或不可訪問",
                "folder_path": folder_path
            }
        
        # 收集統計信息
        folder_stats = []
        total_files = 0
        total_size = 0
        
        try:
            for item in scan_path.iterdir():
                if item.is_dir():
                    # 統計子資料夾
                    subdir_files = 0
                    subdir_size = 0
                    
                    try:
                        # 遞迴統計但限制深度
                        for root, dirs, files in os.walk(item):
                            # 計算當前深度
                            depth = len(Path(root).relative_to(item).parts)
                            if depth >= max_depth:
                                dirs.clear()  # 不再深入
                                continue
                                
                            for file in files:
                                try:
                                    file_path = Path(root) / file
                                    if file_path.exists():
                                        subdir_files += 1
                                        subdir_size += file_path.stat().st_size
                                except (OSError, PermissionError):
                                    continue
                    except (OSError, PermissionError):
                        pass
                    
                    folder_stats.append({
                        "name": item.name,
                        "type": "folder",
                        "file_count": subdir_files,
                        "size_bytes": subdir_size,
                        "size_human": format_file_size(subdir_size),
                        "path": str(item.relative_to(Path(Q_DRIVE_PATH)))
                    })
                    
                    total_files += subdir_files
                    total_size += subdir_size
                
                elif item.is_file():
                    # 統計檔案
                    try:
                        file_size = item.stat().st_size
                        folder_stats.append({
                            "name": item.name,
                            "type": "file", 
                            "file_count": 1,
                            "size_bytes": file_size,
                            "size_human": format_file_size(file_size),
                            "path": str(item.relative_to(Path(Q_DRIVE_PATH)))
                        })
                        
                        total_files += 1
                        total_size += file_size
                    except (OSError, PermissionError):
                        pass
                        
        except (OSError, PermissionError) as e:
            return {
                "error": f"無法訪問資料夾: {str(e)}",
                "folder_path": folder_path
            }
        
        # 按文件數量排序
        folder_stats.sort(key=lambda x: x["file_count"], reverse=True)
        
        return {
            "folder_path": folder_path,
            "total_files": total_files,
            "total_size_bytes": total_size,
            "total_size_human": format_file_size(total_size),
            "max_depth_scanned": max_depth,
            "items": folder_stats[:100],  # 限制返回前100項
            "total_items": len(folder_stats)
        }
        
    except Exception as e:
        logger.error(f"獲取資料夾統計失敗: {str(e)}")
        return {
            "error": f"獲取資料夾統計失敗: {str(e)}",
            "folder_path": folder_path
        }

def format_file_size(size_bytes: int) -> str:
    """格式化文件大小"""
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    import math
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_names[i]}"


# 啟動應用
if __name__ == "__main__":
    uvicorn.run("main:app", host=APP_HOST, port=APP_PORT, reload=True)
