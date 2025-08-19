import os
import sys
import logging
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
from fastapi import FastAPI, HTTPException, Body, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import subprocess
from fastapi.responses import FileResponse, PlainTextResponse
from scripts.monitor_indexing import get_status_text, get_progress_text, get_monitor_text, get_indexing_status

# 添加項目根目錄到路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

class StatusResponse(BaseModel):
    status: str
    q_drive_accessible: bool
    version: str = "1.0.0"

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
    
    # 創建緩存鍵，包含語言和平台信息
    if use_dynamic_rag:
        # Dynamic RAG 引擎需要根據平台和語言進行緩存
        current_platform = platform or "ollama" # 如果未提供平台，默認為ollama
        cache_key = f"dynamic_{current_platform}_{language}_{ollama_model}_{ollama_embedding_model}"
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
                    dynamic_language_key, None, ollama_model, ollama_embedding_model, current_platform, folder_path
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
    return {
        "status": "運行中",
        "q_drive_accessible": q_drive_accessible,
        "version": "1.0.0"
    }

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
        
        # 根據是否使用查詢改寫選擇函數
        if request.use_query_rewrite:
            answer, sources, documents, rewritten_query = engine.get_answer_with_query_rewrite(request.question)
        else:
            answer, sources, documents = engine.get_answer_with_sources(request.question)
            rewritten_query = None
        
        # 處理來源 - 按相關度排序取得最高分的文件
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
        
        response = {
            "answer": answer,
            "sources": source_info_list,
            "language": engine.get_language(),  # 從引擎獲取正確的語言
            "file_count_warning": None # 移除此處的警告，由前端validate-folder處理
        }
        
        # 如果有改寫的查詢，添加到回應中
        if rewritten_query is not None:
            response["rewritten_query"] = rewritten_query
        
        return response
        
    except Exception as e:
        logger.error(f"處理問題時出錯: {str(e)}")
        raise HTTPException(status_code=500, detail=f"處理問題時出錯: {str(e)}")




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
        full_path = os.path.join(Q_DRIVE_PATH, folder_path_clean)
        
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
        
        for model in models:
            if model['folder_name'] == folder_name:
                target_model = model
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

# 啟動應用
if __name__ == "__main__":
    uvicorn.run("main:app", host=APP_HOST, port=APP_PORT, reload=True)
