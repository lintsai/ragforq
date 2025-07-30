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

from config.config import APP_HOST, APP_PORT, is_q_drive_accessible, ADMIN_TOKEN
from rag_engine.rag_engine import RAGEngine
from indexer.document_indexer import DocumentIndexer
from langchain_core.documents import Document
from utils.ollama_utils import ollama_utils
from utils.vector_db_manager import vector_db_manager

# 設置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    show_relevance: bool = True
    selected_model: Optional[str] = None  # 選擇的模型文件夾名稱（支援版本）
    language: str = "繁體中文"  # 新增語言參數

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

# 初始化RAG引擎緩存
rag_engines = {}

def get_rag_engine(selected_model: Optional[str] = None):
    """
    獲取或初始化RAG引擎，支持智能模型選擇和訓練狀態檢查
    
    Args:
        selected_model: 選擇的模型文件夾名稱，如果為 None 則自動選擇最佳可用模型
    """
    global rag_engines
    
    try:
        if selected_model:
            # 使用指定的模型
            if selected_model not in rag_engines:
                logger.info(f"初始化RAG引擎 - 指定模型: {selected_model}")
                
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
                
                # 創建安全的超時RAG引擎
                from utils.timeout_rag_engine import create_safe_rag_engine
                rag_engines[selected_model] = create_safe_rag_engine(
                    vector_db_path=model_path,
                    ollama_model=model_info['OLLAMA_MODEL'],
                    ollama_embedding_model=model_info['OLLAMA_EMBEDDING_MODEL'],
                    timeout=30  # 30秒超時
                )
            
            return rag_engines[selected_model]
        else:
            # 自動選擇最佳可用模型
            if 'auto_selected' not in rag_engines:
                logger.info("自動選擇最佳可用模型...")
                
                # 首先嘗試獲取可用的向量模型
                usable_models = vector_db_manager.get_usable_models()
                
                if usable_models:
                    # 使用第一個可用的向量模型
                    best_model = usable_models[0]
                    model_info = best_model['model_info']
                    model_path = best_model['folder_path']
                    
                    logger.info(f"自動選擇向量模型: {best_model['display_name']}")
                    
                    # 創建安全的超時RAG引擎
                    from utils.timeout_rag_engine import create_safe_rag_engine
                    rag_engines['auto_selected'] = create_safe_rag_engine(
                        vector_db_path=model_path,
                        ollama_model=model_info['OLLAMA_MODEL'],
                        ollama_embedding_model=model_info['OLLAMA_EMBEDDING_MODEL'],
                        timeout=30  # 30秒超時
                    )
                else:
                    # 回退到默認配置
                    logger.info("沒有可用的向量模型，使用默認配置...")
                    
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
                    
                    logger.info(f"使用默認模型 - 語言模型: {default_language_model}, 嵌入模型: {default_embedding_model}")
                    
                    # 使用默認模型創建文檔索引器和RAG引擎
                    document_indexer = DocumentIndexer(ollama_embedding_model=default_embedding_model)
                    rag_engines['auto_selected'] = RAGEngine(document_indexer, ollama_model=default_language_model)
            
            return rag_engines['auto_selected']
    
    except HTTPException:
        # 重新拋出 HTTP 異常
        raise
    except Exception as e:
        logger.error(f"初始化 RAG 引擎時出錯: {str(e)}")
        raise HTTPException(status_code=500, detail=f"初始化 RAG 引擎失敗: {str(e)}")

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
        # 獲取RAG引擎
        engine = get_rag_engine(request.selected_model)
        
        # 根據是否使用查詢改寫選擇函數
        if request.use_query_rewrite:
            answer, sources, documents, rewritten_query = engine.get_answer_with_query_rewrite(request.question, request.language)
        else:
            answer, sources, documents = engine.get_answer_with_sources(request.question, request.language)
            rewritten_query = None
        
        # 處理來源
        source_info_list = []
        if request.include_sources and documents:
            seen_files = set()
            
            for doc in documents:
                metadata = doc.metadata
                file_path = metadata.get("file_path", "")
                
                if file_path and file_path not in seen_files:
                    seen_files.add(file_path)
                    
                    # 提取位置信息
                    location_info = None
                    if "page_number" in metadata:
                        location_info = f"頁碼: {metadata['page_number']}"
                    elif "block_number" in metadata:
                        location_info = f"塊: {metadata['block_number']}"
                    elif "sheet_name" in metadata:
                        location_info = f"工作表: {metadata['sheet_name']}"
                    
                    # 如果啟用了相關性理由，獲取相關性分數和理由
                    relevance_reason = None
                    score = None
                    if request.show_relevance:
                        # 安全獲取分數值
                        score = metadata.get("score")
                        # 如果存在文件內容，使用LLM生成相關性理由
                        try:
                            if hasattr(doc, "page_content") and doc.page_content and doc.page_content.strip():
                                relevance_reason = engine.generate_relevance_reason(request.question, doc.page_content)
                        except Exception as e:
                            logger.error(f"生成相關性理由時出錯: {str(e)}")
                            relevance_reason = "無法生成相關性理由"
                    
                    source_info = SourceInfo(
                        file_name=metadata.get("file_name", os.path.basename(file_path)),
                        file_path=file_path,
                        location_info=location_info,
                        relevance_reason=relevance_reason,
                        score=score
                    )
                    source_info_list.append(source_info)
            
            # 如果指定了最大來源數量，則限制結果
            if request.max_sources:
                source_info_list = source_info_list[:request.max_sources]
        
        response = {
            "answer": answer,
            "sources": source_info_list
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
    log_dir = Path("logs")
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
        
    log_file = Path("logs") / filename
    
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
        # 先將模型資訊寫入暫存檔案，讓 initial_indexing 程序讀取
        training_info = {
            "ollama_model": training_request.ollama_model,
            "ollama_embedding_model": training_request.ollama_embedding_model,
            "version": training_request.version
        }
        with open("temp_training_info.json", "w") as f:
            json.dump(training_info, f)
        
        # 然後使用 supervisorctl 啟動訓練程序
        cmd = [
            "supervisorctl", "start", "initial_indexing"
        ]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            logger.error(f"啟動初始訓練失敗: {stderr.decode()}")
            raise HTTPException(status_code=500, detail=f"啟動訓練失敗: {stderr.decode()}")
        
        return {"status": "訓練已開始", "message": stdout.decode()}
        
    except Exception as e:
        logger.error(f"開始初始訓練失敗: {str(e)}")
        raise HTTPException(status_code=500, detail=f"開始訓練失敗: {str(e)}")

@app.post("/admin/training/incremental")
async def start_incremental_training(request: Request, training_request: TrainingRequest):
    """開始增量訓練"""
    await check_admin(request)
    try:
        # 先將模型資訊寫入暫存檔案，讓 incremental_indexing 程序讀取
        training_info = {
            "ollama_model": training_request.ollama_model,
            "ollama_embedding_model": training_request.ollama_embedding_model,
            "version": training_request.version
        }
        with open("temp_training_info.json", "w") as f:
            json.dump(training_info, f)
        
        # 然後使用 supervisorctl 啟動訓練程序
        cmd = [
            "supervisorctl", "start", "incremental_indexing"
        ]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            logger.error(f"啟動增量訓練失敗: {stderr.decode()}")
            raise HTTPException(status_code=500, detail=f"啟動增量訓練失敗: {stderr.decode()}")
        
        return {"status": "增量訓練已開始", "message": stdout.decode()}
        
    except Exception as e:
        logger.error(f"開始增量訓練失敗: {str(e)}")
        raise HTTPException(status_code=500, detail=f"開始增量訓練失敗: {str(e)}")

@app.post("/admin/training/reindex")
async def start_reindex_training(request: Request, training_request: TrainingRequest):
    """開始重新索引"""
    await check_admin(request)
    try:
        # 先將模型資訊寫入暫存檔案，讓 reindex_indexing 程序讀取
        training_info = {
            "ollama_model": training_request.ollama_model,
            "ollama_embedding_model": training_request.ollama_embedding_model,
            "version": training_request.version
        }
        with open("temp_training_info.json", "w") as f:
            json.dump(training_info, f)
        
        # 然後使用 supervisorctl 啟動訓練程序
        cmd = [
            "supervisorctl", "start", "reindex_indexing"
        ]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            logger.error(f"啟動重新索引失敗: {stderr.decode()}")
            raise HTTPException(status_code=500, detail=f"啟動重新索引失敗: {stderr.decode()}")
        
        return {"status": "重新索引已開始", "message": stdout.decode()}
        
    except Exception as e:
        logger.error(f"開始重新索引失敗: {str(e)}")
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
        raise HTTPException(status_code=500, detail=f"清理無效鎖定失敗: {str(e)}")

@app.get("/health")
async def health_check():
    """健康檢查端點"""
    return {"status": "healthy"}

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

# 啟動應用
if __name__ == "__main__":
    uvicorn.run("main:app", host=APP_HOST, port=APP_PORT, reload=True)
