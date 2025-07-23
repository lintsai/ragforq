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
from scripts.monitor_indexing import get_status_text, get_progress_text, get_monitor_text

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
    獲取或初始化RAG引擎
    
    Args:
        selected_model: 選擇的模型文件夾名稱，如果為 None 則使用默認配置
    """
    global rag_engines
    
    try:
        if selected_model:
            # 使用指定的模型
            if selected_model not in rag_engines:
                logger.info(f"初始化RAG引擎 - 模型: {selected_model}")
                
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
                
                if vector_db_manager.is_training(Path(model_path)):
                    raise HTTPException(status_code=400, detail=f"模型 {selected_model} 正在訓練中")
                
                # 創建文檔索引器和RAG引擎
                document_indexer = DocumentIndexer(
                    vector_db_path=model_path,
                    ollama_embedding_model=model_info['OLLAMA_EMBEDDING_MODEL']
                )
                rag_engines[selected_model] = RAGEngine(
                    document_indexer, 
                    ollama_model=model_info['OLLAMA_MODEL']
                )
            
            return rag_engines[selected_model]
        else:
            # 使用默認配置
            if 'default' not in rag_engines:
                logger.info("初始化默認RAG引擎...")
                
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
                rag_engines['default'] = RAGEngine(document_indexer, ollama_model=default_language_model)
            
            return rag_engines['default']
    
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
            answer, sources, documents, rewritten_query = engine.get_answer_with_query_rewrite(request.question)
        else:
            answer, sources, documents = engine.get_answer_with_sources(request.question)
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



@app.get("/admin/get_indexing_log")
async def get_indexing_log(request: Request, log_type: str = "indexing"):  # log_type: indexing/reindex
    await check_admin(request)
    log_file = f"logs/{log_type}.log" if log_type != "indexing" else "logs/indexing.log"
    if not os.path.exists(log_file):
        return PlainTextResponse("(尚無日誌)", status_code=404)
    return FileResponse(log_file, filename=os.path.basename(log_file), media_type='text/plain')

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
async def monitor_all(request: Request):
    await check_admin(request)
    return {
        "status": get_status_text(),
        "progress": get_progress_text(),
        "realtime": get_monitor_text(once=True)
    }

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
