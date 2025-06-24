import os
import sys
import logging
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Body, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# 添加項目根目錄到路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import APP_HOST, APP_PORT, is_q_drive_accessible
from rag_engine.rag_engine import RAGEngine
from indexer.document_indexer import DocumentIndexer
from langchain_core.documents import Document

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

# 初始化RAG引擎
rag_engine = None

def get_rag_engine():
    """獲取或初始化RAG引擎"""
    global rag_engine
    if rag_engine is None:
        logger.info("初始化RAG引擎...")
        document_indexer = DocumentIndexer()
        rag_engine = RAGEngine(document_indexer)
    return rag_engine

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
        engine = get_rag_engine()
        
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
        document_indexer = DocumentIndexer()
        indexed_files = document_indexer.list_indexed_files()
        return indexed_files
    except Exception as e:
        logger.error(f"獲取文件列表時出錯: {str(e)}")
        raise HTTPException(status_code=500, detail=f"獲取文件列表時出錯: {str(e)}")

# 啟動應用
if __name__ == "__main__":
    uvicorn.run("main:app", host=APP_HOST, port=APP_PORT, reload=True) 