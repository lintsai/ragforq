import os
import sys
import logging
import pickle
import json
from typing import List, Dict, Any, Set, Tuple
from pathlib import Path
from tqdm import tqdm
import time
import concurrent.futures
from functools import partial
import threading

# 線程本地存儲，跟踪每個線程的COM初始化狀態
thread_local = threading.local()

# 添加項目根目錄到路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.file_parsers import FileParser
from config.config import (
    VECTOR_DB_PATH, MAX_TOKENS_CHUNK, OLLAMA_HOST, 
    EMBEDDING_BATCH_SIZE, CHUNK_OVERLAP, 
    FILE_BATCH_SIZE
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from utils.faiss_loader import get_faiss
# 確保 FAISS 正確初始化
faiss = get_faiss()
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

# 設置日誌
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 添加文件處理器，將日誌寫入到 indexing.log 文件
log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
os.makedirs(log_dir, exist_ok=True)
file_handler = logging.FileHandler(os.path.join(log_dir, "indexing.log"), encoding="utf-8")

# 設置日誌格式
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# 添加到 logger
logger.addHandler(file_handler)

# 定義問答模板 (針對 Llama 模型調整)
QA_TEMPLATE = """你是一個專業的文檔問答助手，你的任務是根據提供的上下文信息回答用戶的問題。
請僅基於下面提供的上下文信息回答問題。如果上下文中沒有足夠的信息，請直接回答「根據提供的文檔，我無法回答這個問題」。
請不要編造任何信息，不要添加上下文中沒有的內容。

上下文信息:
-----------------
{context}
-----------------

用戶問題: {question}

你的回答:"""

class DocumentIndexer:
    """文件索引器，負責解析文件內容並建立向量索引"""
    
    def __init__(self, vector_db_path: str = VECTOR_DB_PATH, ollama_embedding_model: str = None):
        """
        初始化文件索引器
        
        Args:
            vector_db_path: 向量數據庫保存路徑
            ollama_embedding_model: Ollama 嵌入模型名稱，如果為 None 則自動選擇
        """
        self.vector_db_path = vector_db_path
        self.documents = []
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=MAX_TOKENS_CHUNK,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
        )
        
        # 使用傳入的嵌入模型
        self.embeddings = OllamaEmbeddings(
            base_url=OLLAMA_HOST,
            model=ollama_embedding_model
        )
        self.ollama_embedding_model = ollama_embedding_model
            
        # 確保向量數據庫目錄存在
        os.makedirs(self.vector_db_path, exist_ok=True)
        
        # 載入已索引文件記錄
        self.indexed_files = self._load_indexed_files()
        
        # 載入索引進度記錄
        self.indexing_progress = self._load_indexing_progress()
        
        # 載入向量存儲
        self.vector_store = self._load_vector_store()
        
        logger.info(f"DocumentIndexer 初始化完成，向量存儲位置: {self.vector_db_path}")
    
    def _load_indexed_files(self) -> Dict[str, float]:
        """
        載入已索引文件記錄
        
        Returns:
            已索引文件的字典，鍵為文件路徑，值為修改時間
        """
        index_record_path = os.path.join(self.vector_db_path, "indexed_files.pkl")
        if os.path.exists(index_record_path):
            try:
                with open(index_record_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.error(f"載入索引記錄失敗: {str(e)}")
        return {}
    
    def _save_indexed_files(self):
        """保存已索引文件記錄"""
        index_record_path = os.path.join(self.vector_db_path, "indexed_files.pkl")
        try:
            with open(index_record_path, 'wb') as f:
                pickle.dump(self.indexed_files, f)
        except Exception as e:
            logger.error(f"保存索引記錄失敗: {str(e)}")

    def _load_indexing_progress(self) -> Dict[str, Any]:
        """
        載入索引進度記錄
        
        Returns:
            包含索引進度的字典，包括待索引文件列表、正在處理的批次信息等
        """
        progress_path = os.path.join(self.vector_db_path, "indexing_progress.json")
        if os.path.exists(progress_path):
            try:
                # 檢查文件是否為空
                if os.path.getsize(progress_path) == 0:
                    logger.warning(f"索引進度記錄文件為空: {progress_path}")
                    return {"pending_files": [], "current_batch": [], "completed_batches": 0, "total_batches": 0, "in_progress": False}
                
                with open(progress_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    # 確保內容不為空
                    if not content:
                        logger.warning(f"索引進度記錄文件內容為空: {progress_path}")
                        return {"pending_files": [], "current_batch": [], "completed_batches": 0, "total_batches": 0, "in_progress": False}
                    
                    try:
                        progress = json.loads(content)
                        logger.info(f"載入索引進度記錄成功，包含 {len(progress.get('pending_files', []))} 個待處理文件")
                        return progress
                    except json.JSONDecodeError as je:
                        logger.error(f"解析索引進度記錄JSON失敗: {str(je)}")
                        logger.error(f"問題文件內容: {content[:100]}...")  # 記錄文件的前100個字符用於診斷
                        # 嘗試修復或建立新文件
                        os.rename(progress_path, f"{progress_path}.bak")
                        logger.info(f"已將損壞的進度文件備份為 {progress_path}.bak")
                        return {"pending_files": [], "current_batch": [], "completed_batches": 0, "total_batches": 0, "in_progress": False}
            except Exception as e:
                logger.error(f"載入索引進度記錄失敗: {str(e)}")
                import traceback
                logger.error(f"詳細錯誤: {traceback.format_exc()}")
        
        logger.info("未找到索引進度記錄，將創建新記錄")
        return {"pending_files": [], "current_batch": [], "completed_batches": 0, "total_batches": 0, "in_progress": False}
    
    def _save_indexing_progress(self, pending_files=None, current_batch=None, completed_batches=None, total_batches=None, in_progress=None):
        """
        保存索引進度記錄
        
        Args:
            pending_files: 待處理的文件列表
            current_batch: 當前正在處理的批次
            completed_batches: 已完成的批次數
            total_batches: 總批次數
            in_progress: 是否有索引任務正在進行
        """
        progress_path = os.path.join(self.vector_db_path, "indexing_progress.json")
        try:
            # 更新傳入的值，非傳入值保持原樣
            if pending_files is not None:
                self.indexing_progress["pending_files"] = pending_files
            if current_batch is not None:
                self.indexing_progress["current_batch"] = current_batch
            if completed_batches is not None:
                self.indexing_progress["completed_batches"] = completed_batches
            if total_batches is not None:
                self.indexing_progress["total_batches"] = total_batches
            if in_progress is not None:
                self.indexing_progress["in_progress"] = in_progress
            
            # 確保路徑存在
            os.makedirs(os.path.dirname(progress_path), exist_ok=True)
            
            # 先寫入臨時文件，再重命名，避免寫入過程中發生錯誤導致文件損壞
            temp_path = f"{progress_path}.tmp"
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(self.indexing_progress, f, ensure_ascii=False, indent=2)
            
            # 安全替換文件
            if os.path.exists(progress_path):
                os.replace(temp_path, progress_path)
            else:
                os.rename(temp_path, progress_path)
                
            if pending_files is not None:
                logger.debug(f"保存索引進度記錄成功，剩餘 {len(pending_files)} 個待處理文件")
        except Exception as e:
            logger.error(f"保存索引進度記錄失敗: {str(e)}")
            import traceback
            logger.error(f"詳細錯誤: {traceback.format_exc()}")
    
    def _load_vector_store(self) -> FAISS:
        """
        載入向量存儲
        
        Returns:
            FAISS向量存儲實例
        """
        try:
            if os.path.exists(self.vector_db_path):
                # 檢索是否存在 index.faiss 和 index.pkl 文件
                index_files_exist = (
                    os.path.exists(os.path.join(self.vector_db_path, "index.faiss")) and
                    os.path.exists(os.path.join(self.vector_db_path, "index.pkl"))
                )
                
                if index_files_exist:
                    logger.info(f"正在從 {self.vector_db_path} 載入現有向量數據庫...")
                    return FAISS.load_local(self.vector_db_path, self.embeddings, allow_dangerous_deserialization=True)
            
            # 如果找不到有效的索引文件，創建一個新的空向量存儲
            logger.info("創建新的向量數據庫...")
            empty_texts = ["初始化向量數據庫"]
            return FAISS.from_texts(empty_texts, self.embeddings)
            
        except Exception as e:
            logger.error(f"載入向量數據庫失敗: {str(e)}")
            import traceback
            logger.error(f"詳細錯誤信息: {traceback.format_exc()}")
            
            # 如果加載失敗，創建一個新的空向量存儲
            logger.info("創建新的向量數據庫...")
            empty_texts = ["初始化向量數據庫"]
            return FAISS.from_texts(empty_texts, self.embeddings)
    
    def _save_vector_store(self):
        """保存向量存儲"""
        try:
            if self.vector_store:
                logger.info(f"正在將向量數據庫保存到 {self.vector_db_path}...")
                self.vector_store.save_local(self.vector_db_path)
                logger.info(f"向量數據庫已成功保存")
            else:
                logger.warning("沒有向量存儲可供保存")
        except Exception as e:
            logger.error(f"保存向量數據庫失敗: {str(e)}")
            import traceback
            logger.error(f"詳細錯誤信息: {traceback.format_exc()}")
    
    def index_file(self, file_path: str) -> bool:
        """
        索引單個文件
        
        Args:
            file_path: 要索引的文件路徑
            
        Returns:
            是否成功索引
        """
        if not os.path.exists(file_path):
            logger.error(f"文件不存在: {file_path}")
            return False
        
        # 獲取文件修改時間
        file_mtime = os.path.getmtime(file_path)
        
        # 如果文件已經索引且未更新，則跳過
        if file_path in self.indexed_files and self.indexed_files[file_path] >= file_mtime:
            logger.debug(f"文件未更新，跳過索引: {file_path}")
            return True
        
        logger.info(f"索引文件: {file_path}")
        
        try:
            # 獲取適合的解析器
            parser = FileParser.get_parser_for_file(file_path)
            if parser is None:
                logger.warning(f"不支持的文件類型或無法創建解析器: {file_path}")
                return False
            
            # 使用安全解析方法解析文件內容
            try:
                content_blocks = parser.safe_parse(file_path)
                if not content_blocks:
                    logger.warning(f"無法從文件中提取內容: {file_path}")
                    # 即使內容為空，也將文件標記為已索引以避免重複處理
                    self.indexed_files[file_path] = file_mtime
                    self._save_indexed_files()
                    return False
            except Exception as parse_error:
                logger.error(f"解析文件內容時出錯 ({file_path}): {str(parse_error)}")
                import traceback
                logger.error(f"詳細錯誤: {traceback.format_exc()}")
                # 即使解析失敗，也將文件標記為已索引以避免重複處理失敗文件
                self.indexed_files[file_path] = file_mtime
                self._save_indexed_files()
                return False
            
            # 刪除該文件之前的索引（如果存在）
            try:
                if file_path in self.indexed_files:
                    self._delete_file_from_index(file_path)
            except Exception as delete_error:
                logger.error(f"刪除舊索引時出錯 ({file_path}): {str(delete_error)}")
                # 繼續處理，不中斷索引過程
            
            # 將內容分割為塊
            try:
                all_chunks = []
                all_metadatas = []
                
                for text, metadata in content_blocks:
                    # 確保文本不為空再進行分割
                    if not text or not text.strip():
                        continue
                        
                    chunks = self.text_splitter.split_text(text)
                    
                    for i, chunk in enumerate(chunks):
                        chunk_metadata = metadata.copy()
                        chunk_metadata.update({
                            "chunk_id": i,
                            "chunk_count": len(chunks)
                        })
                        all_chunks.append(chunk)
                        all_metadatas.append(chunk_metadata)
            except Exception as split_error:
                logger.error(f"分割文本時出錯 ({file_path}): {str(split_error)}")
                # 標記文件為已處理，避免下次再次嘗試
                self.indexed_files[file_path] = file_mtime
                self._save_indexed_files()
                return False
            
            # 確保向量存儲已加載
            try:
                if not self.vector_store:
                    self.vector_store = self._load_vector_store()
            except Exception as load_error:
                logger.error(f"加載向量存儲時出錯: {str(load_error)}")
                return False
            
            # 添加到向量存儲
            try:
                if all_chunks:  # 確保有內容才添加
                    # 批量處理嵌入生成
                    batch_size = EMBEDDING_BATCH_SIZE
                    total_chunks = len(all_chunks)
                    
                    if total_chunks > 5:
                        logger.info(f"批量處理 {total_chunks} 個文本塊，批次大小: {batch_size}")
                    
                    for i in range(0, total_chunks, batch_size):
                        batch_end = min(i + batch_size, total_chunks)
                        batch_chunks = all_chunks[i:batch_end]
                        batch_metadatas = all_metadatas[i:batch_end]
                        
                        # 添加文本批次到向量存儲
                        self.vector_store.add_texts(batch_chunks, batch_metadatas)
                        
                        if total_chunks > 10 and (i + batch_size) % (batch_size * 5) == 0:
                            logger.info(f"已處理 {batch_end}/{total_chunks} 個文本塊")
                else:
                    logger.warning(f"文件無有效內容，跳過向量存儲 ({file_path})")
                    # 標記文件為已處理
                    self.indexed_files[file_path] = file_mtime
                    self._save_indexed_files()
                    return False
            except Exception as add_error:
                logger.error(f"添加到向量存儲時出錯 ({file_path}): {str(add_error)}")
                import traceback
                logger.error(f"詳細錯誤: {traceback.format_exc()}")
                # 標記文件為已處理
                self.indexed_files[file_path] = file_mtime
                self._save_indexed_files()
                return False
            
            # 更新索引記錄
            try:
                self.indexed_files[file_path] = file_mtime
                self._save_indexed_files()
            except Exception as save_error:
                logger.error(f"保存索引記錄時出錯: {str(save_error)}")
                # 繼續處理，不中斷索引過程
            
            logger.info(f"成功索引文件: {file_path}，共 {len(all_chunks)} 個文本塊")
            return True
            
        except Exception as e:
            logger.error(f"索引文件 {file_path} 時出錯: {str(e)}")
            import traceback
            logger.error(f"詳細錯誤堆疊: {traceback.format_exc()}")
            return False
    
    def index_files(self, file_paths: List[str], show_progress: bool = True) -> Tuple[int, int]:
        """
        索引多個文件 (並行處理版本) - 優化版本，支持斷點續傳和錯誤恢復
        
        Args:
            file_paths: 要索引的文件路徑列表
            show_progress: 是否顯示進度條
            
        Returns:
            成功索引的文件數量和失敗的文件數量
        """
        # 過濾不存在的文件
        valid_files = [file_path for file_path in file_paths if os.path.exists(file_path)]
        
        if not valid_files:
            logger.warning("沒有有效的文件需要索引")
            return 0, len(file_paths)
        
        # 檢查文件編碼並預處理
        valid_files = self._preprocess_files_encoding(valid_files)
        
        try:
            # 檢查是否有未完成的索引任務
            if self.indexing_progress["in_progress"] and self.indexing_progress["pending_files"]:
                resume = True
                # 合併之前未完成的文件和新的文件列表，避免重複
                all_files = set(self.indexing_progress["pending_files"]) | set(valid_files)
                valid_files = list(all_files)
                completed_batches = self.indexing_progress["completed_batches"]
                logger.info(f"檢測到未完成的索引任務，從第 {completed_batches} 批繼續處理 {len(valid_files)} 個文件")
            else:
                resume = False
                completed_batches = 0
                logger.info(f"開始索引 {len(valid_files)} 個文件")
            
            # 設置進度條
            pbar = None
            if show_progress:
                pbar = tqdm(total=len(valid_files), desc="索引文件", initial=completed_batches * 100 if resume else 0)
            
            # 使用我們的線程安全方法處理文件
            # 對文件進行優先級排序，處理較小和簡單的文件優先
            prioritized_files = self._prioritize_files(valid_files)
            
            # 使用批次處理以避免內存問題
            batch_size = FILE_BATCH_SIZE  # 使用配置的批次大小而非硬編碼
            total_files = len(prioritized_files)
            total_batches = (total_files + batch_size - 1) // batch_size
            success_count = 0
            fail_count = 0
            
            # 標記索引任務開始
            self._save_indexing_progress(pending_files=valid_files, 
                                        completed_batches=completed_batches, 
                                        total_batches=total_batches, 
                                        in_progress=True)
        except Exception as e:
            logger.error(f"初始化索引任務時出錯: {str(e)}")
            import traceback
            logger.error(f"詳細錯誤: {traceback.format_exc()}")
            return 0, len(file_paths)
        
        try:
            for i in range(completed_batches * batch_size, total_files, batch_size):
                batch_end = min(i + batch_size, total_files)
                batch = valid_files[i:batch_end]
                
                # 更新當前處理的批次信息
                current_batch_idx = i // batch_size
                logger.info(f"開始處理批次 {current_batch_idx + 1}/{total_batches}, 包含 {len(batch)} 個文件")
                self._save_indexing_progress(pending_files=valid_files[batch_end:], 
                                            current_batch=batch, 
                                            completed_batches=current_batch_idx)
                
                # 處理一批文件
                self.parallel_index_files(batch)
                
                # 更新進度條
                if pbar:
                    pbar.update(len(batch))
                    
                # 計算成功和失敗的數量
                for file_path in batch:
                    if file_path in self.indexed_files:
                        success_count += 1
                    else:
                        fail_count += 1
                
                # 完成這一批次的處理，更新進度
                self._save_indexing_progress(completed_batches=current_batch_idx + 1, 
                                           current_batch=[])
                
                # 定期保存索引記錄
                self._save_indexed_files()
                
                # 保存向量存儲
                self._save_vector_store()
                
                logger.info(f"已完成批次 {current_batch_idx + 1}/{total_batches}，處理了 {batch_end}/{total_files} 個文件")
            
            # 所有批次處理完成，標記索引任務結束
            self._save_indexing_progress(pending_files=[], 
                                       current_batch=[], 
                                       in_progress=False)
            
        except Exception as e:
            logger.error(f"索引處理中斷: {str(e)}")
            import traceback
            logger.error(f"詳細錯誤: {traceback.format_exc()}")
            # 保存當前進度，以便下次繼續
            if not pbar:
                logger.info("索引過程中斷，進度已保存，下次可繼續執行")
        finally:
            if pbar:
                pbar.close()
                
        logger.info(f"索引完成：{success_count} 個成功，{fail_count} 個失敗")
        return success_count, fail_count
    
    def _preprocess_files_encoding(self, file_paths: List[str]) -> List[str]:
        """
        預處理文件編碼，過濾掉無法處理的文件
        
        Args:
            file_paths: 文件路徑列表
            
        Returns:
            可處理的文件路徑列表
        """
        valid_files = []
        for file_path in file_paths:
            try:
                # 檢查文件是否可讀
                if not os.access(file_path, os.R_OK):
                    logger.warning(f"文件無讀取權限，跳過: {file_path}")
                    continue
                
                # 檢查文件大小（跳過過大的文件）
                file_size = os.path.getsize(file_path)
                if file_size > 100 * 1024 * 1024:  # 100MB
                    logger.warning(f"文件過大 ({file_size/1024/1024:.1f}MB)，跳過: {file_path}")
                    continue
                
                # 對於文本文件，檢查編碼
                file_ext = os.path.splitext(file_path)[1].lower()
                if file_ext in ['.txt', '.md', '.csv']:
                    if not self._check_text_file_encoding(file_path):
                        logger.warning(f"文件編碼問題，跳過: {file_path}")
                        continue
                
                valid_files.append(file_path)
                
            except Exception as e:
                logger.error(f"預處理文件時出錯 {file_path}: {str(e)}")
                continue
        
        logger.info(f"預處理完成，{len(valid_files)}/{len(file_paths)} 個文件可處理")
        return valid_files
    
    def _check_text_file_encoding(self, file_path: str) -> bool:
        """
        檢查文本文件編碼是否可處理
        
        Args:
            file_path: 文件路徑
            
        Returns:
            是否可處理
        """
        encodings = ['utf-8', 'big5', 'gb18030', 'latin-1', 'cp1252']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding, errors='strict') as f:
                    # 讀取前1KB檢查編碼
                    f.read(1024)
                return True
            except (UnicodeDecodeError, UnicodeError):
                continue
            except Exception:
                return False
        
        return False
    
    def _delete_file_from_index(self, file_path: str) -> bool:
        """
        從索引中刪除文件
        
        Args:
            file_path: 要刪除的文件路徑
            
        Returns:
            是否成功刪除
        """
        if not self.vector_store:
            self.vector_store = self._load_vector_store()
        
        try:
            # 檢索所有向量，篩選出不屬於該文件的向量
            all_docs = self.vector_store.docstore._dict
            docs_to_keep = []
            ids_to_keep = []
            
            for doc_id, doc in all_docs.items():
                if doc.metadata.get("file_path") != file_path:
                    docs_to_keep.append(doc)
                    ids_to_keep.append(doc_id)
            
            # 創建新的向量存儲
            new_vector_store = FAISS.from_documents(docs_to_keep, self.embeddings)
            self.vector_store = new_vector_store
            
            # 從索引記錄中刪除
            if file_path in self.indexed_files:
                del self.indexed_files[file_path]
                self._save_indexed_files()
            
            logger.info(f"已從索引中刪除文件: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"從索引中刪除文件 {file_path} 時出錯: {str(e)}")
            return False
    
    def delete_files(self, file_paths: List[str]) -> int:
        """
        從索引中刪除多個文件
        
        Args:
            file_paths: 要刪除的文件路徑列表
            
        Returns:
            成功刪除的文件數量
        """
        success_count = 0
        
        for file_path in file_paths:
            if self._delete_file_from_index(file_path):
                success_count += 1
        
        # 保存向量存儲
        self._save_vector_store()
        
        logger.info(f"刪除完成: {success_count} 個文件已從索引中刪除")
        return success_count
    
    def get_vector_store(self) -> FAISS:
        """
        獲取向量存儲實例
        
        Returns:
            FAISS向量存儲實例
        """
        if not self.vector_store:
            self.vector_store = self._load_vector_store()
        return self.vector_store
    
    def process_file_changes(self, new_files: Set[str], updated_files: Set[str], deleted_files: Set[str]):
        """
        處理文件變更
        
        Args:
            new_files: 新文件集合
            updated_files: 更新文件集合
            deleted_files: 刪除文件集合
        """
        # 處理新文件和更新文件
        files_to_index = list(new_files) + list(updated_files)
        if files_to_index:
            logger.info(f"索引 {len(files_to_index)} 個新文件或更新文件")
            self.index_files(files_to_index)
        
        # 處理刪除文件
        if deleted_files:
            logger.info(f"從索引中刪除 {len(deleted_files)} 個文件")
            self.delete_files(list(deleted_files))

    def retrieve_documents(self, query: str, top_k: int = 5) -> List[Document]:
        """
        檢索相關文檔
        
        Args:
            query: 查詢文本
            top_k: 返回的文檔數量
            
        Returns:
            相關文檔列表
        """
        try:
            if not self.vector_store:
                self.vector_store = self._load_vector_store()
                logger.info("已加載向量數據庫")
            
            # 檢查向量存儲中的文檔數量
            doc_count = len(self.vector_store.docstore._dict)
            logger.info(f"向量數據庫中共有 {doc_count} 個文檔")
            
            if doc_count == 0:
                logger.warning("向量數據庫為空，請先添加文檔")
                return []
            
            # 使用相似度搜索
            docs_with_scores = self.vector_store.similarity_search_with_score(
                query,
                k=top_k
            )
            
            logger.info(f"找到 {len(docs_with_scores)} 個相關文檔")
            
            # 提取文檔並記錄相似度分數
            results = []
            for doc, score in docs_with_scores:
                logger.debug(f"文檔分數: {score}, 來源: {doc.metadata.get('file_path', '未知')}")
                results.append(doc)
            
            if not results:
                logger.warning(f"未找到與查詢 '{query}' 相關的文檔")
                return []
            
            return results
            
        except Exception as e:
            logger.error(f"檢索文檔時出錯: {str(e)}")
            import traceback
            logger.error(f"詳細錯誤: {traceback.format_exc()}")
            return []

    def add_document(self, file_path: str) -> None:
        """
        添加文件到索引
        
        Args:
            file_path: 文件路徑
        """
        try:
            # 使用批處理方式添加文件以提高效率
            self.index_file(file_path)
            # 每添加一個文件就保存一次索引
            self._save_vector_store()
        except Exception as e:
            logger.error(f"添加文件 {file_path} 到索引時出錯: {str(e)}")
            raise
    
    def save(self) -> None:
        """保存向量數據庫到磁盤"""
        try:
            if not self.documents:
                logger.warning("沒有文件被添加到索引中")
                if self.vector_store:
                    # 如果已有向量存儲，仍然保存索引記錄
                    self._save_indexed_files()
                    logger.info("已保存索引記錄")
                return
                
            # 創建向量存儲
            logger.info(f"從 {len(self.documents)} 個文檔創建向量存儲...")
            self.vector_store = FAISS.from_documents(
                documents=self.documents,
                embedding=self.embeddings
            )
            
            # 保存到磁盤
            logger.info("保存向量存儲...")
            self.vector_store.save_local(self.vector_db_path)
            
            # 保存索引記錄
            self._save_indexed_files()
            
            logger.info(f"向量數據庫和索引記錄已成功保存到 {self.vector_db_path}")
            
            # 清空文檔列表以節省內存
            self.documents = []
            
        except Exception as e:
            logger.error(f"保存向量數據庫時出錯: {str(e)}")
            import traceback
            logger.error(f"詳細錯誤信息: {traceback.format_exc()}")
            raise

    def list_indexed_files(self) -> List[Dict[str, Any]]:
        """
        列出所有已索引的文件及其相關信息
        
        Returns:
            包含文件信息的字典列表，每個字典包含文件路徑和最後修改時間
        """
        try:
            if not self.indexed_files:
                logger.info("目前沒有已索引的文件")
                return []
            
            # 獲取文件信息
            files_info = []
            for file_path, mtime in self.indexed_files.items():
                try:
                    # 檢查文件是否仍然存在
                    if os.path.exists(file_path):
                        file_info = {
                            "file_path": file_path,
                            "last_modified": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mtime)),
                            "file_size": os.path.getsize(file_path),
                            "file_type": os.path.splitext(file_path)[1][1:].upper() or "未知"
                        }
                        files_info.append(file_info)
                    else:
                        logger.warning(f"索引文件已不存在: {file_path}")
                except Exception as e:
                    logger.error(f"獲取文件 {file_path} 信息時出錯: {str(e)}")
                    continue
            
            # 按修改時間排序
            files_info.sort(key=lambda x: x["last_modified"], reverse=True)
            
            logger.info(f"找到 {len(files_info)} 個已索引文件")
            return files_info
            
        except Exception as e:
            logger.error(f"列出索引文件時出錯: {str(e)}")
            import traceback
            logger.error(f"詳細錯誤信息: {traceback.format_exc()}")
            return []

    def _process_file_with_com_init(self, file_path):
        """
        確保每個線程中COM正確初始化的文件處理包裝器
        
        Args:
            file_path: 文件路徑
        
        Returns:
            處理結果
        """
        # 檢查當前線程是否已初始化COM
        if not hasattr(thread_local, 'com_initialized'):
            try:
                import pythoncom
                pythoncom.CoInitialize()
                thread_local.com_initialized = True
                logger.debug(f"線程 {threading.current_thread().name} 初始化COM成功")
            except ImportError:
                logger.debug("未找到pythoncom模塊，跳過COM初始化")
                thread_local.com_initialized = False
            except Exception as e:
                logger.warning(f"線程 {threading.current_thread().name} COM初始化失敗: {str(e)}")
                thread_local.com_initialized = False
                
        # 處理文件
        result = self.index_file(file_path)
        
        return result

    def parallel_index_files(self, file_paths):
        """
        並行索引多個文件 - 優化版本，減少內存使用
        
        Args:
            file_paths: 文件路徑列表
        """
        # 根據系統資源動態調整線程數
        try:
            import psutil
            memory = psutil.virtual_memory()
            # 如果內存使用率超過70%，減少線程數
            if memory.percent > 70:
                max_workers = min(4, len(file_paths))
            else:
                max_workers = min(8, len(file_paths))  # 減少最大線程數
        except ImportError:
            max_workers = min(8, len(file_paths))
        
        if max_workers <= 0:
            max_workers = 1
        
        logger.info(f"使用 {max_workers} 個工作線程進行並行索引處理")
        
        # 使用線程池處理，但限制並發數
        success_count = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 分批提交任務，避免一次性提交太多任務
            batch_size = max_workers * 2
            for i in range(0, len(file_paths), batch_size):
                batch = file_paths[i:i + batch_size]
                
                # 提交批次任務
                future_to_file = {
                    executor.submit(self._process_file_with_com_init, file_path): file_path 
                    for file_path in batch
                }
                
                # 處理完成的任務
                for future in concurrent.futures.as_completed(future_to_file):
                    file_path = future_to_file[future]
                    try:
                        result = future.result()
                        if result:
                            success_count += 1
                    except Exception as e:
                        logger.error(f"處理文件 {file_path} 時出錯: {str(e)}")
                
                # 每批次後進行垃圾回收
                import gc
                gc.collect()
        
        logger.info(f"已完成 {len(file_paths)} 個文件的索引，其中 {success_count} 個成功")

    def index_batch(self, file_paths: List[str], force_reindex: bool = False) -> None:
        """
        索引一批文件
        
        Args:
            file_paths: 要索引的文件路徑列表
            force_reindex: 是否強制重新索引已索引的文件
        """
        # 過濾不存在的文件
        valid_files = [file_path for file_path in file_paths if os.path.exists(file_path)]
        
        if not valid_files:
            logger.warning("沒有有效的文件需要索引")
            return
        
        # 如果不強制重新索引，過濾出未更新的文件
        if not force_reindex:
            files_to_index = []
            for file_path in valid_files:
                file_mtime = os.path.getmtime(file_path)
                if file_path not in self.indexed_files or self.indexed_files[file_path] < file_mtime:
                    files_to_index.append(file_path)
        else:
            # 強制重新索引全部文件
            files_to_index = valid_files
        
        if not files_to_index:
            logger.info("所有文件都已索引且未更新，無需重新索引")
            return
        
        # 使用我們的線程安全COM初始化方法處理文件
        self.parallel_index_files(files_to_index)
        
        # 保存索引記錄
        self._save_indexed_files()

    def _prioritize_files(self, file_paths: List[str]) -> List[str]:
        """
        對文件進行優先級排序，使較小的文件先處理
        
        Args:
            file_paths: 原始文件路徑列表
            
        Returns:
            按優先級排序的文件路徑列表
        """
        # 收集文件信息用於排序
        file_info = []
        for path in file_paths:
            try:
                if os.path.exists(path):
                    # 獲取文件大小和擴展名
                    size = os.path.getsize(path)
                    ext = os.path.splitext(path)[1].lower()
                    
                    # 為不同類型的文件分配複雜度評分
                    complexity_score = 1.0
                    if ext in ['.pdf']:
                        complexity_score = 2.0  # PDF通常很複雜
                    elif ext in ['.docx', '.xlsx', '.pptx']:
                        complexity_score = 1.8  # Office文件通常較複雜
                    elif ext in ['.txt', '.md', '.csv']:
                        complexity_score = 0.8  # 文本文件通常較簡單
                    
                    # 結合文件大小和複雜度得出優先級分數
                    # 越小的分數越優先處理
                    priority_score = size * complexity_score
                    
                    file_info.append((path, priority_score))
            except Exception as e:
                # 如果無法獲取文件信息，給予較低優先級
                logger.warning(f"無法獲取文件信息 {path}: {str(e)}")
                file_info.append((path, float('inf')))  # 使用無窮大表示最低優先級
        
        # 按照優先級分數排序（從小到大）
        file_info.sort(key=lambda x: x[1])
        
        # 提取排序後的文件路徑
        prioritized_paths = [info[0] for info in file_info]
        
        return prioritized_paths
        
    def check_indexing_status(self) -> Dict[str, Any]:
        """
        檢查索引狀態
        
        Returns:
            包含索引狀態的字典，包括是否有進行中的任務、完成百分比、已完成文件數等
        """
        try:
            if not self.indexing_progress["in_progress"]:
                return {
                    "in_progress": False,
                    "message": "沒有正在進行的索引任務",
                    "completed_percentage": 100,
                    "indexed_file_count": len(self.indexed_files)
                }
                
            total_files = len(self.indexing_progress["pending_files"]) + len(self.indexing_progress["current_batch"])
            pending_files = len(self.indexing_progress["pending_files"])
            processed_files = total_files - pending_files
            
            if total_files > 0:
                completed_percentage = (processed_files / total_files) * 100
            else:
                completed_percentage = 100
                
            return {
                "in_progress": True,
                "message": f"索引任務進行中，已完成 {completed_percentage:.2f}%",
                "total_files": total_files,
                "pending_files": pending_files,
                "processed_files": processed_files,
                "completed_percentage": completed_percentage,
                "completed_batches": self.indexing_progress["completed_batches"],
                "total_batches": self.indexing_progress["total_batches"],
                "indexed_file_count": len(self.indexed_files)
            }
        except Exception as e:
            logger.error(f"檢查索引狀態時出錯: {str(e)}")
            return {
                "in_progress": False,
                "message": f"檢查索引狀態時出錯: {str(e)}",
                "error": True
            }

if __name__ == "__main__":
    # 測試代碼
    import tempfile
    
    with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", encoding="utf-8", delete=False) as f:
        f.write("這是測試文件內容，用於測試索引器。")
        test_file = f.name
    
    try:
        indexer = DocumentIndexer()
        success = indexer.index_file(test_file)
        print(f"索引結果: {'成功' if success else '失敗'}")
        
        # 測試查詢
        vector_store = indexer.get_vector_store()
        results = vector_store.similarity_search("測試", k=1)
        print("查詢結果:", results[0].page_content if results else "無結果")
        
    finally:
        os.unlink(test_file)
