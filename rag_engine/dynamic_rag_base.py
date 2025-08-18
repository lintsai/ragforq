"""
Dynamic RAG Engine Base - 動態檢索增強生成引擎的基底類別
"""

import os
import logging
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib

from langchain_core.documents import Document
from utils.hf_langchain_wrapper import HuggingFaceEmbeddings, ChatHuggingFace
from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np

from .interfaces import RAGEngineInterface
from utils.file_parsers import FileParser
from config.config import (
    MAX_TOKENS_CHUNK, CHUNK_OVERLAP,
    SUPPORTED_FILE_TYPES, Q_DRIVE_PATH,
    OLLAMA_REQUEST_TIMEOUT
)

logger = logging.getLogger(__name__)

class SmartFileRetriever:
    """智能文件檢索器 - 根據查詢內容智能選擇相關文件"""
    
    def __init__(self, base_path: str = Q_DRIVE_PATH, folder_path: Optional[str] = None):
        self.base_path = base_path
        self.folder_path = folder_path  # 指定的文件夾路徑過濾
        self.file_cache = {}  # 文件元數據緩存
        self.last_scan_time = 0
        self.cache_duration = 300  # 5分鐘緩存
        self._file_count_warning = None
    
    def retrieve_relevant_files(self, query: str, max_files: int = 10) -> List[str]:
        """
        根據查詢檢索相關文件
        
        Args:
            query: 用戶查詢
            max_files: 最大返回文件數
            
        Returns:
            相關文件路徑列表
        """
        # 更新文件緩存
        self._update_file_cache()

        # 1. 首先根據文件夾路徑過濾文件緩存
        working_file_cache = self.file_cache
        if self.folder_path:
            # 使用更可靠的路徑比較方法
            try:
                target_path = Path(self.base_path, self.folder_path).resolve()
                logger.info(f"限制搜索範圍於: {target_path}")
                
                filtered_cache = {
                    fp: meta for fp, meta in self.file_cache.items()
                    if Path(fp).resolve().is_relative_to(target_path)
                }
                
                logger.info(f"文件夾限制 '{self.folder_path}' 前有 {len(working_file_cache)} 個文件，過濾後剩餘 {len(filtered_cache)} 個文件")
                working_file_cache = filtered_cache
            except Exception as e:
                logger.error(f"處理文件夾路徑 '{self.folder_path}' 時出錯: {e}")
                # 如果路徑處理失敗，則不進行過濾，但記錄錯誤
                pass

            if not working_file_cache:
                logger.warning(f"在指定文件夾 '{self.folder_path}' 中沒有找到任何支持的文件")
                return []

        # 2. 檢查文件數量並設置警告 (無論是否有文件夾限制)
        file_count = len(working_file_cache)
        if file_count > 3000:
            warning_message = f"檢測到處理文件數量過多 ({file_count} 個)，可能影響處理速度。建議縮小搜索範圍。"
            logger.warning(warning_message)
            self._file_count_warning = warning_message
        else:
            self._file_count_warning = None

        # 3. 如果過濾後的文件總數很少，則直接返回所有文件
        if file_count <= 50:
            logger.info(f"找到 {file_count} 個文件，將處理全部文件。")
            return list(working_file_cache.keys())
        
        # 4. 在過濾後的文件中進行搜索
        # 創建一個臨時的檢索器實例來處理當前請求，避免污染共享緩存
        temp_retriever = self.__class__()
        temp_retriever.file_cache = working_file_cache
        
        # 關鍵詞匹配
        keyword_matches = temp_retriever._match_by_keywords(query)
        
        # 路徑語義分析
        path_matches = temp_retriever._analyze_file_paths(query)
        
        # 合併和去重
        all_matches = list(set(keyword_matches + path_matches))
        
        # 評分和排序
        scored_files = temp_retriever._score_files(query, all_matches)
        
        # 返回前N個文件
        relevant_files = [file_path for file_path, score in scored_files[:max_files]]
        
        logger.info(f"在 {file_count} 個文件中找到 {len(relevant_files)} 個相關文件")
        return relevant_files
    
    def _quick_estimate_file_count(self, scan_path: str, max_sample_dirs: int = 20) -> int:
        """快速估算文件數量，避免完整掃描"""
        try:
            # 採樣估算法：只掃描部分目錄來估算總數
            sample_count = 0
            total_dirs = 0
            sampled_dirs = 0
            
            # 第一層目錄採樣
            for root, dirs, files in os.walk(scan_path):
                # 只處理前幾層目錄
                depth = root.replace(scan_path, '').count(os.sep)
                if depth > 3:  # 限制深度
                    dirs[:] = []
                    continue
                
                total_dirs += 1
                if sampled_dirs < max_sample_dirs:
                    # 計算當前目錄的支持文件數量
                    supported_files = sum(1 for f in files if os.path.splitext(f)[1].lower() in SUPPORTED_FILE_TYPES)
                    sample_count += supported_files
                    sampled_dirs += 1
                
                # 跳過系統目錄
                dirs[:] = [d for d in dirs if not d.startswith('.') and d.lower() not in ['system', 'temp', 'tmp', '$recycle.bin']]
            
            # 根據採樣結果估算總數
            if sampled_dirs > 0:
                avg_files_per_dir = sample_count / sampled_dirs
                estimated_total = int(avg_files_per_dir * total_dirs)
                logger.info(f"快速估算：採樣 {sampled_dirs} 個目錄，平均每目錄 {avg_files_per_dir:.1f} 個文件，估算總數 {estimated_total}")
                return estimated_total
            
            return 0
            
        except Exception as e:
            logger.error(f"快速估算文件數量失敗: {str(e)}")
            return 0

    def _update_file_cache(self):
        """更新文件緩存 - 性能優化版本"""
        current_time = time.time()
        if current_time - self.last_scan_time < self.cache_duration:
            return
        
        # 確定掃描路徑
        scan_path = self.base_path
        logger.info(f"更新文件緩存，基礎路徑: {scan_path}")
        
        # 快速估算文件數量
        estimated_count = self._quick_estimate_file_count(scan_path)
        
        # 根據估算結果決定掃描策略
        if estimated_count > 10000:
            logger.warning(f"估算文件數量過多 ({estimated_count} 個)，使用快速掃描模式")
            self._file_count_warning = f"檢測到大量文件 (估算約 {estimated_count} 個)，建議縮小搜索範圍以提高檢索效率"
            # 使用更激進的限制
            max_files = 2000
            max_depth = 5
        elif estimated_count > 5000:
            logger.info(f"估算文件數量較多 ({estimated_count} 個)，使用中等掃描模式")
            max_files = 3000
            max_depth = 6
        else:
            logger.info(f"估算文件數量適中 ({estimated_count} 個)，使用完整掃描模式")
            max_files = 5000
            max_depth = 8
        
        self.file_cache = {}
        
        try:
            
            file_count = 0
            
            # 使用更高效的掃描策略
            self._scan_directories_efficiently(scan_path, max_files, max_depth)
            
            self.last_scan_time = current_time
            logger.info(f"文件緩存更新完成，共 {len(self.file_cache)} 個文件")
            
        except Exception as e:
            logger.error(f"更新文件緩存失敗: {str(e)}")
            # 如果掃描失敗，至少嘗試掃描根目錄的前50個文件
            self._fallback_scan(scan_path)

    def _scan_directories_efficiently(self, scan_path: str, max_files: int, max_depth: int):
        """高效掃描目錄 - 使用批量處理和優化策略"""
        file_count = 0
        
        try:
            # 使用生成器避免一次性加載所有文件
            for root, dirs, files in os.walk(scan_path):
                # 計算當前深度
                depth = root.replace(scan_path, '').count(os.sep)
                    
                if depth >= max_depth:
                    dirs[:] = []  # 不再深入子目錄
                    continue
                
                # 跳過系統目錄和隱藏目錄
                dirs[:] = [d for d in dirs if not d.startswith('.') and d.lower() not in ['system', 'temp', 'tmp', '$recycle.bin']]
                
                # 批量處理文件，避免逐個處理
                supported_files = [f for f in files[:300] if os.path.splitext(f)[1].lower() in SUPPORTED_FILE_TYPES 
                                 and not f.startswith('.') and not f.startswith('~') and not f.lower().endswith('.tmp')]
                
                # 使用批量stat操作
                for file in supported_files:
                    if file_count >= max_files:
                        logger.info(f"達到文件數量限制 ({max_files})，停止掃描")
                        return
                    
                    file_path = os.path.join(root, file)
                    
                    try:
                        # 使用更快的文件大小檢查
                        file_size = os.path.getsize(file_path)
                        if file_size < 100:  # 跳過過小的文件
                            continue
                        
                        # 只在需要時獲取完整stat信息
                        stat_info = os.stat(file_path)
                        
                        # 計算相對路徑
                        display_path = os.path.relpath(file_path, self.base_path).replace('\\', '/')
                        
                        self.file_cache[file_path] = {
                            'name': file,
                            'size': file_size,
                            'mtime': stat_info.st_mtime,
                            'ext': os.path.splitext(file)[1].lower(),
                            'relative_path': display_path,
                            'depth': depth,
                            'folder_limited': bool(self.folder_path)
                        }
                        file_count += 1
                        
                    except (OSError, PermissionError):
                        continue
                
                # 每處理1000個文件記錄一次進度
                if file_count > 0 and file_count % 1000 == 0:
                    logger.info(f"已掃描 {file_count} 個文件...")
                    
        except Exception as e:
            logger.error(f"高效掃描失敗: {str(e)}")

    def _fallback_scan(self, scan_path: str):
        """回退掃描方法 - 只掃描根目錄"""
        try:
            logger.info("使用回退掃描方法...")
            files = os.listdir(scan_path)[:100]  # 只掃描前100個項目
            
            for file in files:
                file_path = os.path.join(scan_path, file)
                if os.path.isfile(file_path):
                    file_ext = os.path.splitext(file)[1].lower()
                    if file_ext in SUPPORTED_FILE_TYPES:
                        try:
                            stat_info = os.stat(file_path)
                            if stat_info.st_size >= 100:  # 跳過過小的文件
                                self.file_cache[file_path] = {
                                    'name': file,
                                    'size': stat_info.st_size,
                                    'mtime': stat_info.st_mtime,
                                    'ext': file_ext,
                                    'relative_path': file,
                                    'depth': 0,
                                    'folder_limited': bool(self.folder_path)
                                }
                        except (OSError, PermissionError):
                            continue
            
            logger.info(f"回退掃描完成，找到 {len(self.file_cache)} 個文件")
            
        except Exception as e:
            logger.error(f"回退掃描也失敗: {str(e)}")
            self.file_cache = {}
            # 如果掃描失敗，至少嘗試掃描根目錄
            try:
                logger.info("嘗試僅掃描根目錄...")
                for file in os.listdir(self.base_path)[:50]:  # 只掃描根目錄的前50個文件
                    file_path = os.path.join(self.base_path, file)
                    if os.path.isfile(file_path):
                        file_ext = os.path.splitext(file)[1].lower()
                        if file_ext in SUPPORTED_FILE_TYPES:
                            try:
                                stat = os.stat(file_path)
                                self.file_cache[file_path] = {
                                    'name': file,
                                    'size': stat.st_size,
                                    'mtime': stat.st_mtime,
                                    'ext': file_ext,
                                    'relative_path': file
                                }
                            except (OSError, PermissionError):
                                continue
                logger.info(f"根目錄掃描完成，找到 {len(self.file_cache)} 個文件")
            except Exception as fallback_error:
                logger.error(f"根目錄掃描也失敗: {str(fallback_error)}")
                # 如果完全失敗，創建一個空緩存
                self.file_cache = {}
    
    def _match_by_keywords(self, query: str) -> List[str]:
        """基於關鍵詞匹配文件"""
        query_lower = query.lower()
        query_words = query_lower.split()
        
        matches = []
        for file_path, metadata in self.file_cache.items():
            file_name_lower = metadata['name'].lower()
            path_lower = metadata['relative_path'].lower()
            
            # 計算匹配分數
            score = 0
            for word in query_words:
                # 完整詞匹配
                if word in file_name_lower:
                    score += 2  # 文件名匹配權重更高
                if word in path_lower:
                    score += 1  # 路徑匹配
                
                # 部分詞匹配（更寬鬆的匹配）
                for file_word in file_name_lower.split():
                    if word in file_word or file_word in word:
                        score += 1
            
            # 如果沒有關鍵詞匹配，但查詢很短，則包含所有文件
            if score == 0 and len(query_words) <= 2:
                score = 0.1  # 給一個很小的分數
            
            if score > 0:
                matches.append((file_path, score))
        
        # 按分數排序
        matches.sort(key=lambda x: x[1], reverse=True)
        return [file_path for file_path, score in matches]
    
    def _analyze_file_paths(self, query: str) -> List[str]:
        """基於路徑語義分析匹配文件"""
        # 簡化版本：基於路徑關鍵詞
        path_keywords = {
            '政策': ['policy', 'regulation', '政策', '規定'],
            '流程': ['process', 'procedure', '流程', '程序'],
            '手冊': ['manual', 'handbook', '手冊', '指南'],
            '報告': ['report', '報告', '分析'],
            '合約': ['contract', 'agreement', '合約', '協議']
        }
        
        matches = []
        query_lower = query.lower()
        
        for category, keywords in path_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                for file_path, metadata in self.file_cache.items():
                    path_lower = metadata['relative_path'].lower()
                    if any(keyword in path_lower for keyword in keywords):
                        matches.append(file_path)
        
        return matches
    
    def _score_files(self, query: str, file_paths: List[str]) -> List[Tuple[str, float]]:
        """為文件評分並排序"""
        scored_files = []
        
        for file_path in file_paths:
            if file_path not in self.file_cache:
                continue
                
            metadata = self.file_cache[file_path]
            score = 0
            
            # 文件名相關性
            file_name_lower = metadata['name'].lower()
            query_lower = query.lower()
            
            # 關鍵詞匹配分數
            query_words = query_lower.split()
            for word in query_words:
                if word in file_name_lower:
                    score += 2
                if word in metadata['relative_path'].lower():
                    score += 1
            
            # 文件類型權重
            type_weights = {
                '.pdf': 1.2,
                '.docx': 1.1,
                '.txt': 1.0,
                '.md': 1.0,
                '.xlsx': 0.9,
                '.pptx': 0.8
            }
            score *= type_weights.get(metadata['ext'], 1.0)
            
            # 文件大小權重（適中大小的文件可能包含更多有用信息）
            size_kb = metadata['size'] / 1024
            if 10 <= size_kb <= 1000:  # 10KB - 1MB
                score *= 1.1
            elif size_kb > 5000:  # 大於5MB的文件降權
                score *= 0.8
            
            # 最近修改時間權重
            days_old = (time.time() - metadata['mtime']) / (24 * 3600)
            if days_old < 30:  # 30天內的文件
                score *= 1.1
            elif days_old > 365:  # 超過一年的文件
                score *= 0.9
            
            scored_files.append((file_path, score))
        
        # 按分數排序
        scored_files.sort(key=lambda x: x[1], reverse=True)
        return scored_files


class DynamicContentProcessor:
    """動態內容處理器 - 即時解析和處理文件內容"""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=MAX_TOKENS_CHUNK,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
        )
        self.content_cache = {}  # 內容緩存
        self.cache_duration = 600  # 10分鐘緩存
    
    def process_files(self, file_paths: List[str]) -> List[Document]:
        """
        並行處理多個文件
        
        Args:
            file_paths: 文件路徑列表
            
        Returns:
            處理後的文檔列表
        """
        documents = []
        
        # 使用線程池並行處理
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_file = {
                executor.submit(self._process_single_file, file_path): file_path 
                for file_path in file_paths
            }
            
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    file_documents = future.result()
                    documents.extend(file_documents)
                except Exception as e:
                    logger.error(f"處理文件 {file_path} 失敗: {str(e)}")
        
        return documents
    
    def _process_single_file(self, file_path: str) -> List[Document]:
        """處理單個文件"""
        # 檢查緩存
        cache_key = self._get_cache_key(file_path)
        if cache_key in self.content_cache:
            cache_entry = self.content_cache[cache_key]
            if time.time() - cache_entry['timestamp'] < self.cache_duration:
                return cache_entry['documents']
        
        try:
            # 獲取解析器
            parser = FileParser.get_parser_for_file(file_path)
            if not parser:
                logger.warning(f"無法獲取文件解析器: {file_path}")
                return []
            
            # 解析文件內容
            content_blocks = parser.safe_parse(file_path)
            if not content_blocks:
                return []
            
            # 創建文檔
            documents = []
            for text, metadata in content_blocks:
                if not text or not text.strip():
                    continue
                
                # 分割文本
                chunks = self.text_splitter.split_text(text)
                
                for i, chunk in enumerate(chunks):
                    chunk_metadata = metadata.copy()
                    chunk_metadata.update({
                        "chunk_id": i,
                        "chunk_count": len(chunks),
                        "file_path": file_path
                    })
                    
                    doc = Document(
                        page_content=chunk,
                        metadata=chunk_metadata
                    )
                    documents.append(doc)
            
            # 緩存結果
            self.content_cache[cache_key] = {
                'documents': documents,
                'timestamp': time.time()
            }
            
            return documents
            
        except Exception as e:
            logger.error(f"處理文件 {file_path} 時出錯: {str(e)}")
            return []
    
    def _get_cache_key(self, file_path: str) -> str:
        """生成緩存鍵"""
        try:
            stat = os.stat(file_path)
            content = f"{file_path}_{stat.st_mtime}_{stat.st_size}"
            return hashlib.md5(content.encode()).hexdigest()
        except OSError:
            return hashlib.md5(file_path.encode()).hexdigest()


class RealTimeVectorizer:
    """即時向量化引擎"""
    
    def __init__(self, embedding_model: str, platform: str = "ollama"):
        # 根據平台選擇嵌入模型
        if platform == "ollama":
            from utils.ollama_embeddings import OllamaEmbeddings
            self.embeddings = OllamaEmbeddings(model_name=embedding_model)
            logger.info(f"使用 Ollama 嵌入模型: {embedding_model}")
        else: # 默認為 huggingface
            self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
            logger.info(f"使用 Hugging Face 嵌入模型: {embedding_model}")
        
        self.query_cache = {}  # 查詢向量緩存
        self.cache_duration = 1800  # 30分鐘緩存
    
    def vectorize_query(self, query: str) -> np.ndarray:
        """向量化查詢"""
        # 檢查緩存
        cache_key = hashlib.md5(query.encode()).hexdigest()
        if cache_key in self.query_cache:
            cache_entry = self.query_cache[cache_key]
            if time.time() - cache_entry['timestamp'] < self.cache_duration:
                return cache_entry['vector']
        
        try:
            vector = self.embeddings.embed_query(query)
            vector_array = np.array(vector)
            
            # 緩存結果
            self.query_cache[cache_key] = {
                'vector': vector_array,
                'timestamp': time.time()
            }
            
            return vector_array
        except Exception as e:
            logger.error(f"查詢向量化失敗: {str(e)}")
            return np.array([])
    
    def vectorize_documents(self, documents: List[Document]) -> List[Tuple[Document, np.ndarray]]:
        """向量化文檔"""
        doc_vectors = []
        
        try:
            # 批量向量化
            texts = [doc.page_content for doc in documents]
            vectors = self.embeddings.embed_documents(texts)
            
            for doc, vector in zip(documents, vectors):
                doc_vectors.append((doc, np.array(vector)))
                
        except Exception as e:
            logger.error(f"文檔向量化失敗: {str(e)}")
        
        return doc_vectors
    
    def calculate_similarities(self, query_vector: np.ndarray, 
                             doc_vectors: List[Tuple[Document, np.ndarray]]) -> List[Tuple[Document, float]]:
        """計算相似度"""
        if len(query_vector) == 0:
            return []
        
        similarities = []
        
        for doc, doc_vector in doc_vectors:
            if len(doc_vector) == 0:
                continue
                
            try:
                # 手動計算餘弦相似度
                similarity = self._cosine_similarity(query_vector, doc_vector)
                similarities.append((doc, similarity))
            except Exception as e:
                logger.error(f"計算相似度失敗: {str(e)}")
                continue
        
        # 按相似度排序
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        手動計算餘弦相似度
        
        Args:
            vec1: 第一個向量
            vec2: 第二個向量
            
        Returns:
            餘弦相似度值
        """
        try:
            # 計算點積
            dot_product = np.dot(vec1, vec2)
            
            # 計算向量的模長
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            # 避免除零錯誤
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            # 計算餘弦相似度
            similarity = dot_product / (norm1 * norm2)
            
            # 確保結果在 [-1, 1] 範圍內
            return float(np.clip(similarity, -1.0, 1.0))
            
        except Exception as e:
            logger.error(f"計算餘弦相似度時出錯: {str(e)}")
            return 0.0


class DynamicRAGEngineBase(RAGEngineInterface):
    """動態RAG引擎基底類別 - 包含共通邏輯"""

    REWRITE_PROMPT_TEMPLATE = ""
    ANSWER_PROMPT_TEMPLATE = ""
    RELEVANCE_PROMPT_TEMPLATE = ""
    
    def __init__(self, ollama_model: str, ollama_embedding_model: str, platform: str = "ollama", folder_path: Optional[str] = None):
        self.ollama_model = ollama_model
        self.ollama_embedding_model = ollama_embedding_model
        self.platform = platform
        self.folder_path = folder_path
        
        # 初始化組件，傳遞文件夾路徑
        self.file_retriever = SmartFileRetriever(folder_path=folder_path)
        self.content_processor = DynamicContentProcessor()
        self.vectorizer = RealTimeVectorizer(ollama_embedding_model, platform=self.platform)
        
        # 初始化語言模型
        llm_params = {"temperature": 0.1}
        if self.platform != "ollama":
            llm_params["max_new_tokens"] = 4096

        try:
            if self.platform == "ollama":
                from langchain_ollama import OllamaLLM
                from config.config import OLLAMA_HOST
                self.llm = OllamaLLM(
                    model=ollama_model,
                    base_url=OLLAMA_HOST,
                    temperature=llm_params["temperature"],
                    timeout=OLLAMA_REQUEST_TIMEOUT,
                    request_timeout=OLLAMA_REQUEST_TIMEOUT
                )
                logger.info(f"使用 Ollama 語言模型: {ollama_model}")
            else: # 默認為 huggingface
                self.llm = ChatHuggingFace(
                    model_name=ollama_model,
                    **llm_params
                )
                logger.info(f"使用 Hugging Face 語言模型 (透過 ModelManager): {ollama_model}")
        except ImportError:
            logger.warning("langchain_ollama 未安裝，Ollama 模型將透過 Hugging Face 包裝器處理")
            self.llm = ChatHuggingFace(
                model_name=ollama_model,
                **llm_params
            )
        except Exception as e:
            logger.error(f"語言模型初始化失敗: {str(e)}")
            logger.info("回退到使用 ChatHuggingFace 作為最終方案")
            self.llm = ChatHuggingFace(
                model_name=ollama_model,
                **llm_params
            )

        # 顯式記錄最終語言與子類
        try:
            lang_info = self.get_language()
        except Exception:
            lang_info = "未知"
        logger.info(f"Dynamic RAG Engine 初始化完成 - 模型: {ollama_model}，語言: {lang_info}，引擎: {self.__class__.__name__}")

    def rewrite_query(self, original_query: str) -> str:
        """查詢重寫 - 參照傳統RAG的重試機制"""
        try:
            if len(original_query.strip()) <= 3:
                return original_query
            
            # 參照傳統RAG的重試機制
            from config.config import OLLAMA_MAX_RETRIES, OLLAMA_RETRY_DELAY
            import time
            
            for attempt in range(OLLAMA_MAX_RETRIES if hasattr(__import__('config.config'), 'OLLAMA_MAX_RETRIES') else 3):
                try:
                    prompt = self.REWRITE_PROMPT_TEMPLATE.format(original_query=original_query)
                    
                    # 使用超時控制
                    import concurrent.futures
                    def _invoke_rewrite():
                        response = self.llm.invoke(prompt)
                        return response.content.strip() if hasattr(response, 'content') else str(response).strip()
                    
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(_invoke_rewrite)
                        try:
                            rewritten_query = future.result(timeout=30)  # 30秒超時
                            
                            # 質量檢查
                            if not rewritten_query or len(rewritten_query) < 2:
                                if attempt < 2:
                                    continue
                                return original_query
                            
                            # 檢查是否過多標點符號
                            punctuation_count = sum(1 for char in rewritten_query if char in '，,。.！!？?；;：:')
                            if punctuation_count > len(rewritten_query) * 0.3:
                                if attempt < 2:
                                    continue
                                return original_query
                            
                            # 檢查長度是否合理
                            if len(rewritten_query) > len(original_query) * 3:
                                if attempt < 2:
                                    continue
                                return original_query

                            logger.info(f"優化後查詢: {rewritten_query}")
                            return rewritten_query
                            
                        except concurrent.futures.TimeoutError:
                            if attempt < 2:
                                logger.warning(f"查詢重寫超時，第 {attempt + 1} 次重試...")
                                time.sleep(1)
                                continue
                            else:
                                logger.error("查詢重寫多次超時，使用原始查詢")
                                return original_query
                
                except Exception as e:
                    if attempt < 2:
                        logger.warning(f"查詢重寫出錯，第 {attempt + 1} 次重試: {str(e)}")
                        time.sleep(1)
                        continue
                    else:
                        logger.error(f"查詢重寫多次失敗: {str(e)}")
                        return original_query
            
            return original_query
            
        except Exception as e:
            logger.error(f"查詢重寫失敗: {str(e)}")
            return original_query
    
    def get_file_count_warning(self) -> str:
        """獲取文件數量警告"""
        return getattr(self.file_retriever, '_file_count_warning', None)
    
    def answer_question(self, question: str) -> str:
        """回答問題 - 動態RAG流程（優化版本）"""
        try:
            logger.info(f"開始動態RAG處理: {question}")
            
            # 記錄文件夾限制信息
            if self.folder_path:
                logger.info(f"搜索範圍限制在文件夾: {self.folder_path}")
            
            # 1. 查詢重寫優化
            optimized_query = self.rewrite_query(question)
            
            # 2. 檢索相關文件（增加數量）
            relevant_files = self.file_retriever.retrieve_relevant_files(optimized_query, max_files=12)
            
            # 記錄文件夾限制的結果
            if self.folder_path:
                logger.info(f"文件夾限制 '{self.folder_path}' 已在檢索階段生效，找到 {len(relevant_files)} 個相關文件")
            
            if not relevant_files:
                return self._generate_general_knowledge_answer(question)
            
            # 3. 處理文件內容
            documents = self.content_processor.process_files(relevant_files)
            
            if not documents:
                return self._generate_general_knowledge_answer(question)
            
            # 4. 向量化和相似度計算
            query_vector = self.vectorizer.vectorize_query(optimized_query)
            doc_vectors = self.vectorizer.vectorize_documents(documents)
            similarities = self.vectorizer.calculate_similarities(query_vector, doc_vectors)
            
            # 5. 使用更智能的文檔選擇策略
            top_docs = self._select_best_documents(similarities, question)
            
            if not top_docs:
                return self._generate_general_knowledge_answer(question)
            
            # 6. 生成豐富的上下文
            context = self._format_enhanced_context(top_docs)
            answer = self._generate_answer(question, context)
            
            return answer
            
        except Exception as e:
            logger.error(f"動態RAG處理失敗: {str(e)}")
            return self._get_error_message()
    
    def _select_best_documents(self, similarities: List[Tuple[Document, float]], question: str) -> List[Document]:
        """
        智能選擇最佳文檔 - 參照傳統RAG的文檔選擇策略
        
        Args:
            similarities: 文檔相似度列表
            question: 原始問題
            
        Returns:
            選中的文檔列表
        """
        if not similarities:
            return []

        from config.config import SIMILARITY_TOP_K
        max_docs = min(SIMILARITY_TOP_K, 10)

        # 第一輪：收集每個文件的最佳分數
        file_best_scores = {}
        for doc, score in similarities:
            file_path = doc.metadata.get('file_path', 'unknown')
            if file_path not in file_best_scores or score > file_best_scores[file_path]:
                file_best_scores[file_path] = score

        # 計算動態閾值 (相似度越高越好)
        if file_best_scores:
            scores = list(file_best_scores.values())
            avg_score = sum(scores) / len(scores)
            # 設置一個合理的閾值，例如平均分的80%，但不低於一個絕對值
            dynamic_threshold = max(avg_score * 0.8, 0.4)
            logger.info(f"動態閾值計算: 平均分={avg_score:.3f}, 最終閾值={dynamic_threshold:.3f}")
        else:
            dynamic_threshold = 0.5

        # 第二輪：按文件去重，並根據閾值篩選
        selected_docs = []
        seen_files = set()
        # 按分數從高到低排序
        sorted_similarities = sorted(similarities, key=lambda x: x[1], reverse=True)

        for doc, score in sorted_similarities:
            if len(selected_docs) >= max_docs:
                break
            
            file_path = doc.metadata.get('file_path', 'unknown')
            if file_path not in seen_files:
                if score >= dynamic_threshold:
                    doc.metadata['score'] = score
                    selected_docs.append(doc)
                    seen_files.add(file_path)

        # 如果篩選後文檔過少，則放寬條件，取分數最高的幾個
        if not selected_docs and sorted_similarities:
            logger.info("沒有文檔通過動態閾值，放寬條件選取最高分的3個文檔")
            # 確保每個文件只選一次
            top_files = {}
            for doc, score in sorted_similarities:
                file_path = doc.metadata.get('file_path', 'unknown')
                if file_path not in top_files:
                    top_files[file_path] = doc
                    if len(top_files) >= 3:
                        break
            selected_docs = list(top_files.values())

        logger.info(f"選擇了 {len(selected_docs)} 個文檔")
        return selected_docs
    
    def _format_enhanced_context(self, documents: List[Document]) -> str:
        """
        格式化增強上下文 - 參考傳統RAG的做法，使用更結構化的格式
        
        Args:
            documents: 文檔列表
            
        Returns:
            格式化的上下文
        """
        context_parts = []
        max_total_length = 4000  # 限制總長度
        current_length = 0
        
        for i, doc in enumerate(documents, 1):
            file_name = doc.metadata.get("file_name", "未知文件")
            relative_path = doc.metadata.get("file_path", "未知路徑")
            content = doc.page_content.strip()
            
            if content and current_length < max_total_length:
                # 計算可用長度
                header = f"--- 相關文件 {i} ---\n來源: {file_name}\n路徑: {relative_path}\n"
                available_length = max_total_length - current_length - len(header)
                
                if available_length <= 0:
                    break

                if len(content) > available_length:
                    content = content[:available_length] + "... (內容截斷)"
                
                context_parts.append(f"{header}內容摘要:\n{content}\n")
                current_length += len(header) + len(content)
        
        return "\n".join(context_parts)

    
    def _format_context(self, documents: List[Document]) -> str:
        """格式化上下文"""
        context_parts = []
        for i, doc in enumerate(documents, 1):
            file_name = doc.metadata.get("file_name", "未知文件")
            content = doc.page_content.strip()
            context_parts.append(f"相關內容 {i} (來源: {file_name}):\n{content}\n")
        
        return "\n".join(context_parts)
    
    def _generate_answer(self, question: str, context: str) -> str:
        """生成回答"""
        try:
            if self.llm is None:
                return f"根據文檔內容，關於「{question}」的信息如下：\n\n{context[:500]}...\n\n注意：Dynamic RAG 語言模型暫時禁用，這是基於檢索內容的簡化回答。"
            
            prompt = self.ANSWER_PROMPT_TEMPLATE.format(context=context, question=question)
            
            response = self.llm.invoke(prompt)
            result = response.content.strip() if hasattr(response, 'content') else str(response).strip()
            
            # 檢查回答長度，如果太短則使用回退方法
            if not result or len(result.strip()) < 5:
                return self._get_general_fallback(question)
            
            # 嘗試確保輸出符合目標語言
            return self._ensure_language(result)
            
        except Exception as e:
            logger.error(f"生成回答過程中發生錯誤: {str(e)}")
            # 提供包含上下文的回退答案
            return self._generate_fallback_answer(question, context)

    def _generate_fallback_answer(self, question: str, context: str) -> str:
        """當LLM調用失敗時，提供一個基於上下文的簡化回答"""
        return f"抱歉，AI模型在生成回答時遇到問題。基於檢索到的文件，以下是相關摘要：\n\n{context[:1000]}...\n\n請您根據這些信息自行判斷。"

    
    def _generate_general_knowledge_answer(self, question: str) -> str:
        """生成常識回答 - 當找不到相關文檔時，基於常識提供回答（子類實現）"""
        # 默認實現，子類應該重寫此方法
        return f"抱歉，我在文檔中找不到與「{question}」相關的具體信息。這可能是因為相關文檔不在當前的檢索範圍內，或者問題涉及的內容需要更具體的關鍵詞。建議您嘗試使用更具體的關鍵詞重新提問。"
    
    def _ensure_language(self, result: str) -> str:
        """確保輸出符合目標語言（子類可重寫）"""
        return result
    
    def _get_error_message(self) -> str:
        """獲取錯誤消息（子類可重寫）"""
        return "處理問題時發生錯誤，請稍後再試。"
    
    def _get_general_fallback(self, query: str) -> str:
        """獲取通用回退回答（子類應該重寫此方法）"""
        # 默認實現，子類應該重寫此方法
        return f"根據一般IT知識，關於「{query}」的相關信息可能需要查閱更多QSI內部文檔。"
    
    def generate_relevance_reason(self, question: str, doc_content: str) -> str:
        """生成相關性理由"""
        if not question or not question.strip() or not doc_content or not doc_content.strip():
            return "無法生成相關性理由：查詢或文檔為空"

        try:
            trimmed_content = doc_content[:1000].strip()
            prompt = self.RELEVANCE_PROMPT_TEMPLATE.format(question=question, trimmed_content=trimmed_content)
            response = self.llm.invoke(prompt)
            reason = response.content.strip() if hasattr(response, 'content') else str(response).strip()
            return reason if reason else "無法確定相關性理由"
        except Exception as e:
            logger.error(f"生成相關性理由時出錯: {str(e)}")
            return "生成相關性理由失敗"

    def generate_batch_relevance_reasons(self, question: str, documents: List[Document]) -> List[str]:
        """批量生成相關性理由"""
        reasons = []
        for doc in documents:
            reasons.append(self.generate_relevance_reason(question, doc.page_content))
        return reasons
    
    def _get_no_docs_message(self) -> str:
        return "未找到相關文檔"
    
    def _get_error_message(self) -> str:
        return "生成過程中發生錯誤，請稍後再試。\n\n💡 這可能是因為模型尚未完全下載或初始化。如果您是首次使用，請等待模型下載完成後再試。\n\n建議：\n- 檢查網路連接\n- 選擇較小的模型進行測試\n- 查看系統狀態確認模型是否就緒"
    
    def _get_timeout_message(self) -> str:
        return "處理超時，請稍後再試"
    
    def retrieve_documents(self, query: str, top_k: int = 5) -> List[Document]:
        """動態檢索文檔"""
        try:
            relevant_files = self.file_retriever.retrieve_relevant_files(query, max_files=top_k * 2)
            documents = self.content_processor.process_files(relevant_files)
            
            if not documents:
                return []
            
            query_vector = self.vectorizer.vectorize_query(query)
            doc_vectors = self.vectorizer.vectorize_documents(documents)
            similarities = self.vectorizer.calculate_similarities(query_vector, doc_vectors)
            
            return [doc for doc, score in similarities[:top_k] if score > 0.2]
            
        except Exception as e:
            logger.error(f"動態檢索文檔失敗: {str(e)}")
            return []

    def _ensure_language(self, text: str) -> str:
        """在必要時將輸出轉換為目標語言，確保最終回答語言一致"""
        try:
            target_lang = None
            try:
                target_lang = self.get_language()
            except Exception:
                target_lang = None

            if not target_lang or not text or len(text) < 5:
                return text

            # 簡單語言特徵統計
            total_len = max(1, len(text))
            ascii_letters = sum(1 for c in text if c.isascii() and c.isalpha())
            chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
            thai_chars = sum(1 for c in text if '\u0e00' <= c <= '\u0e7f')

            ascii_ratio = ascii_letters / total_len
            zh_ratio = chinese_chars / total_len
            th_ratio = thai_chars / total_len

            def translate(to_lang_prompt: str) -> str:
                try:
                    translate_prompt = f"{to_lang_prompt}\n\n———\n{text}\n———\n只輸出翻譯結果。"
                    resp = self.llm.invoke(translate_prompt)
                    return resp.content.strip() if hasattr(response, 'content') else str(response).strip()
                except Exception:
                    return text

            if target_lang in ("繁體中文", "简体中文"):
                # 主要為中文，若中文比例過低且英文比例高，嘗試翻譯
                if zh_ratio < 0.20 and ascii_ratio > 0.50:
                    return translate("請將以下內容翻譯為繁體中文") if target_lang == "繁體中文" else translate("请将以下内容翻译为简体中文")
                return text

            if target_lang == "English":
                # 主要為英文，若英文比例過低但中文或泰文比例較高，嘗試翻譯
                if ascii_ratio < 0.40 and (zh_ratio > 0.20 or th_ratio > 0.20):
                    return translate("Please translate the following content into English")
                return text

            if target_lang == "ไทย":
                if th_ratio < 0.10 and (ascii_ratio > 0.50 or zh_ratio > 0.20):
                    return translate("โปรดแปลเนื้อหาต่อไปนี้เป็นภาษาไทย")
                return text

            return text
        except Exception:
            return text
    
    
