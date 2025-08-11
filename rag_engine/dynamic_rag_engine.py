"""
Dynamic RAG Engine - 動態檢索增強生成引擎
無需預先建立向量資料庫，查詢時即時檢索和處理文件
"""

import os
import logging
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib

from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np

from .interfaces import RAGEngineInterface
from utils.file_parsers import FileParser
from config.config import (
    OLLAMA_HOST, MAX_TOKENS_CHUNK, CHUNK_OVERLAP,
    SUPPORTED_FILE_TYPES, Q_DRIVE_PATH
)

logger = logging.getLogger(__name__)

class SmartFileRetriever:
    """智能文件檢索器 - 根據查詢內容智能選擇相關文件"""
    
    def __init__(self, base_path: str = Q_DRIVE_PATH):
        self.base_path = base_path
        self.file_cache = {}  # 文件元數據緩存
        self.last_scan_time = 0
        self.cache_duration = 300  # 5分鐘緩存
    
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
        
        # 1. 關鍵詞匹配
        keyword_matches = self._match_by_keywords(query)
        
        # 2. 路徑語義分析
        path_matches = self._analyze_file_paths(query)
        
        # 3. 合併和去重
        all_matches = list(set(keyword_matches + path_matches))
        
        # 4. 評分和排序
        scored_files = self._score_files(query, all_matches)
        
        # 5. 返回前N個文件
        return [file_path for file_path, score in scored_files[:max_files]]
    
    def _update_file_cache(self):
        """更新文件緩存"""
        current_time = time.time()
        if current_time - self.last_scan_time < self.cache_duration:
            return
        
        logger.info("更新文件緩存...")
        self.file_cache = {}
        
        try:
            # 限制掃描深度和文件數量以避免性能問題
            max_files = 1000  # 最多緩存1000個文件
            max_depth = 5     # 最大掃描深度
            file_count = 0
            
            for root, dirs, files in os.walk(self.base_path):
                # 計算當前深度
                depth = root.replace(self.base_path, '').count(os.sep)
                if depth >= max_depth:
                    dirs[:] = []  # 不再深入子目錄
                    continue
                
                # 限制每個目錄中處理的文件數量
                files = files[:100]  # 每個目錄最多處理100個文件
                
                for file in files:
                    if file_count >= max_files:
                        logger.info(f"達到文件數量限制 ({max_files})，停止掃描")
                        break
                        
                    file_path = os.path.join(root, file)
                    file_ext = os.path.splitext(file)[1].lower()
                    
                    if file_ext in SUPPORTED_FILE_TYPES:
                        try:
                            stat = os.stat(file_path)
                            self.file_cache[file_path] = {
                                'name': file,
                                'size': stat.st_size,
                                'mtime': stat.st_mtime,
                                'ext': file_ext,
                                'relative_path': os.path.relpath(file_path, self.base_path)
                            }
                            file_count += 1
                        except (OSError, PermissionError):
                            continue
                
                if file_count >= max_files:
                    break
            
            self.last_scan_time = current_time
            logger.info(f"文件緩存更新完成，共 {len(self.file_cache)} 個文件")
            
        except Exception as e:
            logger.error(f"更新文件緩存失敗: {str(e)}")
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
    
    def __init__(self, embedding_model: str):
        self.embeddings = OllamaEmbeddings(
            base_url=OLLAMA_HOST,
            model=embedding_model
        )
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


class DynamicRAGEngine(RAGEngineInterface):
    """動態RAG引擎 - 無需預先建立向量資料庫"""
    
    def __init__(self, ollama_model: str, ollama_embedding_model: str, language: str = "繁體中文"):
        # 注意：不需要document_indexer參數
        self.ollama_model = ollama_model
        self.ollama_embedding_model = ollama_embedding_model
        self.language = language
        
        # 初始化組件
        self.file_retriever = SmartFileRetriever()
        self.content_processor = DynamicContentProcessor()
        self.vectorizer = RealTimeVectorizer(ollama_embedding_model)
        
        # 初始化語言模型
        self.llm = ChatOllama(
            base_url=OLLAMA_HOST,
            model=ollama_model,
            temperature=0.1
        )
        
        logger.info(f"Dynamic RAG Engine 初始化完成 - 語言: {language}, 模型: {ollama_model}")
    
    def get_language(self) -> str:
        return self.language
    
    def rewrite_query(self, original_query: str) -> str:
        """查詢重寫 - 針對動態檢索優化"""
        # 對於動態RAG，查詢重寫更注重關鍵詞提取
        try:
            prompt = f"""
請將以下用戶問題轉換為更適合文件檢索的關鍵詞組合。
保持原語言，提取核心概念和相關術語。

原問題: {original_query}

請提供：
1. 核心關鍵詞（2-3個）
2. 相關術語或同義詞
3. 可能的文件類型或分類

優化後的檢索查詢:"""
            
            response = self.llm.invoke(prompt)
            return response.content.strip()
            
        except Exception as e:
            logger.error(f"查詢重寫失敗: {str(e)}")
            return original_query
    
    def answer_question(self, question: str) -> str:
        """回答問題 - 動態RAG流程"""
        try:
            # 1. 智能文件檢索
            logger.info(f"開始動態RAG處理: {question}")
            relevant_files = self.file_retriever.retrieve_relevant_files(question, max_files=8)
            
            if not relevant_files:
                return self._generate_general_knowledge_answer(question)
            
            logger.info(f"檢索到 {len(relevant_files)} 個相關文件")
            
            # 2. 動態內容處理
            documents = self.content_processor.process_files(relevant_files)
            
            if not documents:
                return self._generate_general_knowledge_answer(question)
            
            logger.info(f"處理得到 {len(documents)} 個文檔段落")
            
            # 3. 即時向量化和相似度計算
            query_vector = self.vectorizer.vectorize_query(question)
            doc_vectors = self.vectorizer.vectorize_documents(documents)
            similarities = self.vectorizer.calculate_similarities(query_vector, doc_vectors)
            
            # 4. 選擇最相關的內容
            top_docs = [doc for doc, score in similarities[:5] if score > 0.3]
            
            if not top_docs:
                return self._generate_general_knowledge_answer(question)
            
            # 5. 生成回答
            context = self._format_context(top_docs)
            answer = self._generate_answer(question, context)
            
            return answer
            
        except Exception as e:
            logger.error(f"動態RAG處理失敗: {str(e)}")
            return self._get_error_message()
    
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
            prompt = f"""你是一個專業的文檔問答助手。請根據提供的上下文信息回答用戶的問題。

上下文信息:
{context}

用戶問題: {question}

請提供準確、詳細的回答。如果上下文中沒有足夠信息，請明確說明。

回答:"""
            
            response = self.llm.invoke(prompt)
            return response.content.strip()
            
        except Exception as e:
            logger.error(f"生成回答失敗: {str(e)}")
            return self._get_error_message()
    
    def _generate_general_knowledge_answer(self, question: str) -> str:
        """生成常識回答"""
        return f"抱歉，我在文檔中找不到與「{question}」相關的具體信息。這可能是因為相關文檔不在當前的檢索範圍內，或者問題涉及的內容需要更具體的關鍵詞。建議您嘗試使用更具體的關鍵詞重新提問。"
    
    def generate_relevance_reason(self, question: str, doc_content: str) -> str:
        """生成相關性理由"""
        return "基於動態檢索的相似度計算"
    
    def _get_no_docs_message(self) -> str:
        return "未找到相關文檔"
    
    def _get_error_message(self) -> str:
        return "處理過程中發生錯誤，請稍後再試"
    
    def _get_timeout_message(self) -> str:
        return "處理超時，請稍後再試"
    
    # 重寫基類方法以適應動態RAG
    def retrieve_documents(self, query: str, top_k: int = 5) -> List[Document]:
        """動態檢索文檔"""
        try:
            # 使用動態RAG流程
            relevant_files = self.file_retriever.retrieve_relevant_files(query, max_files=top_k * 2)
            documents = self.content_processor.process_files(relevant_files)
            
            if not documents:
                return []
            
            # 向量化和相似度計算
            query_vector = self.vectorizer.vectorize_query(query)
            doc_vectors = self.vectorizer.vectorize_documents(documents)
            similarities = self.vectorizer.calculate_similarities(query_vector, doc_vectors)
            
            # 返回最相關的文檔
            return [doc for doc, score in similarities[:top_k] if score > 0.2]
            
        except Exception as e:
            logger.error(f"動態檢索文檔失敗: {str(e)}")
            return []