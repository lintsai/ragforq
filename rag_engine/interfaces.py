from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional
from langchain_core.documents import Document

class RAGEngineInterface(ABC):
    """RAG引擎接口，定義所有RAG引擎必須實現的方法"""
    
    def __init__(self, document_indexer, ollama_model: str = None):
        """
        初始化RAG引擎
        
        Args:
            document_indexer: 文檔索引器實例
            ollama_model: Ollama 語言模型名稱
        """
        self.document_indexer = document_indexer
        self.vector_store = document_indexer.get_vector_store()
        self.ollama_model = ollama_model
    
    @abstractmethod
    def get_language(self) -> str:
        """返回此引擎支持的語言"""
        pass
    
    @abstractmethod
    def rewrite_query(self, original_query: str) -> str:
        """
        將用戶查詢優化為適合向量檢索的查詢
        
        Args:
            original_query: 原始用戶查詢
            
        Returns:
            優化後的查詢
        """
        pass
    
    @abstractmethod
    def answer_question(self, question: str) -> str:
        """
        回答用戶問題
        
        Args:
            question: 用戶問題
            
        Returns:
            回答內容
        """
        pass
    
    def retrieve_documents(self, query: str, top_k: int = 5) -> List[Document]:
        """
        檢索與查詢最相關的文檔（優化版本）- 改進相關性和去重
        
        Args:
            query: 用戶查詢
            top_k: 返回的最大文檔數量
            
        Returns:
            按相關度排序的文檔列表
        """
        try:
            vector_store = self.vector_store
            # 檢索更多文檔以便後續按文件去重和排序
            search_count = max(top_k * 4, 20)  # 增加檢索數量
            documents = vector_store.similarity_search_with_score(query, k=search_count)
            
            # 按相似度排序（距離越小越相關）
            sorted_docs = sorted(documents, key=lambda x: x[1])
            
            # 使用動態閾值和文件去重
            filtered_docs = []
            seen_files = set()
            file_best_scores = {}
            
            # 第一輪：收集每個文件的最佳分數
            for doc, score in sorted_docs:
                file_path = doc.metadata.get('file_path', 'unknown')
                if file_path not in file_best_scores or score < file_best_scores[file_path]:
                    file_best_scores[file_path] = score
            
            # 計算動態閾值
            if file_best_scores:
                scores = list(file_best_scores.values())
                avg_score = sum(scores) / len(scores)
                dynamic_threshold = min(avg_score * 1.5, 1.8)  # 動態調整閾值
            else:
                dynamic_threshold = 1.5
            
            # 第二輪：按文件去重，每個文件只保留最相關的段落
            for doc, score in sorted_docs:
                file_path = doc.metadata.get('file_path', 'unknown')
                
                if score < dynamic_threshold and file_path not in seen_files:
                    doc.metadata['score'] = score
                    doc.metadata['relevance_rank'] = len(filtered_docs) + 1
                    filtered_docs.append(doc)
                    seen_files.add(file_path)
                    
                    # 達到所需數量就停止
                    if len(filtered_docs) >= top_k:
                        break
            
            logger.info(f"檢索到 {len(filtered_docs)} 個相關文檔，動態閾值: {dynamic_threshold:.2f}")
            return filtered_docs
            
        except Exception as e:
            logger.error(f"檢索文檔時出錯: {str(e)}")
            # 如果失敗，使用簡單搜索
            try:
                documents = vector_store.similarity_search(query, k=max(top_k * 2, 10))
                return documents[:top_k]
            except Exception:
                return []
    
    def format_context(self, docs: List[Document]) -> str:
        """格式化文檔內容（通用實現）"""
        if not docs:
            return self._get_no_docs_message()
            
        context_parts = []
        # 使用更多文檔內容，但限制總長度避免超出模型限制
        max_docs = min(len(docs), 8)  # 最多使用8個文檔段落
        total_length = 0
        max_total_length = 4000  # 限制總字數避免超出模型上下文
        
        for i, doc in enumerate(docs[:max_docs], 1):
            content = doc.page_content.strip()
            if content:
                # 檢查是否會超出總長度限制
                if total_length + len(content) > max_total_length:
                    # 如果會超出，截取部分內容
                    remaining_length = max_total_length - total_length
                    if remaining_length > 100:  # 至少保留100字符才有意義
                        content = content[:remaining_length] + "..."
                    else:
                        break  # 如果剩餘空間太少，就不再添加
                
                # 添加文件來源信息以便用戶了解內容來源
                file_name = doc.metadata.get("file_name", "未知文件")
                context_parts.append(f"相關內容 {i} (來源: {file_name}):\n{content}\n")
                total_length += len(content)
                
                # 如果已達到長度限制，停止添加
                if total_length >= max_total_length:
                    break
                
        return "\n".join(context_parts)
    
    def format_sources(self, documents: List[Document]) -> str:
        """格式化文檔來源列表（通用實現）"""
        sources = []
        seen_files = set()
        
        for doc in documents:
            metadata = doc.metadata
            file_path = metadata.get("file_path", "")
            file_name = metadata.get("file_name", "未知文件")
            
            if file_path and file_path not in seen_files:
                seen_files.add(file_path)
                location_info = ""
                if "page_number" in metadata:
                    location_info = f"（頁碼: {metadata['page_number']}）"
                elif "block_number" in metadata:
                    location_info = f"（塊: {metadata['block_number']}）"
                elif "sheet_name" in metadata:
                    location_info = f"（工作表: {metadata['sheet_name']}）"
                
                sources.append(f"- {file_name} {location_info}")
        
        return "\n".join(sources)
    
    def get_answer_with_sources(self, question: str) -> Tuple[str, str, List[Document]]:
        """
        回答問題並包含來源信息
        
        Args:
            question: 用戶問題
            
        Returns:
            (回答, 來源列表字符串, 相關文檔) 的元組
        """
        # 先檢索文檔，確保回答和來源使用相同的文檔
        optimized_query = self.rewrite_query(question)
        docs = self.retrieve_documents(optimized_query)
        
        if not docs:
            # 如果沒有找到文檔，使用常識回答
            answer = self._generate_general_knowledge_answer(question) if hasattr(self, '_generate_general_knowledge_answer') else self._get_no_docs_message()
            return answer, "", []
        
        # 基於檢索到的文檔生成回答
        context = self.format_context(docs)
        # 避免遞迴調用：如果子類有自定義的 _generate_answer 方法，使用它；否則直接使用基本回答
        if hasattr(self, '_generate_answer') and self._generate_answer != RAGEngineInterface._generate_answer:
            answer = self._generate_answer(question, context)
        else:
            # 這裡不能調用 answer_question，因為會造成遞迴
            answer = f"根據提供的文檔內容：\n\n{context[:500]}..."
        
        sources = self.format_sources(docs)
        return answer, sources, docs
    
    def get_answer_with_query_rewrite(self, original_query: str) -> Tuple[str, str, List[Document], str]:
        """
        使用查詢重寫策略來優化問答效果
        
        Args:
            original_query: 原始用戶問題
            
        Returns:
            (回答, 來源列表字符串, 相關文檔, 重寫查詢) 的元組
        """
        # 先檢索文檔，確保回答和來源使用相同的文檔
        rewritten_query = self.rewrite_query(original_query)
        docs = self.retrieve_documents(rewritten_query)
        
        if not docs:
            # 如果沒有找到文檔，使用常識回答
            answer = self._generate_general_knowledge_answer(original_query) if hasattr(self, '_generate_general_knowledge_answer') else self._get_no_docs_message()
            return answer, "", [], rewritten_query
        
        # 基於檢索到的文檔生成回答
        context = self.format_context(docs)
        # 避免遞迴調用：如果子類有自定義的 _generate_answer 方法，使用它；否則直接使用基本回答
        if hasattr(self, '_generate_answer') and self._generate_answer != RAGEngineInterface._generate_answer:
            answer = self._generate_answer(original_query, context)
        else:
            # 這裡不能調用 answer_question，因為會造成遞迴
            answer = f"根據提供的文檔內容：\n\n{context[:500]}..."
        
        sources = self.format_sources(docs)
        return answer, sources, docs, rewritten_query
    
    @abstractmethod
    def generate_relevance_reason(self, question: str, doc_content: str) -> str:
        """
        生成文檔與查詢之間的相關性理由
        
        Args:
            question: 用戶查詢
            doc_content: 文檔內容
            
        Returns:
            相關性理由描述
        """
        pass
    
    @abstractmethod
    def _get_no_docs_message(self) -> str:
        """獲取無文檔時的訊息"""
        pass
    
    @abstractmethod
    def _get_error_message(self) -> str:
        """獲取錯誤訊息"""
        pass
    
    @abstractmethod
    def _get_timeout_message(self) -> str:
        """獲取超時訊息"""
        pass
    
    def _generate_answer(self, question: str, context: str) -> str:
        """
        生成回答的默認實現（子類可以重寫）
        
        Args:
            question: 用戶問題
            context: 格式化的文檔上下文
            
        Returns:
            生成的回答
        """
        # 默認實現：基於上下文提供簡單回答，避免遞迴調用 answer_question
        return f"根據提供的文檔內容，關於「{question}」的相關信息如下：\n\n{context[:500]}..."