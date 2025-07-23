import os
import sys
import logging
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path

# 添加項目根目錄到路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import SIMILARITY_TOP_K, OLLAMA_HOST
from indexer.document_indexer import DocumentIndexer
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from langchain_community.vectorstores.faiss import FAISS
from langchain_ollama import OllamaLLM

# 設置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 定義問答模板
QA_TEMPLATE = """你是一個專業的文檔問答助手，你的任務是根據提供的上下文信息回答用戶的問題。
請只使用提供的上下文信息來回答問題。如果上下文中沒有足夠的信息，請回答「我無法從文檔中找到足夠的信息來回答這個問題」。
請不要編造信息，不要添加上下文中沒有的內容。

上下文信息:
{context}

用戶問題: {question}

回答:"""

# 定義資源列表模板
SOURCE_TEMPLATE = """以下是與問題相關的文檔來源列表：

{sources}

請問您還有其他問題嗎？"""


class RAGEngine:
    """RAG查詢引擎，負責檢索文件並回答問題"""
    
    def __init__(self, document_indexer: DocumentIndexer, ollama_model: str = None):
        """
        初始化RAG引擎
        
        Args:
            document_indexer: 文檔索引器實例
            ollama_model: Ollama 語言模型名稱，如果為 None 則需要在使用時指定
        """
        self.document_indexer = document_indexer
        self.vector_store = document_indexer.get_vector_store()
        
        # 使用傳入的語言模型
        self.llm = OllamaLLM(
            model=ollama_model,
            base_url=OLLAMA_HOST,
            temperature=0.7
        )
        self.ollama_model = ollama_model
        logger.info(f"使用本地 Ollama 模型: {ollama_model}")
        
        # 定義問答提示模板
        self.qa_prompt = PromptTemplate(
            template="""你是一個專業的廣明光電IT文檔問答助手。

請根據以下上下文信息回答用戶的問題。上下文信息主要來自問題相關的文檔。
如果上下文中包含相關信息，請優先使用文檔內容回答。
只有在上下文完全沒有相關信息時，才輸出特殊標記：[NO_DOCUMENT_ANSWER]。

判斷標準：
- 如果上下文中有任何與問題相關的信息（即使不完整），都應該基於文檔內容回答
- 只有當上下文完全無關或為空時，才使用[NO_DOCUMENT_ANSWER]標記

上下文信息:
-----------------
{context}
-----------------

用戶問題: {question}

請直接回答問題，優先使用文檔中的信息。如果文檔信息不完整，可以說明「根據文檔顯示...」並提供可用信息。
回答應該直接切入重點，簡潔清晰。

你的回答 (請直接回答，優先使用文檔內容):""",
            input_variables=["context", "question"]
        )
        
        # 創建問答鏈
        self.qa_chain = self.qa_prompt | self.llm | StrOutputParser()

        # --- 新增：通用知識問答模板與鏈 ---
        self.general_knowledge_prompt = PromptTemplate(
            template="""你是一個聰明且樂於助人的並專注於廣明光電IT知識的 AI 助手。請根據你的通用知識和邏輯推理能力，盡力回答以下問題。

用戶問題: {question}

你的回答:""",
            input_variables=["question"]
        )
        self.general_knowledge_chain = self.general_knowledge_prompt | self.llm | StrOutputParser()
    
    def retrieve_documents(self, query: str, top_k: int = 10) -> List[Document]:
        """
        檢索與查詢相關的文檔
        
        Args:
            query: 用戶查詢
            top_k: 返回的最大文檔數量
            
        Returns:
            相關文檔列表
        """
        try:
            vector_store = self.vector_store
            # 增加檢索數量以提高找到相關文檔的機會
            documents = vector_store.similarity_search_with_score(query, k=top_k)
            
            # 過濾相似度過低的文檔（相似度閾值可調整）
            filtered_docs = []
            for doc, score in documents:
                # FAISS使用距離，距離越小相似度越高
                # 設置一個合理的距離閾值（可根據實際情況調整）
                if score < 1.5:  # 距離小於1.5認為是相關的
                    doc.metadata['score'] = score
                    filtered_docs.append(doc)
            
            logger.info(f"檢索到 {len(documents)} 個文檔，過濾後保留 {len(filtered_docs)} 個相關文檔")
            return filtered_docs
        except Exception as e:
            logger.error(f"檢索文檔時出錯: {str(e)}")
            # 如果similarity_search_with_score失敗，回退到普通搜索
            try:
                documents = vector_store.similarity_search(query, k=top_k)
                logger.info(f"回退搜索：檢索到 {len(documents)} 個相關文檔")
                return documents
            except Exception as e2:
                logger.error(f"回退搜索也失敗: {str(e2)}")
                return []
    
    def format_context(self, docs: List[Document]) -> str:
        """
        格式化文檔內容為上下文
        
        Args:
            docs: 文檔列表
            
        Returns:
            格式化的上下文字符串
        """
        if not docs:
            return "沒有找到相關文檔。"
            
        context_parts = []
        for i, doc in enumerate(docs, 1):
            content = doc.page_content.strip()
            if content:
                context_parts.append(f"文檔 {i}:\n{content}\n")
                
        return "\n".join(context_parts)
    
    def format_sources(self, documents: List[Document]) -> str:
        """
        格式化文檔來源列表
        
        Args:
            documents: 文檔列表
            
        Returns:
            格式化的來源列表字符串
        """
        sources = []
        seen_files = set()
        
        for doc in documents:
            metadata = doc.metadata
            file_path = metadata.get("file_path", "")
            file_name = metadata.get("file_name", "未知文件")
            
            if file_path and file_path not in seen_files:
                seen_files.add(file_path)
                # 提取頁碼或塊信息
                location_info = ""
                if "page_number" in metadata:
                    location_info = f"（頁碼: {metadata['page_number']}）"
                elif "block_number" in metadata:
                    location_info = f"（塊: {metadata['block_number']}）"
                elif "sheet_name" in metadata:
                    location_info = f"（工作表: {metadata['sheet_name']}）"
                
                sources.append(f"- {file_name} {location_info}")
        
        return "\n".join(sources)
    
    def rewrite_query(self, original_query: str, language: str = "繁體中文") -> str:
        """
        改寫用戶查詢，使其更適合 RAG 檢索
        
        Args:
            original_query: 原始用戶查詢
            language: 目標語言
            
        Returns:
            改寫後的查詢
        """
        try:
            # 定義查詢改寫提示
            rewrite_prompt = PromptTemplate(
                template="""你是一個專業的廣明光電IT知識查詢優化專家。請識別以下用戶問題，並將其改寫為更適合從文檔檢索系統中獲取準確答案的格式。

原始問題: {question}

改寫時請遵循以下原則:
1. 提取核心關鍵詞和實體。
2. 移除不必要的修飾詞和語氣詞。
3. 保持問題的本質不變。
4. 確保改寫後的問題更有針對性、更明確。
5. 不要添加原問題中不存在的信息。
6. 如果原問題已經很清晰，可以保持不變。
7. 以廣明光電IT知識及廣明光電的產業知識為範圍
8. 請用 "{language}" 進行改寫。

改寫後的問題 (直接輸出改寫結果，不要添加任何前綴或說明):""",
                input_variables=["question", "language"]
            )
            
            # 創建查詢改寫鏈
            rewrite_chain = rewrite_prompt | self.llm | StrOutputParser()
            
            # 獲取改寫結果
            rewritten_query = rewrite_chain.invoke({
                "question": original_query,
                "language": language
            })
            logger.info(f"原始查詢: {original_query}")
            logger.info(f"改寫查詢: {rewritten_query}")
            
            return rewritten_query.strip()
            
        except Exception as e:
            logger.error(f"改寫查詢時出錯: {str(e)}")
            return original_query  # 如果出錯，返回原始查詢
    
    def get_answer_with_query_rewrite(self, original_query: str, language: str = "繁體中文") -> Tuple[str, str, List[Document], str]:
        """
        使用查詢改寫來優化問答效果
        
        Args:
            original_query: 原始用戶問題
            language: 目標語言
            
        Returns:
            (回答, 來源列表字符串, 相關文檔, 改寫後的查詢) 的元組
        """
        # 改寫查詢
        rewritten_query = self.rewrite_query(original_query, language)
        
        # 使用改寫後的查詢獲取答案
        answer = self.answer_question(rewritten_query, language)
        
        # 使用改寫後的查詢檢索文檔
        docs = self.retrieve_documents(rewritten_query)
        
        if not docs:
            return answer, "", [], rewritten_query
        
        sources = self.format_sources(docs)
        
        return answer, sources, docs, rewritten_query
    
    def answer_question(self, question: str, language: str = "繁體中文") -> str:
        """回答問題，如果文檔中沒有答案，則使用通用知識並附帶免責聲明"""
        try:
            # 第一步：嘗試從文檔中獲取答案
            docs = self.retrieve_documents(question)
            context = self.format_context(docs)

            # 更新 QA prompt 以包含語言指令
            qa_template_with_lang = self.qa_prompt.template + f"\n\n請用 '{language}' 來回答。"
            qa_prompt_with_lang = PromptTemplate(
                template=qa_template_with_lang,
                input_variables=["context", "question"]
            )
            qa_chain_with_lang = qa_prompt_with_lang | self.llm | StrOutputParser()

            rag_response = qa_chain_with_lang.invoke({
                "context": context,
                "question": question
            })
            
            # 第二步：檢查是否觸發了Fallback
            if "[NO_DOCUMENT_ANSWER]" in rag_response:
                logger.info(f"在文檔中未找到答案，問題 '{question}' 將切換到通用知識模式...")
                
                # 更新通用知識 prompt 以包含語言指令
                general_template_with_lang = self.general_knowledge_prompt.template + f"\n\n請用 '{language}' 來回答。"
                general_prompt_with_lang = PromptTemplate(
                    template=general_template_with_lang,
                    input_variables=["question"]
                )
                general_chain_with_lang = general_prompt_with_lang | self.llm | StrOutputParser()

                # 觸發通用知識鏈
                general_response = general_chain_with_lang.invoke({
                    "question": question
                })
                
                # 構建帶有免責聲明的最終回答
                disclaimer = "（在您提供的文檔中未找到直接答案，以下是基於通用知識的回答。請注意，此內容僅為模型基於公開資訊的推論，並非肯定答案。）"
                final_answer = f"{disclaimer}\n\n---\n\n{general_response}"
                return final_answer
            else:
                # 如果RAG成功，直接返回結果
                return rag_response
            
        except Exception as e:
            logger.error(f"回答問題時出錯: {str(e)}")
            return f"處理問題時發生錯誤: {str(e)}"
    
    def get_answer_with_sources(self, question: str, language: str = "繁體中文") -> Tuple[str, str, List[Document]]:
        """
        回答問題並包含來源信息
        
        Args:
            question: 用戶問題
            language: 目標語言
            
        Returns:
            (回答, 來源列表字符串, 相關文檔) 的元組
        """
        answer = self.answer_question(question, language)
        
        if not self.retrieve_documents(question):
            return answer, "", []
        
        sources = self.format_sources(self.retrieve_documents(question))
        
        return answer, sources, self.retrieve_documents(question)
        
    def generate_relevance_reason(self, question: str, doc_content: str) -> str:
        """
        生成文檔與查詢之間的相關性理由
        
        Args:
            question: 用戶查詢
            doc_content: 文檔內容
            
        Returns:
            相關性理由描述
        """
        # 檢查輸入參數
        if not question or not question.strip():
            return "無法生成相關性理由：查詢為空"
        
        if not doc_content or not doc_content.strip():
            return "無法生成相關性理由：文檔內容為空"
            
        try:
            # 限制處理內容長度，避免過長
            trimmed_content = doc_content[:1000].strip()
            
            # 建立相關性理由提示模板
            relevance_prompt = PromptTemplate(
                template="""你是一個文檔相關性評估專家。請簡明扼要地解釋為什麼下面的文檔內容與用戶查詢相關。
                
用戶查詢: {question}

文檔內容:
-----------------
{doc_content}
-----------------

請提供1-2句簡短的解釋，說明這個文檔為什麼與查詢相關。不要重複查詢內容，直接解釋關聯性。
(直接輸出解釋，不要添加任何前綴如"這個文檔相關因為"等):""",
                input_variables=["question", "doc_content"]
            )
            
            # 創建相關性理由鏈
            relevance_chain = relevance_prompt | self.llm | StrOutputParser()
            
            # 獲取相關性理由
            reason = relevance_chain.invoke({
                "question": question,
                "doc_content": trimmed_content
            })
            
            return reason.strip() if reason else "無法確定相關性理由"
            
        except Exception as e:
            logger.error(f"生成相關性理由時出錯: {str(e)}")
            return "無法生成相關性理由"


# 使用示例
if __name__ == "__main__":
    # 導入示例文件索引器
    from indexer.document_indexer import DocumentIndexer
    
    # 創建RAG引擎
    indexer = DocumentIndexer()
    rag_engine = RAGEngine(indexer)
    
    # 示例問題
    question = "ITP是甚麼？"
    
    # 獲取回答和來源
    answer, sources, _ = rag_engine.get_answer_with_sources(question)
    
    # 輸出結果
    print("問題:", question)
    print("\n回答:", answer)
    print("\n來源:\n", sources)
