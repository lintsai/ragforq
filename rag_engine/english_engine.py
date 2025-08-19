import os
import sys
import logging
import concurrent.futures
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path

# 添加項目根目錄到路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 設置日誌
logger = logging.getLogger(__name__)

from rag_engine.interfaces import RAGEngineInterface
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from utils.hf_langchain_wrapper import HuggingFaceLLM, ChatHuggingFace
# 設置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnglishRAGEngine(RAGEngineInterface):
    """English RAG Engine Implementation"""
    
    def __init__(self, document_indexer, ollama_model: str = None, platform: str = None):
        super().__init__(document_indexer, ollama_model)
        
        # 如果沒有指定平台，自動檢測
        if platform is None:
            from config.config import detect_platform_from_model
            platform = detect_platform_from_model(ollama_model)
        
        # 根據傳入的平台參數初始化不同的 LLM
        if platform == "ollama":
            from langchain_ollama import OllamaLLM
            from config.config import OLLAMA_HOST
            self.llm = OllamaLLM(
                model=ollama_model,
                base_url=OLLAMA_HOST,
                temperature=0.4,  # 統一：與動態引擎一致
                top_p=0.9,
                repeat_penalty=1.12,
                num_predict=800
            )
            logger.info(f"English RAG engine initialized (Ollama generic) with model: {ollama_model}")
        else:
            # Hugging Face 平台：統一的簡潔輸出偏好
            llm_params = {
                "temperature": 0.3,  # 與動態引擎調整後一致
                "max_new_tokens": 768,
                "top_p": 0.9,
                "top_k": 50,
                "repetition_penalty": 1.18
            }
            self.llm = ChatHuggingFace(
                model_name=ollama_model,
                **llm_params
            )
            logger.info(f"English RAG engine initialized (Hugging Face generic) with model: {ollama_model} params: {llm_params}")
    
    def get_language(self) -> str:
        return "English"
    
    def rewrite_query(self, original_query: str) -> str:
        """
        Optimize English query into a more precise and comprehensive search description - with retry mechanism
        """
        from config.config import OLLAMA_MAX_RETRIES, OLLAMA_RETRY_DELAY, OLLAMA_QUERY_OPTIMIZATION_TIMEOUT
        import time
        
        for attempt in range(OLLAMA_MAX_RETRIES):
            try:
                rewrite_prompt = PromptTemplate(
                    template="""You are a search optimization expert. Convert the following question into a more comprehensive descriptive statement suitable for searching in a knowledge base.

Requirements:
1. Keep it in English
2. Expand relevant professional terms and synonyms
3. Include different expressions that might appear in documents
4. Preserve the core concept and semantic meaning of the question

Original question: {question}

Optimized search query:""",
                    input_variables=["question"]
                )
                
                def _invoke_rewrite():
                    # Direct invocation without chain operations
                    prompt_text = rewrite_prompt.format(question=original_query)
                    response = self.llm.invoke(prompt_text)
                    if hasattr(response, 'content'):
                        return response.content
                    else:
                        return str(response)
                
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(_invoke_rewrite)
                    try:
                        optimized_query = future.result(timeout=OLLAMA_QUERY_OPTIMIZATION_TIMEOUT)
                        logger.info(f"English query optimization: {original_query} -> {optimized_query}")
                        return optimized_query.strip()
                    except concurrent.futures.TimeoutError:
                        if attempt < OLLAMA_MAX_RETRIES - 1:
                            logger.warning(f"Query optimization timeout, retry {attempt + 1}...")
                            time.sleep(OLLAMA_RETRY_DELAY)
                            continue
                        else:
                            logger.error("Query optimization multiple timeouts, using original query")
                            return original_query
                
            except Exception as e:
                if attempt < OLLAMA_MAX_RETRIES - 1:
                    logger.warning(f"Query optimization error, retry {attempt + 1}: {str(e)}")
                    time.sleep(OLLAMA_RETRY_DELAY)
                    continue
                else:
                    logger.error(f"Query optimization multiple failures: {str(e)}")
                    return original_query
        
        return original_query
    
    def answer_question(self, question: str) -> str:
        """Answer question in English"""
        try:
            # Optimize query
            optimized_query = self.rewrite_query(question)
            
            # Retrieve documents
            docs = self.retrieve_documents(optimized_query)
            
            if not docs:
                return self._generate_general_knowledge_answer(question)
            
            # Format context
            context = self.format_context(docs)
            
            # Generate answer
            return self._generate_answer(question, context)
            
        except Exception as e:
            logger.error(f"Error during English Q&A: {str(e)}")
            return f"{self._get_error_message()}: {str(e)}"
    
    def _generate_answer(self, question: str, context: str) -> str:
        """Generate English answer"""
        template = """You are a professional AI document assistant. Strictly answer the "User Question" based on the "Context Information".

**Task Requirements:**
1.  **Direct Answer:** Provide the core answer directly to the "User Question", omitting unnecessary introductions or background information.
2.  **Be Concise:** The answer should be as concise and precise as possible, avoiding lengthy explanations and repetitive content.
3.  **Context-Based:** The answer must be entirely based on the "Context Information".
4.  **Consistent Language:** Answer in the same language as the question (English).
5.  **Unknown Handling:** If the "Context Information" is insufficient to answer, only reply "Based on the provided documents, I could not find the relevant information."

**Context Information:**
---
{context}
---

**User Question:** {question}

**Answer:**
"""

        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        chain = prompt | self.llm | StrOutputParser()
        
        def _invoke():
            # Direct invocation without chain operations
            prompt_text = prompt.format(context=context, question=question)
            response = self.llm.invoke(prompt_text)
            if hasattr(response, 'content'):
                return response.content
            else:
                return str(response)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_invoke)
            try:
                from config.config import OLLAMA_ANSWER_GENERATION_TIMEOUT
                answer = future.result(timeout=OLLAMA_ANSWER_GENERATION_TIMEOUT)
                if not answer or len(answer.strip()) < 5:
                    return self._get_general_fallback(question)
                cleaned = self._clean_answer_text(answer.strip())
                return cleaned
                
            except concurrent.futures.TimeoutError:
                logger.error("English answer generation timeout")
                return self._get_timeout_message()
    
    def generate_relevance_reason(self, question: str, doc_content: str) -> str:
        """Generate English relevance reason"""
        if not question or not question.strip():
            return "Unable to generate relevance reason: query is empty"
        
        if not doc_content or not doc_content.strip():
            return "Unable to generate relevance reason: document content is empty"
            
        try:
            trimmed_content = doc_content[:1000].strip()
            
            relevance_prompt = PromptTemplate(
                template="""You are a document relevance assessment expert. Briefly explain why the document content below is relevant to the user query. Please respond strictly in English.

User Query: {question}
Document Content:
---
{doc_content}
---
Relevance Reason:""",
                input_variables=["question", "doc_content"]
            )
            
            relevance_chain = relevance_prompt | self.llm | StrOutputParser()
            
            def _invoke_relevance():
                return relevance_chain.invoke({
                    "question": question,
                    "doc_content": trimmed_content
                })
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_invoke_relevance)
                try:
                    reason = future.result(timeout=20)
                    return reason.strip() if reason else "Unable to determine relevance reason"
                except concurrent.futures.TimeoutError:
                    return "Relevance reason generation timeout"
            
        except Exception as e:
            logger.error(f"Error generating English relevance reason: {str(e)}")
            return "Relevance reason generation failed"
    
    def _get_no_docs_message(self) -> str:
        return "Sorry, no relevant information found in QSI documents for your question."
    
    def _get_error_message(self) -> str:
        return "An error occurred while processing the question"
    
    def _get_timeout_message(self) -> str:
        return "System timeout, please try again later."
    
    def _generate_general_knowledge_answer(self, question: str) -> str:
        """Provide general knowledge answer when no relevant documents found"""
        try:
            general_prompt = PromptTemplate(
                template="""You are an IT expert assistant. Although no relevant information was found in QSI internal documents, please provide an answer based on general IT knowledge.

Question: {question}

Please note:
1. Answer in English only
2. Provide useful answers based on general IT knowledge
3. Clearly state this is based on general knowledge, not from QSI internal documents
4. If it's QSI-specific, suggest contacting relevant departments

English answer:""",
                input_variables=["question"]
            )
            
            general_chain = general_prompt | self.llm | StrOutputParser()
            
            def _invoke_general():
                return general_chain.invoke({"question": question})
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_invoke_general)
                try:
                    answer = future.result(timeout=30)
                    # Add disclaimer
                    disclaimer = "\n\n※ Note: The above answer is based on general IT knowledge, not from QSI internal documents. For accurate information, please contact relevant departments."
                    return answer.strip() + disclaimer
                except concurrent.futures.TimeoutError:
                    logger.error("General knowledge answer generation timeout")
                    return self._get_no_docs_message()
        
        except Exception as e:
            logger.error(f"Error generating general knowledge answer: {str(e)}")
            return self._get_no_docs_message()
    
    def generate_batch_relevance_reasons(self, question: str, doc_contents: list) -> list:
        """Generate relevance reasons for multiple documents in batch to improve performance"""
        if not question or not question.strip() or not doc_contents:
            return ["Unable to generate relevance reason"] * len(doc_contents)
        
        try:
            # Build batch processing prompt
            docs_text = ""
            for i, content in enumerate(doc_contents, 1):
                if content and content.strip():
                    docs_text += f"Document {i}: {content[:300]}...\n\n"
                else:
                    docs_text += f"Document {i}: (empty content)\n\n"
            
            batch_prompt = PromptTemplate(
                template="""Please generate relevance reasons for the following documents with respect to the user query. Each reason should be one concise sentence.

User query: {question}

Document contents:
{docs_text}

Please generate relevance reasons for each document in order, using this format:
1. [Relevance reason for document 1]
2. [Relevance reason for document 2]
3. [Relevance reason for document 3]
...

Relevance reasons:""",
                input_variables=["question", "docs_text"]
            )
            
            batch_chain = batch_prompt | self.llm | StrOutputParser()
            
            def _invoke_batch():
                return batch_chain.invoke({
                    "question": question,
                    "docs_text": docs_text
                })
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_invoke_batch)
                try:
                    batch_result = future.result(timeout=25)
                    
                    # Parse batch results
                    reasons = []
                    lines = batch_result.strip().split('\n')
                    for line in lines:
                        line = line.strip()
                        if line and (line.startswith(('1.', '2.', '3.', '4.', '5.')) or '.' in line[:3]):
                            # Remove numbering, keep reason
                            reason = line.split('.', 1)[1].strip() if '.' in line else line
                            reasons.append(reason)
                    
                    # Ensure correct number of reasons
                    while len(reasons) < len(doc_contents):
                        reasons.append("Relevant document")
                    
                    return reasons[:len(doc_contents)]
                    
                except concurrent.futures.TimeoutError:
                    logger.error("Batch relevance reason generation timeout")
                    return [f"Relevant document {i+1}" for i in range(len(doc_contents))]
        
        except Exception as e:
            logger.error(f"Error in batch relevance reason generation: {str(e)}")
            return [f"Relevant document {i+1}" for i in range(len(doc_contents))]
    
    def _get_general_fallback(self, query: str) -> str:
        return f"Based on general IT knowledge, information about '{query}' may require consulting additional QSI internal documentation."

    # --- Answer Cleaning (similar to dynamic engine, language-agnostic tweaks) ---
    def _clean_answer_text(self, text: str) -> str:
        """Normalize & de-duplicate answer: remove control chars, repeated disclaimers, spaced letters, duplicated paragraphs, truncate length."""
        try:
            import re
            original_len = len(text)
            text = re.sub(r'[\u200b\u200c\u200d\ufeff]', '', text)
            text = ''.join(ch for ch in text if (ch.isprintable() or ch in '\n\r\t'))
            # Merge spaced letters (K P I -> KPI)
            text = re.sub(r'\b([A-Za-z])\s+([A-Za-z])\s+([A-Za-z])\b', lambda m: ''.join(m.groups()), text)
            text = re.sub(r'\b([A-Za-z])\s+([A-Za-z])\b', lambda m: ''.join(m.groups()), text)
            lines = [l.strip() for l in text.splitlines() if l.strip()]
            filtered = []
            seen = set()
            noise_markers = ['final revision', 'final version', 'revised again', 'corrected version', '已修正']
            disclaimer_markers = ['disclaimer', 'the above answer', 'note:', '以上回答']
            for ln in lines:
                low = ln.lower()
                if any(nm in low for nm in noise_markers):
                    key = 'nm:' + ''.join(ch for ch in low if ch.isalnum())[:24]
                    if key in seen:
                        continue
                    seen.add(key)
                if any(dm in low for dm in disclaimer_markers):
                    if 'disc' in seen:
                        continue
                    seen.add('disc')
                filtered.append(ln)
            text = '\n'.join(filtered)
            paragraphs = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]
            dedup = []
            prev = None
            for p in paragraphs:
                h = hash(p[:160])
                if h == prev:
                    continue
                prev = h
                dedup.append(p)
            text = '\n\n'.join(dedup)
            if len(text) > 1500:
                text = text[:1500].rstrip() + '...'
            logger.debug(f"[EnglishRAGEngine] Cleaned answer {original_len} -> {len(text)} chars")
            return text
        except Exception:
            return text
    
    
