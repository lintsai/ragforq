import os
import sys
import logging
import concurrent.futures
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path

# 添加項目根目錄到路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
                temperature=0.4
            )
            logger.info(f"English RAG engine initialized (Ollama) with model: {ollama_model}")
        else:
            # Hugging Face 平台
            self.llm = ChatHuggingFace(
                model_name=ollama_model,
                temperature=0.4
            )
            logger.info(f"English RAG engine initialized (Hugging Face) with model: {ollama_model}")
    
    def get_language(self) -> str:
        return "English"
    
    def rewrite_query(self, original_query: str) -> str:
        """
        Optimize English query into a more precise and comprehensive search description
        """
        try:
            rewrite_prompt = PromptTemplate(
                template="""You are a search optimization expert. Convert the following question into keywords or a descriptive phrase suitable for searching in a knowledge base. Please respond strictly in English.

Original question: {question}

Optimized search query:""",
                input_variables=["question"]
            )
            
            rewrite_chain = rewrite_prompt | self.llm | StrOutputParser()
            
            def _invoke_rewrite():
                return rewrite_chain.invoke({"question": original_query})
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_invoke_rewrite)
                try:
                    optimized_query = future.result(timeout=15)
                    logger.info(f"English query optimization: {original_query} -> {optimized_query}")
                    return optimized_query.strip()
                except concurrent.futures.TimeoutError:
                    logger.error("English query optimization timeout, using original query")
                    return original_query
            
        except Exception as e:
            logger.error(f"Error during English query optimization: {str(e)}")
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
        template = """You are a professional AI document assistant. Strictly answer the "User Question" based on the "Context Information" below. Please respond in the same language as the question (English).

Context Information:
---
{context}
---

User Question: {question}

Please provide an accurate and detailed answer. If there is not enough information in the context, please state clearly "Based on the provided documents, I could not find the relevant information."

Answer:"""

        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        chain = prompt | self.llm | StrOutputParser()
        
        def _invoke():
            return chain.invoke({"context": context, "question": question})
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_invoke)
            try:
                answer = future.result(timeout=60)
                
                if not answer or len(answer.strip()) < 5:
                    return self._get_general_fallback(question)
                
                # Detect and fix repetitive content
                cleaned_answer = self._clean_repetitive_content(answer.strip())
                return cleaned_answer
                
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
    
    def _clean_repetitive_content(self, text: str) -> str:
        """Clean repetitive content"""
        if not text:
            return text
        
        # Split into paragraphs
        paragraphs = text.split('\n\n')
        cleaned_paragraphs = []
        seen_content = set()
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # Check for duplicate paragraphs (using first 100 characters as fingerprint)
            fingerprint = paragraph[:100].strip()
            if fingerprint not in seen_content:
                seen_content.add(fingerprint)
                cleaned_paragraphs.append(paragraph)
            else:
                logger.warning(f"Detected duplicate paragraph, removed: {fingerprint[:50]}...")
        
        # Recombine
        cleaned_text = '\n\n'.join(cleaned_paragraphs)
        
        # Check for sentence-level repetition
        sentences = cleaned_text.split('. ')
        cleaned_sentences = []
        seen_sentences = set()
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Check for duplicate sentences (using first 50 characters as fingerprint)
            sentence_fingerprint = sentence[:50].strip()
            if sentence_fingerprint not in seen_sentences:
                seen_sentences.add(sentence_fingerprint)
                cleaned_sentences.append(sentence)
            else:
                logger.warning(f"Detected duplicate sentence, removed: {sentence_fingerprint[:30]}...")
        
        final_text = '. '.join(cleaned_sentences)
        if final_text and not final_text.endswith('.'):
            final_text += '.'
        
        # If cleaned content is too short, return truncated original content
        if len(final_text.strip()) < 50:
            logger.warning("Cleaned content too short, returning truncated original")
            return text[:500] + "..." if len(text) > 500 else text
        
        return final_text
