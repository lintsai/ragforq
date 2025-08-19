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
from utils.ollama_utils import ollama_utils
# 設置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimplifiedChineseRAGEngine(RAGEngineInterface):
    """简体中文RAG引擎实现"""
    
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
            logger.info(f"简体中文RAG引擎初始化完成 (Ollama)，使用模型: {ollama_model}")
        else:
            # Hugging Face 平台
            llm_params = {
                "temperature": 0.1,
                "max_new_tokens": 1024,
                "top_p": 0.9,
                "top_k": 50,
                "repetition_penalty": 1.15
            }
            self.llm = ChatHuggingFace(
                model_name=ollama_model,
                **llm_params
            )
            logger.info(f"简体中文RAG引擎初始化完成 (Hugging Face)，使用模型: {ollama_model} with params: {llm_params}")
    
    def get_language(self) -> str:
        return "简体中文"
    
    def rewrite_query(self, original_query: str) -> str:
        """
        将简体中文查询优化为更精准适合向量检索的完整描述 - 带重试机制
        """
        from config.config import OLLAMA_MAX_RETRIES, OLLAMA_RETRY_DELAY, OLLAMA_QUERY_OPTIMIZATION_TIMEOUT
        import time
        
        for attempt in range(OLLAMA_MAX_RETRIES):
            try:
                rewrite_prompt = PromptTemplate(
                    template="""你是一个搜索优化专家。请将以下问题转换为更适合在知识库中检索的关键词或描述性语句。请严格使用简体中文输出。

原始问题: {question}

优化后的检索查询:""",
                    input_variables=["question"]
                )
                
                def _invoke_rewrite():
                    # 直接调用而不使用链式操作
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
                        logger.info(f"简体中文查询优化: {original_query} -> {optimized_query}")
                        return optimized_query.strip()
                    except concurrent.futures.TimeoutError:
                        if attempt < OLLAMA_MAX_RETRIES - 1:
                            logger.warning(f"查询优化超时，第 {attempt + 1} 次重试...")
                            time.sleep(OLLAMA_RETRY_DELAY)
                            continue
                        else:
                            logger.error("查询优化多次超时，使用原始查询")
                            return original_query
                
            except Exception as e:
                if attempt < OLLAMA_MAX_RETRIES - 1:
                    logger.warning(f"查询优化出错，第 {attempt + 1} 次重试: {str(e)}")
                    time.sleep(OLLAMA_RETRY_DELAY)
                    continue
                else:
                    logger.error(f"查询优化多次失败: {str(e)}")
                    return original_query
        
        return original_query
    
    def answer_question(self, question: str) -> str:
        """使用简体中文回答问题"""
        try:
            # 优化查询
            optimized_query = self.rewrite_query(question)
            
            # 检索文档
            docs = self.retrieve_documents(optimized_query)
            
            if not docs:
                return self._generate_general_knowledge_answer(question)
            
            # 格式化上下文
            context = self.format_context(docs)
            
            # 生成回答
            return self._generate_answer(question, context)
            
        except Exception as e:
            logger.error(f"简体中文问答时出错: {str(e)}")
            return f"{self._get_error_message()}: {str(e)}"
    
    def _generate_answer(self, question: str, context: str) -> str:
        """生成简体中文回答"""
        template = """你是一个专业的AI文档问答助手。请严格根据“上下文信息”回答“用户问题”。

**任务要求:**
1.  **直接回答:** 直接针对“用户问题”提供核心答案，省略不必要的引言或背景信息。
2.  **保持简洁:** 回答应尽可能简洁、精确，避免冗长的解释和重复的内容。
3.  **基于上下文:** 答案必须完全基于“上下文信息”。
4.  **语言一致:** 使用与问题相同的语言（简体中文）回答。
5.  **未知处理:** 如果“上下文信息”不足以回答，仅回答“根据提供的文件，我找不到相关信息”。

**上下文信息:**
---
{context}
---

**用户问题:** {question}

**回答:**
"""

        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        chain = prompt | self.llm | StrOutputParser()
        
        def _invoke():
            # 直接调用而不使用链式操作
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
                
                return answer.strip()
                
            except concurrent.futures.TimeoutError:
                logger.error("简体中文回答生成超时")
                return self._get_timeout_message()
    
    def generate_relevance_reason(self, question: str, doc_content: str) -> str:
        """生成简体中文相关性理由"""
        if not question or not question.strip():
            return "无法生成相关性理由：查询为空"
        
        if not doc_content or not doc_content.strip():
            return "无法生成相关性理由：文档内容为空"
            
        try:
            trimmed_content = doc_content[:1000].strip()
            
            relevance_prompt = PromptTemplate(
                template="""你是一个文档相关性评估专家。请简明扼要地解释为什么下面的文档内容与用户查询相关。请严格使用简体中文回答。

用户查询: {question}
文档内容:
---
{doc_content}
---
相关性理由：""",
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
                    return reason.strip() if reason else "无法确定相关性理由"
                except concurrent.futures.TimeoutError:
                    return "生成相关性理由超时"
            
        except Exception as e:
            logger.error(f"生成简体中文相关性理由时出错: {str(e)}")
            return "生成相关性理由失败"
    
    def _get_no_docs_message(self) -> str:
        return "抱歉，在QSI文档中找不到与您问题相关的信息。"
    
    def _get_error_message(self) -> str:
        return "处理问题时发生错误"
    
    def _get_timeout_message(self) -> str:
        return "系统处理超时，请稍后再试。"
    
    def _generate_general_knowledge_answer(self, question: str) -> str:
        """当找不到相关文档时，基于常识提供回答"""
        try:
            general_prompt = PromptTemplate(
                template="""你是一个IT领域的专家助手。虽然在QSI内部文档中找不到相关资料，但请基于一般IT常识来回答以下问题。

问题：{question}

请注意：
1. 使用简体中文回答
2. 基于一般IT知识提供有用的回答
3. 明确说明这是基于常识的回答，不是来自QSI内部文档
4. 如果是QSI特定的问题，建议联系相关部门

简体中文回答：""",
                input_variables=["question"]
            )
            
            general_chain = general_prompt | self.llm | StrOutputParser()
            
            def _invoke_general():
                return general_chain.invoke({"question": question})
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_invoke_general)
                try:
                    answer = future.result(timeout=30)
                    # 添加免责声明
                    disclaimer = "\n\n※ 注意：以上回答基于一般IT常识，非来自QSI内部文档。如需准确信息，请联系相关部门。"
                    return answer.strip() + disclaimer
                except concurrent.futures.TimeoutError:
                    logger.error("常识回答生成超时")
                    return self._get_no_docs_message()
        
        except Exception as e:
            logger.error(f"生成常识回答时出错: {str(e)}")
            return self._get_no_docs_message()
    
    def generate_batch_relevance_reasons(self, question: str, doc_contents: list) -> list:
        """批量生成多个文档的相关性理由，提高效能"""
        if not question or not question.strip() or not doc_contents:
            return ["无法生成相关性理由"] * len(doc_contents)
        
        try:
            # 构建批量处理的prompt
            docs_text = ""
            for i, content in enumerate(doc_contents, 1):
                if content and content.strip():
                    docs_text += f"文档{i}: {content[:300]}...\n\n"
                else:
                    docs_text += f"文档{i}: (空内容)\n\n"
            
            batch_prompt = PromptTemplate(
                template="""请为以下文档分别生成与用户查询的相关性理由。每个理由用一句话简洁说明。

用户查询: {question}

文档内容:
{docs_text}

请按顺序为每个文档生成相关性理由，格式如下:
1. [文档1的相关性理由]
2. [文档2的相关性理由]
3. [文档3的相关性理由]
...

相关性理由:""",
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
                    
                    # 解析批量结果
                    reasons = []
                    lines = batch_result.strip().split('\n')
                    for line in lines:
                        line = line.strip()
                        if line and (line.startswith(('1.', '2.', '3.', '4.', '5.')) or '.' in line[:3]):
                            # 移除序号，保留理由
                            reason = line.split('.', 1)[1].strip() if '.' in line else line
                            reasons.append(reason)
                    
                    # 确保返回正确数量的理由
                    while len(reasons) < len(doc_contents):
                        reasons.append("相关文档")
                    
                    return reasons[:len(doc_contents)]
                    
                except concurrent.futures.TimeoutError:
                    logger.error("批量生成相关性理由超时")
                    return [f"相关文档 {i+1}" for i in range(len(doc_contents))]
        
        except Exception as e:
            logger.error(f"批量生成相关性理由时出错: {str(e)}")
            return [f"相关文档 {i+1}" for i in range(len(doc_contents))]
    
    def _get_general_fallback(self, query: str) -> str:
        return f"根据一般IT知识，关于「{query}」的相关信息可能需要查阅更多QSI内部文档。"
    
    
