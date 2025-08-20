"""
Dynamic RAG Engine for Simplified Chinese
"""
import logging
from .dynamic_rag_base import DynamicRAGEngineBase

logger = logging.getLogger(__name__)

class DynamicSimplifiedChineseRAGEngine(DynamicRAGEngineBase):
    """動態簡體中文RAG引擎"""

    REWRITE_PROMPT_TEMPLATE = """你是一个搜索优化专家。请将以下问题转换为更适合在知识库中检索的完整描述性语句。

要求：
1. 保持简体中文
2. 扩展相关的专业术语和同义词
3. 包含可能在文档中出现的不同表达方式
4. 保留问题的核心概念和语意

原始问题: {original_query}

优化后的检索查询:"""

    ANSWER_PROMPT_TEMPLATE = """你是一个专业的AI文档问答助手。请严格根据“上下文信息”回答“用户问题”。

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

    RELEVANCE_PROMPT_TEMPLATE = """你是一个文档相关性评估专家。请简明扼要地解释为什么下面的文档内容与用户查询相关。请严格使用简体中文回答。

用户查询: {question}
文档内容:
---
{trimmed_content}
---
相关性理由:"""

    BATCH_RELEVANCE_PROMPT_TEMPLATE = """请为以下每个文档生成与用户查询的相关性理由。每个理由都应该是独立的一句话，简洁说明其关联性。

用户查询: {question}

---
{docs_text}
---

请严格按照以下格式输出，每个文档一行，不要有任何额外的解释或标题：
1. [文档1的相关性理由]
2. [文档2的相关性理由]
3. [文档3的相关性理由]
...
"""

    def get_language(self) -> str:
        return "简体中文"

    def get_concise_prefix(self) -> str:
        return (
            "你是一个精简的企业文件助理。请直接回答问题，限制在 8 行以内；不要重复、不要自我修正、不要加结论性赘语或再次总结。仅输出最有用的要点。\n\n"
        )
    
    def _generate_general_knowledge_answer(self, question: str) -> str:
        """当找不到相关文档时，基于常识提供简体中文回答"""
        try:
            if self.llm is None:
                return f"抱歉，我在文档中找不到与「{question}」相关的具体信息。这可能是因为相关文档不在当前的检索范围内，或者问题涉及的内容需要更具体的关键词。建议您尝试使用更具体的关键词重新提问。"
            
            general_prompt = f"""你是一个IT领域的专家助手。虽然在QSI内部文档中找不到相关资料，但请基于一般IT常识来回答以下问题。

问题：{question}

请注意：
1. 使用简体中文回答
2. 基于一般IT知识提供有用的回答
3. 明确说明这是基于常识的回答，不是来自QSI内部文档
4. 如果是QSI特定的问题，建议联系相关部门
5. 回答要简洁明了，避免重复内容

简体中文回答："""
            
            try:
                response = self.llm.invoke(general_prompt)
                answer = response.content.strip() if hasattr(response, 'content') else str(response).strip()
                
                # 添加简体中文免责声明
                disclaimer = "\n\n※ 注意：以上回答基于一般IT常识，非来自QSI内部文档。如需准确信息，请联系相关部门。"
                return answer + disclaimer
                
            except Exception as e:
                logger.error(f"简体中文动态RAG常识回答生成失败: {str(e)}")
                return self._get_general_fallback(question)
        
        except Exception as e:
            logger.error(f"简体中文动态RAG常识回答处理失败: {str(e)}")
            return self._get_general_fallback(question)
    
    def _get_general_fallback(self, query: str) -> str:
        """获取简体中文通用回退回答"""
        return f"根据一般IT知识，关于「{query}」的相关信息可能需要查阅更多QSI内部文档。"
    
    def _ensure_language(self, result: str) -> str:
        """确保输出符合简体中文"""
        return result
    
    def get_file_count_warning(self) -> str:
        """获取文件数量警告"""
        return getattr(self.file_retriever, '_file_count_warning', None)
    
    def _get_error_message(self) -> str:
        """获取简体中文错误消息"""
        return "处理问题时发生错误，请稍后再试。"
