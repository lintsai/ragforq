"""
Dynamic RAG Engine for Simplified Chinese
"""
from .dynamic_rag_base import DynamicRAGEngineBase

class DynamicSimplifiedChineseRAGEngine(DynamicRAGEngineBase):
    """動態簡體中文RAG引擎"""

    REWRITE_PROMPT_TEMPLATE = """请从以下问题中提取2-3个最重要的关键词，用空格分隔。

问题: {original_query}

关键词:"""

    ANSWER_PROMPT_TEMPLATE = """你是一个专业的AI文档问答助手。请严格根据以下“上下文信息”来回答“用户问题”。请使用与问题相同的语言（简体中文）来回答。

上下文信息:
---
{context}
---

用户问题: {question}

请提供一个准确、详细的回答。如果上下文中没有足够信息，请明确说明“根据提供的文件，我找不到相关信息”。

回答:"""

    RELEVANCE_PROMPT_TEMPLATE = """你是一个文档相关性评估专家。请简明扼要地解释为什么下面的文档内容与用户查询相关。请严格使用简体中文回答。

用户查询: {question}
文档内容:
---
{trimmed_content}
---
相关性理由:"""

    def get_language(self) -> str:
        return "简体中文"
