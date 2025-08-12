"""
Dynamic RAG Engine for English
"""
from .dynamic_rag_base import DynamicRAGEngineBase

class DynamicEnglishRAGEngine(DynamicRAGEngineBase):
    """Dynamic English RAG Engine"""

    REWRITE_PROMPT_TEMPLATE = """Extract 2-3 most important keywords from the following question, separated by spaces.

Question: {original_query}

Keywords:"""

    ANSWER_PROMPT_TEMPLATE = """You are a professional AI document assistant. Strictly answer the "User Question" based on the "Context Information" below. Please respond in the same language as the question (English).

Context Information:
---
{context}
---

User Question: {question}

Please provide an accurate and detailed answer. If there is not enough information in the context, please state clearly "Based on the provided documents, I could not find the relevant information."

Answer:"""

    RELEVANCE_PROMPT_TEMPLATE = """You are a document relevance assessment expert. Briefly explain why the document content below is relevant to the user query. Please respond strictly in English.

User Query: {question}
Document Content:
---
{trimmed_content}
---
Relevance Reason:"""

    def get_language(self) -> str:
        return "English"
