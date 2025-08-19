"""
Dynamic RAG Engine for English
"""
import logging
from .dynamic_rag_base import DynamicRAGEngineBase

logger = logging.getLogger(__name__)

class DynamicEnglishRAGEngine(DynamicRAGEngineBase):
    """Dynamic English RAG Engine"""

    REWRITE_PROMPT_TEMPLATE = """You are a search optimization expert. Convert the following question into a more comprehensive descriptive statement suitable for searching in a knowledge base.

Requirements:
1. Keep it in English
2. Expand relevant professional terms and synonyms
3. Include different expressions that might appear in documents
4. Preserve the core concept and semantic meaning of the question

Original question: {original_query}

Optimized search query:"""

    ANSWER_PROMPT_TEMPLATE = """You are a professional AI document assistant. Strictly answer the "User Question" based on the "Context Information".

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

    RELEVANCE_PROMPT_TEMPLATE = """You are a document relevance assessment expert. Briefly explain why the document content below is relevant to the user query. Please respond strictly in English.

User Query: {question}
Document Content:
---
{trimmed_content}
---
Relevance Reason:"""

    BATCH_RELEVANCE_PROMPT_TEMPLATE = """Please generate a relevance reason for each document below in relation to the user's query. Each reason should be a single, concise sentence explaining the connection.

User Query: {question}

---
{docs_text}
---

Please output in the following format, one line per document, without any extra explanations or headers:
1. [Relevance reason for document 1]
2. [Relevance reason for document 2]
3. [Relevance reason for document 3]
...
"""

    def get_language(self) -> str:
        return "English"

    def get_concise_prefix(self) -> str:
        # Enforce concise, non-redundant English answers
        return (
            "You are a concise enterprise document assistant. Answer directly in under 8 lines; no repetition, no self-revisions, no meta commentary. Provide only the essential points.\n\n"
        )
    
    def _generate_general_knowledge_answer(self, question: str) -> str:
        """Provide general knowledge answer in English when no relevant documents found"""
        try:
            if self.llm is None:
                return f"Sorry, I cannot find specific information related to '{question}' in the documents. This may be because the relevant documents are not within the current search scope, or the question requires more specific keywords. I suggest you try rephrasing with more specific keywords."
            
            general_prompt = f"""You are an IT expert assistant. Although no relevant information was found in QSI internal documents, please provide an answer based on general IT knowledge.

Question: {question}

Please note:
1. Answer in English only
2. Provide useful answers based on general IT knowledge
3. Clearly state this is based on general knowledge, not from QSI internal documents
4. If it's QSI-specific, suggest contacting relevant departments
5. Keep answers concise and clear, avoid repetitive content

English answer:"""
            
            try:
                response = self.llm.invoke(general_prompt)
                answer = response.content.strip() if hasattr(response, 'content') else str(response).strip()
                
                # Add English disclaimer
                disclaimer = "\n\n※ Note: The above answer is based on general IT knowledge, not from QSI internal documents. For accurate information, please contact relevant departments."
                return answer + disclaimer
                
            except Exception as e:
                logger.error(f"English dynamic RAG knowledge answer generation failed: {str(e)}")
                return self._get_general_fallback(question)
        
        except Exception as e:
            logger.error(f"English dynamic RAG knowledge answer processing failed: {str(e)}")
            return self._get_general_fallback(question)
    
    def _get_general_fallback(self, query: str) -> str:
        """Get English general fallback answer"""
        return f"Based on general IT knowledge, information about '{query}' may require consulting additional QSI internal documentation."
    
    def _ensure_language(self, result: str) -> str:
        """Ensure output is in English"""
        return result
    
    def get_file_count_warning(self) -> str:
        """Get file count warning"""
        return getattr(self.file_retriever, '_file_count_warning', None)
