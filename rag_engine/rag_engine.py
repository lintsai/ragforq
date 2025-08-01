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
# 清理後的導入
from langchain_community.vectorstores.faiss import FAISS
from langchain_ollama import OllamaLLM

# 設置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        
        # 使用傳入的語言模型，適中溫度平衡一致性和創造性
        self.llm = OllamaLLM(
            model=ollama_model,
            base_url=OLLAMA_HOST,
            temperature=0.4  # 適中溫度，平衡一致性和回答品質
        )
        self.ollama_model = ollama_model
        logger.info(f"使用本地 Ollama 模型: {ollama_model}")
    
    def retrieve_documents(self, query: str, top_k: int = 5) -> List[Document]:
        """
        檢索與查詢最相關的文檔，優先返回相關度最高的文檔
        
        Args:
            query: 用戶查詢
            top_k: 返回的最大文檔數量（減少到5個以提高質量）
            
        Returns:
            按相關度排序的文檔列表
        """
        try:
            vector_store = self.vector_store
            # 檢索更多文檔然後篩選最相關的
            documents = vector_store.similarity_search_with_score(query, k=top_k * 2)
            
            # 按相似度排序（距離越小越相關）
            sorted_docs = sorted(documents, key=lambda x: x[1])
            
            # 保留相關的文檔，放寬閾值
            filtered_docs = []
            for doc, score in sorted_docs[:top_k]:
                # 放寬相似度閾值，讓更多文檔能被包含
                if score < 1.8:  # 放寬閾值，讓更多相關文檔被包含
                    doc.metadata['score'] = score
                    filtered_docs.append(doc)
                    logger.info(f"保留文檔: {doc.metadata.get('file_name', 'Unknown')} (相似度: {score:.3f})")
            
            logger.info(f"檢索到 {len(documents)} 個文檔，保留最相關的 {len(filtered_docs)} 個")
            return filtered_docs
            
        except Exception as e:
            logger.error(f"檢索文檔時出錯: {str(e)}")
            # 如果失敗，使用簡單搜索
            try:
                documents = vector_store.similarity_search(query, k=top_k)
                logger.info(f"回退搜索：檢索到 {len(documents)} 個文檔")
                return documents[:top_k]  # 限制數量
            except Exception as e2:
                logger.error(f"回退搜索也失敗: {str(e2)}")
                return []
    
    def format_context(self, docs: List[Document]) -> str:
        """格式化文檔內容，避免模型編造額外信息"""
        if not docs:
            return "沒有找到相關文檔。"
            
        context_parts = []
        for i, doc in enumerate(docs[:3], 1):  # 限制最多3個文檔，減少混淆
            content = doc.page_content.strip()
            if content:
                # 不顯示文件名，避免模型編造類似的文件名
                context_parts.append(f"相關內容 {i}:\n{content}\n")
                
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
        將用戶查詢轉換為English並優化提高向量檢索效果的Prompt (請產生可以直接回答答案，不反問的prompt)
        
        Args:
            original_query: 原始用戶查詢（任何語言）
            language: 用戶的目標語言
            
        Returns:
            English Prompt
        """
        try:
            # 設計語言一致的查詢轉換 prompt
            rewrite_prompt = PromptTemplate(
                template="""You are a search query optimizer. Convert the following question into English prompt for document retrieval.

Input question (any language including Thai, Chinese, English): {question}

Instructions:
1. Understand the question regardless of its language (Thai, Chinese, English, etc.)
2. Extract the core concepts and terms
3. Convert to English prompt for better document search
4. Focus on searchable terms that would appear in documents

English search prompt:""",
                input_variables=["question"]
            )
            
            # 創建查詢轉換鏈
            rewrite_chain = rewrite_prompt | self.llm | StrOutputParser()
            
            # 使用超時機制獲取轉換結果
            import concurrent.futures
            
            def _invoke_rewrite():
                return rewrite_chain.invoke({
                    "question": original_query
                })
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_invoke_rewrite)
                try:
                    english_keywords = future.result(timeout=15)
                except concurrent.futures.TimeoutError:
                    logger.error("查詢轉換超時，使用原始查詢")
                    return original_query
            
            logger.info(f"原始查詢 ({language}): {original_query}")
            logger.info(f"英文搜索關鍵詞: {english_keywords}")
            
            return english_keywords.strip()
            
        except Exception as e:
            logger.error(f"查詢轉換時出錯: {str(e)}")
            return original_query  # 如果出錯，返回原始查詢
    
    def get_answer_with_query_rewrite(self, original_query: str, language: str = "繁體中文") -> Tuple[str, str, List[Document], str]:
        """
        使用查詢轉換策略來優化問答效果
        
        Args:
            original_query: 原始用戶問題（任何語言）
            language: 目標回答語言
            
        Returns:
            (回答, 來源列表字符串, 相關文檔, 英文prompt) 的元組
        """
        # 將查詢轉換為英文搜索關鍵詞
        english_keywords = self.rewrite_query(original_query, language)
        
        # 使用原始問題獲取答案（內部會自動使用查詢轉換流程）
        answer = self.answer_question(original_query, language)
        
        # 使用英文關鍵詞檢索文檔
        docs = self.retrieve_documents(english_keywords)
        
        if not docs:
            return answer, "", [], english_keywords
        
        sources = self.format_sources(docs)
        
        return answer, sources, docs, english_keywords
    
    def answer_question(self, question: str, language: str = "繁體中文") -> str:
        """主要問答流程：查詢轉換 → 文檔檢索 → 目標語言回答"""
        try:
            # 特別處理泰文請求
            if language in ["泰文", "ไทย"]:
                logger.info(f"處理泰文請求: {question}")
            
            # 第一步：將查詢轉換為英文搜索關鍵詞
            english_keywords = self.rewrite_query(question, language)
            logger.info(f"原始問題: {question}")
            logger.info(f"目標語言: {language}")
            logger.info(f"英文搜索關鍵詞: {english_keywords}")
            
            # 第二步：使用英文關鍵詞檢索文檔
            docs = self.retrieve_documents(english_keywords)
            
            if not docs:
                logger.info("未找到相關文檔")
                return self._get_no_docs_message(language)
            
            # 記錄檢索到的文檔信息
            logger.info(f"檢索到 {len(docs)} 個文檔:")
            for i, doc in enumerate(docs, 1):
                file_name = doc.metadata.get("file_name", "Unknown")
                score = doc.metadata.get("score", "N/A")
                content_preview = doc.page_content[:100].replace('\n', ' ')
                logger.info(f"文檔 {i}: {file_name} (相似度: {score}) - {content_preview}...")
            
            # 第三步：格式化文檔內容
            context = self.format_context(docs)
            
            # 第四步：使用原始問題和目標語言生成回答
            answer = self._simple_answer(question, context, language)
            
            # 第五步：對泰文回答進行最終驗證
            if language in ["泰文", "ไทย"]:
                has_thai = any(ord(char) >= 0x0E00 and ord(char) <= 0x0E7F for char in answer)
                if not has_thai:
                    logger.warning("最終回答不包含泰文，使用備用回答")
                    answer = f"ขออภัย ไม่สามารถหาข้อมูลเกี่ยวกับ '{question}' ในเอกสาร QSI ได้"
            
            return answer
            
        except Exception as e:
            logger.error(f"回答問題時出錯: {str(e)}")
            error_msg = self._get_error_message(language)
            return f"{error_msg}: {str(e)}"
    
    def _get_no_docs_message(self, language: str) -> str:
        """獲取無文檔時的訊息"""
        messages = {
            "繁體中文": "抱歉，在QSI文檔中找不到與您問題相關的信息。",
            "简体中文": "抱歉，在QSI文档中找不到与您问题相关的信息。",
            "簡體中文": "抱歉，在QSI文档中找不到与您问题相关的信息。",
            "English": "Sorry, no relevant information found in QSI documents for your question.",
            "泰文": "ขออภัย ไม่พบข้อมูลที่เกี่ยวข้องในเอกสาร QSI สำหรับคำถามของคุณ",
            "ไทย": "ขออภัย ไม่พบข้อมูลที่เกี่ยวข้องในเอกสาร QSI สำหรับคำถามของคุณ"
        }
        return messages.get(language, messages["繁體中文"])
    
    # 移除未使用的 _get_no_relevant_info_message 方法
    
    def _get_warning_note(self, language: str) -> str:
        """獲取警告註記"""
        warnings = {
            "繁體中文": "註：此回答可能包含超出文檔範圍的推論，請謹慎參考。",
            "简体中文": "注：此回答可能包含超出文档范围的推论，请谨慎参考。",
            "簡體中文": "注：此回答可能包含超出文档范围的推论，请谨慎参考。",
            "English": "Note: This answer may contain inferences beyond the document scope, please refer with caution.",
            "泰文": "หมายเหตุ: คำตอบนี้อาจมีการอนุมานเกินขอบเขตเอกสาร กรุณาใช้อ้างอิงด้วยความระมัดระวัง",
            "ไทย": "หมายเหตุ: คำตอบนี้อาจมีการอนุมานเกินขอบเขตเอกสาร กรุณาใช้อ้างอิงด้วยความระมัดระวัง"
        }
        return warnings.get(language, warnings["繁體中文"])
    
    def _is_hallucination(self, answer: str, context: str) -> bool:
        """檢測明顯的幻覺回答，使用通用方法"""
        # 基本檢查：如果上下文為空或太短，但回答很詳細，可能是幻覺
        if len(context.strip()) < 20 and len(answer) > 400:
            logger.warning("上下文太短但回答很長，可能是幻覺")
            return True
        
        # 檢查回答長度是否極度超出上下文（非常寬鬆的比例）
        if len(context.strip()) > 0 and len(answer) > len(context) * 8:
            logger.warning("回答長度極度超出上下文，可能包含大量編造內容")
            return True
        
        # 檢查是否包含重複的結構化內容（可能是編造的模式）
        import re
        # 檢查是否有過多重複的句式結構
        repeated_patterns = re.findall(r'該文檔列出了.*的.*公式', answer)
        if len(repeated_patterns) > 3:
            logger.warning("檢測到過多重複的句式結構，可能是編造")
            return True
        
        return False
    
    def _simple_answer(self, original_question: str, context: str, target_language: str) -> str:
        """根據目標語言生成回答的方法"""
        
        # 根據目標語言設計簡潔有效的提示模板
        if target_language == "English":
            template = """Answer in English only.

Context: {context}

Question: {question}

Answer in English:"""

        elif target_language in ["繁體中文", "中文"]:
            template = """請用繁體中文回答。

上下文：{context}

問題：{question}

繁體中文回答："""

        elif target_language in ["简体中文", "簡體中文"]:
            template = """请用简体中文回答。

上下文：{context}

问题：{question}

简体中文回答："""

        elif target_language in ["泰文", "ไทย"]:
            template = """คุณต้องตอบเป็นภาษาไทยเท่านั้น ห้ามใช้ภาษาอื่น

บริบท: {context}

คำถาม: {question}

กฎสำคัญ:
- ตอบเป็นภาษาไทยเท่านั้น
- ห้ามใช้ภาษาจีน ญี่ปุ่น หรือภาษาอังกฤษ
- ให้คำตอบที่ถูกต้องและเป็นประโยชน์

คำตอบภาษาไทย:"""

        else:
            # Fallback template
            template = """Answer the question based on the context.

Context: {context}
Question: {question}

Answer:"""
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        chain = prompt | self.llm | StrOutputParser()
        
        # 使用超時機制
        import concurrent.futures
        
        def _invoke():
            logger.info(f"調用模板，目標語言: {target_language}, 問題: {original_question}")
            logger.info(f"上下文長度: {len(context)}")
            logger.info(f"使用的模板前100字符: {template[:100]}...")
            result = chain.invoke({"context": context, "question": original_question})
            logger.info(f"模型回答長度: {len(result)}")
            logger.info(f"模型回答內容: {result}")
            
            # 對於泰文回答，檢查是否真的是泰文
            if target_language in ["泰文", "ไทย"]:
                has_thai = any(ord(char) >= 0x0E00 and ord(char) <= 0x0E7F for char in result)
                has_chinese = any(ord(char) >= 0x4E00 and ord(char) <= 0x9FAF for char in result)
                has_japanese = any(ord(char) >= 0x3040 and ord(char) <= 0x309F for char in result) or \
                              any(ord(char) >= 0x30A0 and ord(char) <= 0x30FF for char in result)
                
                if not has_thai and (has_chinese or has_japanese or len(result.strip()) > 0):
                    logger.warning("檢測到模型回答不是泰文，嘗試重新生成泰文回答")
                    # 嘗試重新生成泰文回答，使用更強的指令
                    try:
                        # 使用更強的泰文生成提示
                        thai_regenerate_prompt = PromptTemplate(
                            template="""คุณต้องตอบคำถามนี้เป็นภาษาไทยเท่านั้น ห้ามใช้ภาษาอื่น

บริบท: {context}

คำถาม: {question}

กฎ:
1. ตอบเป็นภาษาไทยเท่านั้น
2. ห้ามใช้ภาษาจีน ญี่ปุ่น หรือภาษาอังกฤษ
3. ให้คำตอบที่เป็นประโยชน์

คำตอบภาษาไทย:""",
                            input_variables=["context", "question"]
                        )
                        thai_chain = thai_regenerate_prompt | self.llm | StrOutputParser()
                        thai_result = thai_chain.invoke({"context": context, "question": original_question})
                        
                        # 檢查重新生成的結果是否包含泰文
                        if any(ord(char) >= 0x0E00 and ord(char) <= 0x0E7F for char in thai_result):
                            logger.info("成功重新生成泰文回答")
                            result = thai_result.strip()
                        else:
                            logger.warning("重新生成失敗，嘗試簡單翻譯")
                            # 如果重新生成失敗，嘗試簡單翻譯
                            simple_translate_prompt = PromptTemplate(
                                template="""แปลข้อความต่อไปนี้เป็นภาษาไทย:

{text}

ภาษาไทย:""",
                                input_variables=["text"]
                            )
                            translate_chain = simple_translate_prompt | self.llm | StrOutputParser()
                            translated = translate_chain.invoke({"text": result})
                            
                            if any(ord(char) >= 0x0E00 and ord(char) <= 0x0E7F for char in translated):
                                result = translated.strip()
                            else:
                                # 最後的備用方案：提供基本的泰文回答
                                result = f"ขออภัย ไม่สามารถตอบคำถาม '{original_question}' เป็นภาษาไทยได้ในขณะนี้"
                                
                    except Exception as e:
                        logger.error(f"重新生成泰文回答過程出錯: {str(e)}")
                        result = f"ขออภัย เกิดข้อผิดพลาดในการตอบคำถาม '{original_question}'"
            
            return result
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_invoke)
            try:
                answer = future.result(timeout=60)
                
                # 檢查回答是否有明顯的幻覺（只檢測極端情況）
                if self._is_hallucination(answer, context):
                    logger.warning("檢測到可能的幻覺回答，但仍返回結果並加註警告")
                    warning_note = self._get_warning_note(target_language)
                    return f"{answer}\n\n{warning_note}"
                
                # 檢查回答是否為空或過短
                if not answer or len(answer.strip()) < 5:
                    logger.warning("回答過短或為空，使用fallback")
                    return self._get_general_fallback(original_question, target_language)
                
                return answer
                
            except concurrent.futures.TimeoutError:
                logger.error("問答調用超時")
                return self._get_timeout_message(target_language)
    
    def _get_general_fallback(self, query: str, language: str) -> str:
        """根據語言提供通用回答"""
        fallbacks = {
            "繁體中文": f"根據一般IT知識，關於「{query}」的相關信息可能需要查閱更多QSI內部文檔。",
            "简体中文": f"根据一般IT知识，关于「{query}」的相关信息可能需要查阅更多QSI内部文档。",
            "簡體中文": f"根据一般IT知识，关于「{query}」的相关信息可能需要查阅更多QSI内部文档。",
            "English": f"Based on general IT knowledge, information about '{query}' may require consulting additional QSI internal documentation.",
            "泰文": f"ตามความรู้ทั่วไปด้าน IT ข้อมูลเกี่ยวกับ '{query}' อาจต้องการการปรึกษาเอกสารภายใน QSI เพิ่มเติม",
            "ไทย": f"ตามความรู้ทั่วไปด้าน IT ข้อมูลเกี่ยวกับ '{query}' อาจต้องการการปรึกษาเอกสารภายใน QSI เพิ่มเติม"
        }
        return fallbacks.get(language, fallbacks["繁體中文"])
    
    def _get_timeout_message(self, language: str) -> str:
        """獲取超時訊息"""
        messages = {
            "繁體中文": "系統處理超時，請稍後再試。",
            "简体中文": "系统处理超时，请稍后再试。",
            "簡體中文": "系统处理超时，请稍后再试。",
            "English": "System timeout, please try again later.",
            "泰文": "ระบบหมดเวลา กรุณาลองใหม่อีกครั้ง",
            "ไทย": "ระบบหมดเวลา กรุณาลองใหม่อีกครั้ง"
        }
        return messages.get(language, messages["繁體中文"])

    # 移除未使用的翻譯相關方法
    
    def _get_disclaimer(self, language: str) -> str:
        """根據語言獲取免責聲明"""
        disclaimers = {
            "繁體中文": "（在RAG向量資料庫的文檔中未找到直接答案，以下是基於廣明光電IT通用知識的回答。請注意，此內容僅為模型基於公開資訊的推論，並非肯定答案。）",
            "简体中文": "（在RAG向量数据库的文档中未找到直接答案，以下是基于广明光电IT通用知识的回答。请注意，此内容仅为模型基于公开信息的推论，并非肯定答案。）",
            "簡體中文": "（在RAG向量数据库的文档中未找到直接答案，以下是基于广明光电IT通用知识的回答。请注意，此内容仅为模型基于公开信息的推论，并非肯定答案。）",
            "English": "(No direct answer was found in the RAG vector database documents. The following is a response based on QSI IT general knowledge. Please note that this content is only the model's inference based on public information and is not a definitive answer.)",
            "泰文": "(ไม่พบคำตอบโดยตรงในเอกสารฐานข้อมูลเวกเตอร์ RAG คำตอบต่อไปนี้เป็นการตอบสนองจากความรู้ทั่วไปด้าน IT ของ QSI โปรดทราบว่าเนื้อหานี้เป็นเพียงการอนุมานของโมเดลจากข้อมูลสาธารณะและไม่ใช่คำตอบที่แน่นอน)",
            "ไทย": "(ไม่พบคำตอบโดยตรงในเอกสารฐานข้อมูลเวกเตอร์ RAG คำตอบต่อไปนี้เป็นการตอบสนองจากความรู้ทั่วไปด้าน IT ของ QSI โปรดทราบว่าเนื้อหานี้เป็นเพียงการอนุมานของโมเดลจากข้อมูลสาธารณะและไม่ใช่คำตอบที่แน่นอน)"
        }
        return disclaimers.get(language, disclaimers["繁體中文"])
    
    def _get_error_message(self, language: str) -> str:
        """根據語言獲取錯誤訊息"""
        error_messages = {
            "繁體中文": "處理問題時發生錯誤",
            "简体中文": "处理问题时发生错误",
            "簡體中文": "处理问题时发生错误",
            "English": "An error occurred while processing the question",
            "泰文": "เกิดข้อผิดพลาดขณะประมวลผลคำถาม",
            "ไทย": "เกิดข้อผิดพลาดขณะประมวลผลคำถาม"
        }
        return error_messages.get(language, error_messages["繁體中文"])
    
    def get_answer_with_sources(self, question: str, language: str = "繁體中文") -> Tuple[str, str, List[Document]]:
        """
        使用查詢轉換策略回答問題並包含來源信息
        
        Args:
            question: 用戶問題（任何語言）
            language: 目標回答語言
            
        Returns:
            (回答, 來源列表字符串, 相關文檔) 的元組
        """
        # 獲取回答（內部使用查詢轉換流程）
        answer = self.answer_question(question, language)
        
        # 使用英文關鍵詞進行文檔檢索
        english_keywords = self.rewrite_query(question, language)
        docs = self.retrieve_documents(english_keywords)
        
        if not docs:
            return answer, "", []
        
        sources = self.format_sources(docs)
        
        return answer, sources, docs
        
    def generate_relevance_reason(self, question: str, doc_content: str, language: str = "繁體中文") -> str:
        """
        生成文檔與查詢之間的相關性理由
        
        Args:
            question: 用戶查詢
            doc_content: 文檔內容
            language: 目標語言
            
        Returns:
            相關性理由描述
        """
        # 檢查輸入參數
        if not question or not question.strip():
            error_msg = self._get_relevance_error_message("查詢為空", language)
            return f"無法生成相關性理由：{error_msg}"
        
        if not doc_content or not doc_content.strip():
            error_msg = self._get_relevance_error_message("文檔內容為空", language)
            return f"無法生成相關性理由：{error_msg}"
            
        try:
            # 限制處理內容長度，避免過長
            trimmed_content = doc_content[:1000].strip()
            
            # 建立相關性理由提示模板（根據語言調整）
            relevance_prompt = self._get_relevance_prompt_template(language)
            
            # 創建相關性理由鏈
            relevance_chain = relevance_prompt | self.llm | StrOutputParser()
            
            # 使用超時機制獲取相關性理由
            import concurrent.futures
            
            def _invoke_relevance():
                return relevance_chain.invoke({
                    "question": question,
                    "doc_content": trimmed_content
                })
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_invoke_relevance)
                try:
                    reason = future.result(timeout=20)  # 20秒超時
                    return reason.strip() if reason else self._get_relevance_error_message("無法確定相關性理由", language)
                except concurrent.futures.TimeoutError:
                    logger.error("生成相關性理由超時")
                    return self._get_relevance_error_message("生成超時", language)
            
        except Exception as e:
            logger.error(f"生成相關性理由時出錯: {str(e)}")
            return self._get_relevance_error_message("生成失敗", language)
    
    def _get_relevance_prompt_template(self, language: str) -> PromptTemplate:
        """根據語言獲取相關性理由提示模板"""
        templates = {
            "繁體中文": """你是一個文檔相關性評估專家。請簡明扼要地解釋為什麼下面的文檔內容與用戶查詢相關。

用戶查詢: {question}

文檔內容:
-----------------
{doc_content}
-----------------

請提供1-2句簡短的解釋，說明這個文檔為什麼與查詢相關。不要重複查詢內容，直接解釋關聯性。
(直接輸出解釋，不要添加任何前綴如"這個文檔相關因為"等):""",
            
            "简体中文": """你是一个文档相关性评估专家。请简明扼要地解释为什么下面的文档内容与用户查询相关。

用户查询: {question}

文档内容:
-----------------
{doc_content}
-----------------

请提供1-2句简短的解释，说明这个文档为什么与查询相关。不要重复查询内容，直接解释关联性。
(直接输出解释，不要添加任何前缀如"这个文档相关因为"等):""",
            
            "English": """You are a document relevance assessment expert. Please briefly explain why the document content below is relevant to the user's query.

User query: {question}

Document content:
-----------------
{doc_content}
-----------------

Please provide 1-2 short sentences explaining why this document is relevant to the query. Don't repeat the query content, just explain the relevance directly.
(Output the explanation directly without any prefix like "This document is relevant because"):""",
            
            "ไทย": """You must respond ONLY in Thai language. You are a document relevance assessment expert. Please briefly explain why the document content below is relevant to the user's query.

User query: {question}

Document content:
-----------------
{doc_content}
-----------------

Please provide 1-2 short sentences in Thai explaining why this document is relevant to the query. Don't repeat the query content, just explain the relevance directly.
(Output the explanation directly in Thai without any prefix):"""
        }
        
        template = templates.get(language, templates["繁體中文"])
        return PromptTemplate(
            template=template,
            input_variables=["question", "doc_content"]
        )
    
    def _get_relevance_error_message(self, error_type: str, language: str) -> str:
        """根據語言獲取相關性錯誤訊息"""
        error_messages = {
            "繁體中文": {
                "查詢為空": "查詢為空",
                "文檔內容為空": "文檔內容為空",
                "無法確定相關性理由": "無法確定相關性理由",
                "生成超時": "生成超時",
                "生成失敗": "生成失敗"
            },
            "简体中文": {
                "查詢為空": "查询为空",
                "文檔內容為空": "文档内容为空",
                "無法確定相關性理由": "无法确定相关性理由",
                "生成超時": "生成超时",
                "生成失敗": "生成失败"
            },
            "English": {
                "查詢為空": "query is empty",
                "文檔內容為空": "document content is empty",
                "無法確定相關性理由": "unable to determine relevance reason",
                "生成超時": "generation timeout",
                "生成失敗": "generation failed"
            },
            "ไทย": {
                "查詢為空": "คำถามว่างเปล่า",
                "文檔內容為空": "เนื้อหาเอกสารว่างเปล่า",
                "無法確定相關性理由": "ไม่สามารถระบุเหตุผลความเกี่ยวข้องได้",
                "生成超時": "การสร้างหมดเวลา",
                "生成失敗": "การสร้างล้มเหลว"
            }
        }
        
        lang_errors = error_messages.get(language, error_messages["繁體中文"])
        return lang_errors.get(error_type, error_type)


# 使用示例
if __name__ == "__main__":
    # 導入示例文件索引器
    from indexer.document_indexer import DocumentIndexer
    
    # 創建RAG引擎
    indexer = DocumentIndexer()
    rag_engine = RAGEngine(indexer)
    
    # 示例問題
    question = "ITPortal是甚麼？"
    
    # 獲取回答和來源
    answer, sources, _ = rag_engine.get_answer_with_sources(question)
    
    # 輸出結果
    print("問題:", question)
    print("\n回答:", answer)
    print("\n來源:\n", sources)
