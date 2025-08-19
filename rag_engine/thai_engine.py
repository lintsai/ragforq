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

class ThaiRAGEngine(RAGEngineInterface):
    """Thai RAG Engine Implementation - เครื่องมือ RAG ภาษาไทย"""
    
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
            logger.info(f"Thai RAG engine initialized (Ollama) with model: {ollama_model}")
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
            logger.info(f"Thai RAG engine initialized (Hugging Face) with model: {ollama_model} with params: {llm_params}")
    
    def get_language(self) -> str:
        return "ไทย"
    
    def rewrite_query(self, original_query: str) -> str:
        """
        ปรับปรุงคำถามภาษาไทยให้เป็นคำอธิบายที่แม่นยำและเหมาะสมสำหรับการค้นหาเวกเตอร์ - พร้อมกลไกลองใหม่
        """
        from config.config import OLLAMA_MAX_RETRIES, OLLAMA_RETRY_DELAY, OLLAMA_QUERY_OPTIMIZATION_TIMEOUT
        import time
        
        for attempt in range(OLLAMA_MAX_RETRIES):
            try:
                rewrite_prompt = PromptTemplate(
                    template="""คุณเป็นผู้เชี่ยวชาญด้านการปรับแต่งการค้นหา กรุณาแปลงคำถามต่อไปนี้ให้เป็นคำอธิบายที่สมบูรณ์และเหมาะสมสำหรับการค้นหาในฐานความรู้

ข้อกำหนด:
1. ใช้ภาษาไทยเท่านั้น
2. ขยายคำศัพท์เฉพาะทางและคำพ้องความหมายที่เกี่ยวข้อง
3. รวมการแสดงออกที่แตกต่างกันที่อาจปรากฏในเอกสาร
4. รักษาแนวคิดหลักและความหมายเชิงความหมายของคำถาม

คำถามเดิม: {question}

คำค้นหาที่ปรับปรุงแล้ว:""",
                    input_variables=["question"]
                )
                
                def _invoke_rewrite():
                    # การเรียกใช้โดยตรงโดยไม่ใช้การดำเนินการแบบลูกโซ่
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
                        logger.info(f"Thai query optimization: {original_query} -> {optimized_query}")
                        return optimized_query.strip()
                    except concurrent.futures.TimeoutError:
                        if attempt < OLLAMA_MAX_RETRIES - 1:
                            logger.warning(f"การปรับแต่งคำค้นหาหมดเวลา ลองใหม่ครั้งที่ {attempt + 1}...")
                            time.sleep(OLLAMA_RETRY_DELAY)
                            continue
                        else:
                            logger.error("การปรับแต่งคำค้นหาหมดเวลาหลายครั้ง ใช้คำค้นหาเดิม")
                            return original_query
                
            except Exception as e:
                if attempt < OLLAMA_MAX_RETRIES - 1:
                    logger.warning(f"การปรับแต่งคำค้นหาผิดพลาด ลองใหม่ครั้งที่ {attempt + 1}: {str(e)}")
                    time.sleep(OLLAMA_RETRY_DELAY)
                    continue
                else:
                    logger.error(f"การปรับแต่งคำค้นหาล้มเหลวหลายครั้ง: {str(e)}")
                    return original_query
        
        return original_query
    
    def answer_question(self, question: str) -> str:
        """ตอบคำถามเป็นภาษาไทย"""
        try:
            # ปรับปรุงคำค้นหา
            optimized_query = self.rewrite_query(question)
            
            # ค้นหาเอกสาร
            docs = self.retrieve_documents(optimized_query)
            
            if not docs:
                return self._generate_general_knowledge_answer(question)
            
            # จัดรูปแบบบริบท
            context = self.format_context(docs)
            
            # สร้างคำตอบ
            return self._generate_answer(question, context)
            
        except Exception as e:
            logger.error(f"Error during Thai Q&A: {str(e)}")
            return f"{self._get_error_message()}: {str(e)}"
    
    def _generate_answer(self, question: str, context: str) -> str:
        """สร้างคำตอบภาษาไทย"""
        template = """คุณเป็นผู้ช่วย AI มืออาชีพสำหรับเอกสาร โปรดตอบ "คำถามของผู้ใช้" โดยอิงจาก "ข้อมูลบริบท"

**ข้อกำหนดงาน:**
1.  **ตอบโดยตรง:** ให้คำตอบหลักโดยตรงสำหรับ "คำถามของผู้ใช้" โดยละเว้นการแนะนำหรือข้อมูลพื้นฐานที่ไม่จำเป็น
2.  **กระชับ:** คำตอบควรกระชับและแม่นยำที่สุดเท่าที่จะทำได้ หลีกเลี่ยงคำอธิบายที่ยาวและเนื้อหาที่ซ้ำซาก
3.  **อิงตามบริบท:** คำตอบต้องอิงตาม "ข้อมูลบริบท" ทั้งหมด
4.  **ภาษาที่สอดคล้องกัน:** ตอบด้วยภาษาเดียวกับคำถาม (ภาษาไทย)
5.  **การจัดการที่ไม่รู้จัก:** หาก "ข้อมูลบริบท" ไม่เพียงพอที่จะตอบ ให้ตอบเพียง "จากเอกสารที่ให้มา ฉันไม่พบข้อมูลที่เกี่ยวข้อง"

**ข้อมูลบริบท:**
---
{context}
---

**คำถามของผู้ใช้:** {question}

**คำตอบ:**
"""

        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        chain = prompt | self.llm | StrOutputParser()
        
        def _invoke():
            result = chain.invoke({"context": context, "question": question})
            
            # ตรวจสอบว่าคำตอบเป็นภาษาไทยจริงหรือไม่
            has_thai = any(ord(char) >= 0x0E00 and ord(char) <= 0x0E7F for char in result)
            if not has_thai and result.strip():
                logger.warning("Generated answer doesn't contain Thai characters, attempting regeneration")
                # พยายามสร้างใหม่ด้วยคำสั่งที่เข้มงวดกว่า
                strict_template = """ตอบคำถามนี้เป็นภาษาไทยเท่านั้น ห้ามใช้ภาษาอื่นโดยเด็ดขาด

บริบท: {context}
คำถาม: {question}

กฎ:
- ใช้เฉพาะภาษาไทย
- ห้ามใช้ภาษาอังกฤษ จีน หรือภาษาอื่น
- ให้คำตอบที่เป็นประโยชน์

คำตอบภาษาไทย:"""
                
                strict_prompt = PromptTemplate(
                    template=strict_template,
                    input_variables=["context", "question"]
                )
                strict_chain = strict_prompt | self.llm | StrOutputParser()
                
                try:
                    strict_result = strict_chain.invoke({"context": context, "question": question})
                    if any(ord(char) >= 0x0E00 and ord(char) <= 0x0E7F for char in strict_result):
                        return strict_result
                except Exception:
                    pass
                
                # หากยังไม่ได้ผล ใช้คำตอบสำรอง
                return f"ขออภัย ไม่สามารถตอบคำถาม '{question}' เป็นภาษาไทยได้ในขณะนี้"
            
            return result
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_invoke)
            try:
                from config.config import OLLAMA_ANSWER_GENERATION_TIMEOUT
                answer = future.result(timeout=OLLAMA_ANSWER_GENERATION_TIMEOUT)
                
                if not answer or len(answer.strip()) < 5:
                    return self._get_general_fallback(question)
                
                return answer.strip()
                
            except concurrent.futures.TimeoutError:
                logger.error("Thai answer generation timeout")
                return self._get_timeout_message()
    
    def generate_relevance_reason(self, question: str, doc_content: str) -> str:
        """สร้างเหตุผลความเกี่ยวข้องเป็นภาษาไทย"""
        if not question or not question.strip():
            return "ไม่สามารถสร้างเหตุผลความเกี่ยวข้องได้: คำถามว่างเปล่า"
        
        if not doc_content or not doc_content.strip():
            return "ไม่สามารถสร้างเหตุผลความเกี่ยวข้องได้: เนื้อหาเอกสารว่างเปล่า"
            
        try:
            trimmed_content = doc_content[:1000].strip()
            
            relevance_prompt = PromptTemplate(
                template="""คุณเป็นผู้เชี่ยวชาญในการประเมินความเกี่ยวข้องของเอกสาร กรุณาอธิบายสั้นๆ ว่าทำไมเนื้อหาเอกสารด้านล่างจึงเกี่ยวข้องกับคำถามของผู้ใช้ กรุณาตอบเป็นภาษาไทยเท่านั้น

คำถามของผู้ใช้: {question}
เนื้อหาเอกสาร:
---
{doc_content}
---
เหตุผลความเกี่ยวข้อง:""",
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
                    return reason.strip() if reason else "ไม่สามารถระบุเหตุผลความเกี่ยวข้องได้"
                except concurrent.futures.TimeoutError:
                    return "การสร้างเหตุผลความเกี่ยวข้องหมดเวลา"
            
        except Exception as e:
            logger.error(f"Error generating Thai relevance reason: {str(e)}")
            return "การสร้างเหตุผลความเกี่ยวข้องล้มเหลว"
    
    def _get_no_docs_message(self) -> str:
        return "ขออภัย ไม่พบข้อมูลที่เกี่ยวข้องในเอกสาร QSI สำหรับคำถามของคุณ"
    
    def _get_error_message(self) -> str:
        return "เกิดข้อผิดพลาดขณะประมวลผลคำถาม"
    
    def _get_timeout_message(self) -> str:
        return "ระบบหมดเวลา กรุณาลองใหม่อีกครั้ง"
    
    def _generate_general_knowledge_answer(self, question: str) -> str:
        """ให้คำตอบจากความรู้ทั่วไปเมื่อไม่พบเอกสารที่เกี่ยวข้อง"""
        try:
            general_prompt = PromptTemplate(
                template="""คุณเป็นผู้ช่วยผู้เชี่ยวชาญด้าน IT แม้ว่าจะไม่พบข้อมูลที่เกี่ยวข้องในเอกสารภายใน QSI แต่กรุณาให้คำตอบโดยอิงจากความรู้ทั่วไปด้าน IT

คำถาม: {question}

กรุณาทราบ:
1. ตอบเป็นภาษาไทยเท่านั้น
2. ให้คำตอบที่เป็นประโยชน์โดยอิงจากความรู้ทั่วไปด้าน IT
3. ระบุอย่างชัดเจนว่านี่เป็นคำตอบจากความรู้ทั่วไป ไม่ใช่จากเอกสารภายใน QSI
4. หากเป็นคำถามเฉพาะของ QSI แนะนำให้ติดต่อแผนกที่เกี่ยวข้อง

คำตอบภาษาไทย:""",
                input_variables=["question"]
            )
            
            general_chain = general_prompt | self.llm | StrOutputParser()
            
            def _invoke_general():
                return general_chain.invoke({"question": question})
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_invoke_general)
                try:
                    answer = future.result(timeout=30)
                    # เพิ่มข้อความปฏิเสธความรับผิดชอบ
                    disclaimer = "\n\n※ หมายเหตุ: คำตอบข้างต้นอิงจากความรู้ทั่วไปด้าน IT ไม่ใช่จากเอกสารภายใน QSI หากต้องการข้อมูลที่แม่นยำ กรุณาติดต่อแผนกที่เกี่ยวข้อง"
                    return answer.strip() + disclaimer
                except concurrent.futures.TimeoutError:
                    logger.error("การสร้างคำตอบจากความรู้ทั่วไปหมดเวลา")
                    return self._get_no_docs_message()
        
        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดในการสร้างคำตอบจากความรู้ทั่วไป: {str(e)}")
            return self._get_no_docs_message()
    
    def generate_batch_relevance_reasons(self, question: str, doc_contents: list) -> list:
        """สร้างเหตุผลความเกี่ยวข้องสำหรับเอกสารหลายฉบับแบบกลุ่ม เพื่อเพิ่มประสิทธิภาพ"""
        if not question or not question.strip() or not doc_contents:
            return ["ไม่สามารถสร้างเหตุผลความเกี่ยวข้องได้"] * len(doc_contents)
        
        try:
            # สร้าง prompt สำหรับการประมวลผลแบบกลุ่ม
            docs_text = ""
            for i, content in enumerate(doc_contents, 1):
                if content and content.strip():
                    docs_text += f"เอกสาร{i}: {content[:300]}...\n\n"
                else:
                    docs_text += f"เอกสาร{i}: (เนื้อหาว่าง)\n\n"
            
            batch_prompt = PromptTemplate(
                template="""กรุณาสร้างเหตุผลความเกี่ยวข้องสำหรับเอกสารต่อไปนี้เทียบกับคำถามของผู้ใช้ แต่ละเหตุผลควรเป็นประโยคสั้นๆ กระชับ

คำถามของผู้ใช้: {question}

เนื้อหาเอกสาร:
{docs_text}

กรุณาสร้างเหตุผลความเกี่ยวข้องสำหรับแต่ละเอกสารตามลำดับ ใช้รูปแบบนี้:
1. [เหตุผลความเกี่ยวข้องสำหรับเอกสาร 1]
2. [เหตุผลความเกี่ยวข้องสำหรับเอกสาร 2]
3. [เหตุผลความเกี่ยวข้องสำหรับเอกสาร 3]
...

เหตุผลความเกี่ยวข้อง:""",
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
                    
                    # แยกวิเคราะห์ผลลัพธ์แบบกลุ่ม
                    reasons = []
                    lines = batch_result.strip().split('\n')
                    for line in lines:
                        line = line.strip()
                        if line and (line.startswith(('1.', '2.', '3.', '4.', '5.')) or '.' in line[:3]):
                            # ลบหมายเลข เก็บเหตุผล
                            reason = line.split('.', 1)[1].strip() if '.' in line else line
                            reasons.append(reason)
                    
                    # ให้แน่ใจว่าส่งคืนจำนวนเหตุผลที่ถูกต้อง
                    while len(reasons) < len(doc_contents):
                        reasons.append("เอกสารที่เกี่ยวข้อง")
                    
                    return reasons[:len(doc_contents)]
                    
                except concurrent.futures.TimeoutError:
                    logger.error("การสร้างเหตุผลความเกี่ยวข้องแบบกลุ่มหมดเวลา")
                    return [f"เอกสารที่เกี่ยวข้อง {i+1}" for i in range(len(doc_contents))]
        
        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดในการสร้างเหตุผลความเกี่ยวข้องแบบกลุ่ม: {str(e)}")
            return [f"เอกสารที่เกี่ยวข้อง {i+1}" for i in range(len(doc_contents))]
    
    def _get_general_fallback(self, query: str) -> str:
        return f"ตามความรู้ทั่วไปด้าน IT ข้อมูลเกี่ยวกับ '{query}' อาจต้องการการปรึกษาเอกสารภายใน QSI เพิ่มเติม"
    
    
