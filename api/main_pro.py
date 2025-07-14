import fire
import logging
from datetime import datetime

# --- 核心 LangChain 組件 ---
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

# --- 升級點 1: 使用指定的 HuggingFace 嵌入模型 ---
from langchain.embeddings import HuggingFaceEmbeddings
import torch

# --- 升級點 2: 直接在腳本中載入並運行 LLM (不再依賴外部 Ollama 服務) ---
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA

# --- 日誌設定 ---
log_filename = f"log_main_pro_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)

class RagForQ:
    def __init__(self,
                 vector_db_path: str = "faiss_index_pro", # 新的索引路徑
                 source_dir: str = "source_documents",
                 embedding_model_name: str = 'bge-large-zh-v1.5', # 指定頂級中文嵌入模型
                 llm_model_path: str = "llama-2-7b-chat.Q5_K_M.gguf" # 指定本地 LLM 模型檔案
                 ):
        self.vector_db_path = vector_db_path
        self.source_dir = source_dir
        self.embedding_model_name = embedding_model_name
        self.llm_model_path = llm_model_path
        
        # 檢測並設定設備 (GPU or CPU)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"檢測到可用設備: {self.device}")

    def _load_embedding_model(self):
        """載入嵌入模型到指定設備"""
        logging.info(f"正在從 HuggingFace 載入嵌入模型: {self.embedding_model_name}")
        embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model_name,
            model_kwargs={'device': self.device},
            encode_kwargs={'normalize_embeddings': True}
        )
        logging.info("嵌入模型載入成功。")
        return embeddings

    def create_vector_db(self):
        """
        建立向量資料庫 (FAISS-GPU)。
        """
        logging.info(f"從 '{self.source_dir}' 載入文件...")
        loader = DirectoryLoader(self.source_dir, glob="**/*.*", show_progress=True, use_multithreading=True)
        documents = loader.load()
        if not documents:
            logging.error(f"在 '{self.source_dir}' 中未找到文件。")
            return

        logging.info(f"載入 {len(documents)} 份文件。開始進行文本切割...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(documents)
        logging.info(f"文件被切割成 {len(docs)} 個片段。")

        embeddings = self._load_embedding_model()

        logging.info("開始建立 FAISS 向量索引。這將利用 GPU 加速...")
        start_time = datetime.now()
        # 升級點 3: 使用 FAISS.from_documents，它會自動使用 GPU (如果 faiss-gpu 已安裝)
        db = FAISS.from_documents(docs, embeddings)
        db.save_local(self.vector_db_path)
        end_time = datetime.now()

        logging.info(f"向量資料庫建立完成並儲存至 '{self.vector_db_path}'。耗時: {end_time - start_time}")

    def q(self, question: str):
        """
        根據問題在知識庫中進行查詢並獲得答案。
        """
        logging.info(f"收到的問題: {question}")
        
        # --- 步驟 1: 載入 LLM 模型 ---
        logging.info(f"正在從 '{self.llm_model_path}' 載入 LLM...")
        # CTransformers 支援 GGUF 格式，並可以利用 GPU 加速
        # n_gpu_layers > 0 會將一部分模型層卸載到 GPU
        # 您的 4090 有 24GB VRAM，可以設定一個很大的值，例如 50 或更高，幾乎把整個模型都放進去
        llm = CTransformers(
            model=self.llm_model_path,
            model_type='llama',
            config={'max_new_tokens': 1024, 'temperature': 0.1, 'context_length': 4096},
            n_gpu_layers=50  # <--- 關鍵的 GPU 加速參數！
        )
        logging.info("LLM 載入成功。")

        # --- 步驟 2: 載入向量資料庫和嵌入模型 ---
        embeddings = self._load_embedding_model()
        logging.info(f"正在從 '{self.vector_db_path}' 載入 FAISS 索引...")
        db = FAISS.load_local(self.vector_db_path, embeddings)
        logging.info("FAISS 索引載入成功。")

        # --- 步驟 3: 建立檢索器和問答鏈 ---
        # as_retriever 會使用向量資料庫進行相似度搜索
        retriever = db.as_retriever(search_kwargs={"k": 5}) # 找出最相關的 5 個文件片段
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff", # "stuff" 會將找到的文件片段全部塞進 prompt
            retriever=retriever,
            return_source_documents=True
        )

        logging.info("開始進行檢索與生成答案...")
        start_time = datetime.now()
        result = qa_chain({"query": question})
        end_time = datetime.now()
        
        logging.info(f"答案生成完成。耗時: {end_time - start_time}")

        # --- 步驟 4: 格式化並輸出結果 ---
        print("\n\n> 問題:")
        print(result["query"])
        print("\n> 答案:")
        print(result["result"])
        print("\n> 參考資料來源:")
        for doc in result["source_documents"]:
            # 打印來源文件的名稱和一小段內容預覽
            print(f"  - 來源: {doc.metadata.get('source', '未知')}, 內容片段: '{doc.page_content[:100]}...'")

if __name__ == '__main__':
    fire.Fire(RagForQ)