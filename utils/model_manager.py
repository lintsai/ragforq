"""
Hugging Face 模型管理器
統一管理語言模型和嵌入模型的加載、緩存和優化
"""

import os
import logging
import torch
# TensorFlow 已移除，僅使用 PyTorch
from typing import Optional, Dict, Any, Union
from pathlib import Path
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoModelForSeq2SeqLM,
    pipeline, BitsAndBytesConfig
)
from sentence_transformers import SentenceTransformer
import numpy as np

from config.config import (
    HF_MODEL_CACHE_DIR, HF_USE_GPU, TORCH_DEVICE, TORCH_DTYPE,
    TF_MEMORY_GROWTH, TF_MIXED_PRECISION, DEFAULT_LLM_MODEL, DEFAULT_EMBEDDING_MODEL,
    INFERENCE_ENGINE
)

logger = logging.getLogger(__name__)

class ModelManager:
    """統一的模型管理器"""
    
    def __init__(self):
        self.cache_dir = Path(HF_MODEL_CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 模型緩存
        self._llm_models = {}
        self._embedding_models = {}
        self._tokenizers = {}
        
        # 設備配置
        self.device = self._setup_device()
        self.torch_dtype = self._get_torch_dtype()
        
        # TensorFlow 已移除，僅使用 PyTorch
        logger.info("使用 PyTorch 作為深度學習框架")
        
        logger.info(f"模型管理器初始化完成，使用設備: {self.device}")
    
    def _setup_device(self) -> str:
        """設置計算設備"""
        if TORCH_DEVICE == "auto":
            if torch.cuda.is_available() and HF_USE_GPU:
                device = "cuda"
                logger.info(f"使用 GPU: {torch.cuda.get_device_name()}")
            else:
                device = "cpu"
                logger.info("使用 CPU")
        else:
            device = TORCH_DEVICE
        
        return device
    
    def _get_torch_dtype(self) -> torch.dtype:
        """獲取 PyTorch 數據類型"""
        if TORCH_DTYPE == "float16":
            return torch.float16
        else:
            return torch.float32
    
    def _setup_tensorflow(self):
        """TensorFlow 已移除"""
        logger.info("TensorFlow 已移除，使用 PyTorch 進行深度學習任務")
    
    def _detect_model_type(self, model_name: str) -> str:
        """檢測模型類型"""
        # T5 系列模型
        if any(t5_indicator in model_name.lower() for t5_indicator in ['t5', 'flan-t5']):
            return "seq2seq"
        
        # BART 系列模型
        if any(bart_indicator in model_name.lower() for bart_indicator in ['bart', 'mbart']):
            return "seq2seq"
        
        # 其他 seq2seq 模型
        if any(seq2seq_indicator in model_name.lower() for seq2seq_indicator in ['pegasus', 'blenderbot']):
            return "seq2seq"
        
        # 默認為 causal LM
        return "causal"
    
    def get_llm_model(self, model_name: Optional[str] = None, use_large_model_optimizations: Optional[bool] = None) -> pipeline:
        """
        獲取語言模型
        
        Args:
            model_name: 模型名稱，如果為 None 則使用默認模型
            
        Returns:
            Hugging Face pipeline 對象
        """
        if model_name is None:
            model_name = DEFAULT_LLM_MODEL
        
        if model_name not in self._llm_models:
            logger.info(f"加載語言模型: {model_name}")
            
            # 自動檢測是否為大型模型
            if use_large_model_optimizations is None:
                use_large_model_optimizations = any(indicator in model_name.lower() for indicator in 
                                                  ['20b', '13b', '7b', '6b', '3b', 'gpt-j', 'gpt-neox', 'bloom', 'chatglm', 'baichuan', 'qwen'])
            
            try:
                # 檢測模型類型
                model_type = self._detect_model_type(model_name)
                logger.info(f"檢測到模型類型: {model_type}")
                
                # 1. 顯式加載 Tokenizer 和 Model（對 SentencePiece/MT5 提供回退）
                try:
                    tokenizer = AutoTokenizer.from_pretrained(
                        model_name,
                        cache_dir=str(self.cache_dir),
                        trust_remote_code=True,
                    )
                except Exception as e_tokenizer:
                    logger.warning(
                        "Tokenizer 載入失敗，嘗試使用 use_fast=False 重新載入。可能需要安裝 'sentencepiece'。錯誤: %s",
                        str(e_tokenizer),
                    )
                    try:
                        tokenizer = AutoTokenizer.from_pretrained(
                            model_name,
                            cache_dir=str(self.cache_dir),
                            trust_remote_code=True,
                            use_fast=False,
                        )
                    except Exception as e_tokenizer_slow:
                        logger.error(
                            "Tokenizer slow 模式亦載入失敗。請確認已安裝 'sentencepiece'（pip install sentencepiece）。錯誤: %s",
                            str(e_tokenizer_slow),
                        )
                        raise
                
                model_kwargs = {
                    "cache_dir": str(self.cache_dir),
                    "trust_remote_code": True,
                }

                # 大型模型特殊配置
                if use_large_model_optimizations and self.device == "cuda":
                    logger.info(f"使用大型模型優化配置載入: {model_name}")
                    model_kwargs.update({
                        "quantization_config": BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_compute_dtype=torch.float16,
                            bnb_4bit_use_double_quant=True,
                            bnb_4bit_quant_type="nf4"
                        ),
                        "device_map": "auto",
                        "low_cpu_mem_usage": True,
                    })

                if "gpt-oss-20b" in model_name.lower():
                    logger.info(f"{model_name} 是 Mxfp4 量化模型，移除 quantization_config 參數以避免衝突")
                    model_kwargs.pop("quantization_config", None)
                
                # 根據模型類型選擇正確的 AutoModel 類
                if model_type == "seq2seq":
                    logger.info(f"使用 AutoModelForSeq2SeqLM 載入模型: {model_name}")
                    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, **model_kwargs)
                    task = "text2text-generation"
                else:
                    logger.info(f"使用 AutoModelForCausalLM 載入模型: {model_name}")
                    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
                    task = "text-generation"

                # For non-large models, manually move to device if not using device_map
                if not ('device_map' in model_kwargs):
                    model.to(self.device)

                # 2. 創建 Pipeline
                pipeline_kwargs = {
                    "task": task,
                    "model": model,
                    "tokenizer": tokenizer,
                    "device": None if 'device_map' in model_kwargs else self.device, # device is handled by device_map or .to()
                }
                
                # 只對 causal LM 模型添加 return_full_text 參數
                if model_type == "causal":
                    pipeline_kwargs["return_full_text"] = False
                
                self._llm_models[model_name] = pipeline(**pipeline_kwargs)
                
                logger.info(f"語言模型 {model_name} 加載成功")
                
            except Exception as e:
                logger.error(f"加載語言模型失敗: {str(e)}")
                # 回退到更小的模型
                fallback_model = "Qwen/Qwen2-0.5B-Instruct"
                if model_name != fallback_model:
                    logger.info(f"嘗試回退到 {fallback_model}")
                    return self.get_llm_model(fallback_model)
                else:
                    raise
        
        return self._llm_models[model_name]
    
    def get_embedding_model(self, model_name: Optional[str] = None) -> SentenceTransformer:
        """
        獲取嵌入模型
        
        Args:
            model_name: 模型名稱，如果為 None 則使用默認模型
            
        Returns:
            SentenceTransformer 對象
        """
        if model_name is None:
            model_name = DEFAULT_EMBEDDING_MODEL
        
        if model_name not in self._embedding_models:
            logger.info(f"加載嵌入模型: {model_name}")
            
            try:
                # 創建嵌入模型時添加更多配置
                self._embedding_models[model_name] = SentenceTransformer(
                    model_name,
                    cache_folder=str(self.cache_dir),
                    device=self.device,
                    trust_remote_code=True
                )
                
                logger.info(f"嵌入模型 {model_name} 加載成功")
                
            except Exception as e:
                logger.error(f"加載嵌入模型失敗: {str(e)}")
                # 回退到多語言模型
                fallback_model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
                if model_name != fallback_model:
                    logger.info(f"嘗試回退到 {fallback_model}")
                    return self.get_embedding_model(fallback_model)
                else:
                    raise
        
        return self._embedding_models[model_name]
    
    def generate_text(self, prompt: str, model_name: Optional[str] = None, 
                     max_length: int = 512, temperature: float = 0.1) -> str:
        """
        生成文本
        
        Args:
            prompt: 輸入提示
            model_name: 模型名稱
            max_length: 最大生成長度
            temperature: 溫度參數
            
        Returns:
            生成的文本
        """
        try:
            # 檢查是否使用 vLLM
            if INFERENCE_ENGINE == "vllm":
                from utils.vllm_manager import get_vllm_manager
                vllm_manager = get_vllm_manager()
                
                # 初始化 vLLM（如果尚未初始化）
                if not vllm_manager.is_initialized:
                    if not vllm_manager.initialize_model(model_name):
                        # 如果 vLLM 初始化失敗，回退到 transformers
                        logger.warning("vLLM 初始化失敗，回退到 transformers")
                        return self._generate_with_transformers(prompt, model_name, max_length, temperature)
                
                return vllm_manager.generate_text(
                    prompt, 
                    max_tokens=max_length,
                    temperature=temperature
                )
            else:
                return self._generate_with_transformers(prompt, model_name, max_length, temperature)
                
        except Exception as e:
            logger.error(f"文本生成失敗: {str(e)}")
            return "生成過程中發生錯誤，請稍後再試。"
    
    def _generate_with_transformers(self, prompt: str, model_name: Optional[str] = None,
                                   max_length: int = 512, temperature: float = 0.1) -> str:
        """使用 Transformers 生成文本（回退方案）"""
        try:
            if model_name is None:
                model_name = DEFAULT_LLM_MODEL
            
            # 檢測模型類型
            model_type = self._detect_model_type(model_name)
            logger.info(f"生成文本時檢測到模型類型: {model_type}")
                
            model = self.get_llm_model(model_name)
            
            # 根據模型類型調整生成參數
            if model_type == "seq2seq":
                # T5 等 seq2seq 模型使用不同的參數
                outputs = model(
                    prompt,
                    max_new_tokens=max_length,
                    temperature=temperature,
                    do_sample=True if temperature > 0 else False
                )
            else:
                # Causal LM 模型
                outputs = model(
                    prompt,
                    max_length=max_length,
                    temperature=temperature,
                    do_sample=True
                )
            
            if outputs and len(outputs) > 0:
                # 統一處理輸出格式 - 兩種模型類型都使用相同的格式
                if isinstance(outputs[0], dict) and 'generated_text' in outputs[0]:
                    result = outputs[0]['generated_text'].strip()
                    # 如果結果為空，返回錯誤信息
                    if not result:
                        return "模型生成了空回應，請嘗試調整問題或參數。"
                    return result
                elif isinstance(outputs[0], str):
                    result = outputs[0].strip()
                    if not result:
                        return "模型生成了空回應，請嘗試調整問題或參數。"
                    return result
                else:
                    # 如果是其他格式，嘗試直接轉換
                    result = str(outputs[0]).strip()
                    if not result:
                        return "模型生成了空回應，請嘗試調整問題或參數。"
                    return result
            else:
                return "抱歉，無法生成回應。"
                
        except Exception as e:
            logger.error(f"Transformers 文本生成失敗: {str(e)}")
            logger.error(f"Prompt: {prompt[:200]}...")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return "生成過程中發生錯誤，請稍後再試。"
    
    def encode_texts(self, texts: Union[str, list], model_name: Optional[str] = None) -> np.ndarray:
        """
        編碼文本為向量
        
        Args:
            texts: 文本或文本列表
            model_name: 模型名稱
            
        Returns:
            編碼向量
        """
        try:
            model = self.get_embedding_model(model_name)
            
            # 處理輸入文本
            if isinstance(texts, str):
                texts = [texts]
            
            # 清理文本，移除可能導致編碼錯誤的字符
            cleaned_texts = []
            for text in texts:
                if isinstance(text, str):
                    # 移除控制字符和無效字符
                    cleaned_text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\r\t')
                    # 限制文本長度避免記憶體問題
                    cleaned_text = cleaned_text[:8192]  # 限制最大長度
                    cleaned_texts.append(cleaned_text if cleaned_text.strip() else "空文本")
                else:
                    cleaned_texts.append("無效文本")
            
            embeddings = model.encode(cleaned_texts, convert_to_numpy=True, show_progress_bar=False)
            return embeddings
            
        except Exception as e:
            logger.error(f"文本編碼失敗: {str(e)}")
            logger.debug(f"問題文本數量: {len(texts) if isinstance(texts, list) else 1}")
            
            # 返回零向量作為回退
            if isinstance(texts, str):
                return np.zeros((1, 384))  # 默認維度
            else:
                return np.zeros((len(texts), 384))
    
    def get_model_info(self) -> Dict[str, Any]:
        """獲取模型信息"""
        return {
            "device": self.device,
            "torch_dtype": str(self.torch_dtype),
            "cache_dir": str(self.cache_dir),
            "loaded_llm_models": list(self._llm_models.keys()),
            "loaded_embedding_models": list(self._embedding_models.keys()),
            "default_llm_model": DEFAULT_LLM_MODEL,
            "default_embedding_model": DEFAULT_EMBEDDING_MODEL
        }
    
    def clear_cache(self, model_type: Optional[str] = None):
        """
        清理模型緩存
        
        Args:
            model_type: 模型類型 ('llm', 'embedding', None 表示全部)
        """
        if model_type is None or model_type == "llm":
            self._llm_models.clear()
            logger.info("LLM 模型緩存已清理")
        
        if model_type is None or model_type == "embedding":
            self._embedding_models.clear()
            logger.info("嵌入模型緩存已清理")
        
        if model_type is None or model_type == "tokenizer":
            self._tokenizers.clear()
            logger.info("分詞器緩存已清理")
    
    def reload_model_manager(self):
        """重新載入模型管理器 - 清理所有緩存"""
        self.clear_cache()
        self._tokenizers.clear()
        logger.info("模型管理器已重新載入")
    
    def optimize_for_inference(self):
        """優化推理性能"""
        try:
            if self.device == "cuda":
                # 清理 GPU 緩存
                torch.cuda.empty_cache()
                
                # 設置 CUDA 優化
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
                
                logger.info("GPU 推理優化已啟用")
            
            # PyTorch 編譯優化（如果支持）
            if hasattr(torch, 'compile'):
                logger.info("PyTorch 2.0 編譯優化可用")
            
        except Exception as e:
            logger.warning(f"推理優化設置失敗: {str(e)}")

# 全局模型管理器實例
model_manager = ModelManager()

def get_model_manager() -> ModelManager:
    """獲取全局模型管理器實例"""
    return model_manager
