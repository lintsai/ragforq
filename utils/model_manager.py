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
    pipeline
)

# 嘗試導入 BitsAndBytesConfig，如果失敗則設為 None
try:
    from transformers import BitsAndBytesConfig
except ImportError:
    BitsAndBytesConfig = None
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
            
            # 自動檢測是否為大型模型 - 基於參數量而非模型名稱
            if use_large_model_optimizations is None:
                model_name_lower = model_name.lower()
                # 檢測大型模型的通用邏輯
                large_model_indicators = ['20b', '13b', '7b', '6b', '3b']
                small_model_indicators = ['0.5b', '1b', '1.5b', '2b']
                
                # 如果明確標示為小型模型，則不使用大型模型優化
                if any(indicator in model_name_lower for indicator in small_model_indicators):
                    use_large_model_optimizations = False
                # 如果明確標示為大型模型，則使用優化
                elif any(indicator in model_name_lower for indicator in large_model_indicators):
                    use_large_model_optimizations = True
                # 其他已知的大型模型架構
                elif any(arch in model_name_lower for arch in ['gpt-j', 'gpt-neox', 'bloom', 'chatglm', 'baichuan']):
                    use_large_model_optimizations = True
                else:
                    # 默認不使用大型模型優化
                    use_large_model_optimizations = False
            
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
                    
                    # 檢查 GPU 記憶體
                    if torch.cuda.is_available():
                        gpu_count = torch.cuda.device_count()
                        total_memory = 0
                        
                        for i in range(gpu_count):
                            gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)  # GB
                            total_memory += gpu_memory
                            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}, 記憶體: {gpu_memory:.1f} GB")
                        
                        logger.info(f"總 GPU 記憶體: {total_memory:.1f} GB")
                        
                        # 根據可用記憶體動態配置
                        if total_memory >= 16:  # 有足夠記憶體載入大型模型
                            logger.info("GPU 記憶體充足，使用自動設備映射")
                            
                            # 選擇合適的數據類型
                            if torch.cuda.is_bf16_supported():
                                dtype = torch.bfloat16
                                logger.info("使用 bfloat16 數據類型")
                            else:
                                dtype = torch.float16
                                logger.info("使用 float16 數據類型")
                            
                            model_kwargs.update({
                                "device_map": "auto",
                                "low_cpu_mem_usage": True,
                                "torch_dtype": dtype,
                            })
                            
                            # 根據記憶體大小決定是否使用量化
                            if total_memory >= 32:
                                logger.info("記憶體充足，不使用量化")
                            elif BitsAndBytesConfig is not None:
                                logger.info("使用 8bit 量化節省記憶體")
                                model_kwargs.update({
                                    "quantization_config": BitsAndBytesConfig(load_in_8bit=True),
                                })
                        else:
                            # 記憶體不足，拋出異常讓系統回退
                            logger.warning(f"GPU 記憶體不足 ({total_memory:.1f} GB < 16 GB)")
                            raise RuntimeError(f"GPU 記憶體不足，需要至少 16GB，當前僅有 {total_memory:.1f}GB")
                    else:
                        raise RuntimeError("未檢測到 CUDA GPU")
                
                # 根據模型類型選擇正確的 AutoModel 類
                if model_type == "seq2seq":
                    logger.info(f"使用 AutoModelForSeq2SeqLM 載入模型: {model_name}")
                    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, **model_kwargs)
                    task = "text2text-generation"
                else:
                    logger.info(f"使用 AutoModelForCausalLM 載入模型: {model_name}")
                    try:
                        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
                        task = "text-generation"
                    except KeyError as e:
                        if "gpt-oss-20b" in model_name.lower() and "gate_up_proj" in str(e):
                            logger.warning(f"MoE 模型加載失敗，嘗試不使用 offload 配置: {e}")
                            # 移除 offload 相關配置重試
                            retry_kwargs = {k: v for k, v in model_kwargs.items() 
                                          if k not in ["offload_folder", "offload_state_dict"]}
                            model = AutoModelForCausalLM.from_pretrained(model_name, **retry_kwargs)
                            task = "text-generation"
                        else:
                            raise

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
                
                # 通用tokenizer配置
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                
                # 設置通用的生成參數
                pipeline_kwargs["model_kwargs"] = {
                    "pad_token_id": tokenizer.eos_token_id,
                    "eos_token_id": tokenizer.eos_token_id,
                }
                
                self._llm_models[model_name] = pipeline(**pipeline_kwargs)
                
                logger.info(f"語言模型 {model_name} 加載成功")
                
            except Exception as e:
                error_msg = str(e)
                logger.error(f"加載語言模型失敗: {error_msg}")
                
                # 特殊處理 CUDA 記憶體分配器錯誤
                if "expandable_segment" in error_msg or "CUDACachingAllocator" in error_msg:
                    logger.error("檢測到 CUDA 記憶體分配器內部錯誤，執行深度清理...")
                    self._deep_gpu_cleanup()
                
                # 清理 GPU 記憶體
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                    logger.info("已清理 GPU 記憶體")
                
                # 智能回退策略 - 動態選擇合適的回退模型
                fallback_model = self._get_fallback_model(model_name, error_msg)
                
                if fallback_model and model_name != fallback_model:
                    logger.info(f"嘗試回退到較小模型: {fallback_model}")
                    # 確保回退模型不使用大型模型優化
                    return self.get_llm_model(fallback_model, use_large_model_optimizations=False)
                else:
                    logger.error("無法找到合適的回退模型或回退模型也載入失敗")
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
    
    def generate_text(self, prompt: str, model_name: Optional[str] = None, **kwargs) -> str:
        """
        生成文本
        
        Args:
            prompt: 輸入提示
            model_name: 模型名稱
            **kwargs: 生成參數 (temperature, max_new_tokens, top_p, top_k, repetition_penalty)
            
        Returns:
            生成的文本
        """
        try:
            # 動態檢查是否使用 vLLM
            from config.config import get_inference_engine
            current_inference_engine = get_inference_engine()
            
            # 為 vLLM 和 transformers 準備參數
            # vLLM uses max_tokens, transformers uses max_new_tokens
            generation_params = kwargs.copy()
            if "max_new_tokens" in generation_params:
                generation_params["max_tokens"] = generation_params.pop("max_new_tokens")

            if current_inference_engine == "vllm":
                from utils.vllm_manager import get_vllm_manager
                vllm_manager = get_vllm_manager()
                
                # 初始化 vLLM（如果尚未初始化）
                if not vllm_manager.is_initialized:
                    if not vllm_manager.initialize_model(model_name):
                        # 如果 vLLM 初始化失敗，回退到 transformers
                        logger.warning("vLLM 初始化失敗，回退到 transformers")
                        return self._generate_with_transformers(prompt, model_name, **kwargs)
                
                return vllm_manager.generate_text(
                    prompt, 
                    **generation_params
                )
            else:
                return self._generate_with_transformers(prompt, model_name, **kwargs)
                
        except Exception as e:
            logger.error(f"文本生成失敗: {str(e)}")
            return "生成過程中發生錯誤，請稍後再試。"
    
    def _generate_with_transformers(self, prompt: str, model_name: Optional[str] = None,
                                   **kwargs) -> str:
        """使用 Transformers 生成文本（回退方案）"""
        try:
            if model_name is None:
                model_name = DEFAULT_LLM_MODEL
            
            # 檢測模型類型
            model_type = self._detect_model_type(model_name)
            logger.info(f"生成文本時檢測到模型類型: {model_type}")
                
            model = self.get_llm_model(model_name)
            
            # 設置生成參數，允許從 kwargs 覆蓋預設值
            generation_kwargs = {
                "pad_token_id": model.tokenizer.eos_token_id,
                "eos_token_id": model.tokenizer.eos_token_id,
                "repetition_penalty": 1.15,     # 適度的重複懲罰
                "no_repeat_ngram_size": 2,      # 避免重複2-gram
                "early_stopping": True,         # 早停機制
                "max_new_tokens": 512, # Default value
                "temperature": 0.1,
                "top_p": 0.95,
                "top_k": 50,
                "do_sample": True,
            }
            generation_kwargs.update(kwargs)

            # 確保 do_sample 為 True 如果 temperature > 0
            if generation_kwargs.get("temperature", 0) > 0:
                generation_kwargs["do_sample"] = True

            # Causal LM 不應返回 prompt
            if model_type == "causal":
                generation_kwargs["return_full_text"] = False

            outputs = model(prompt, **generation_kwargs)
            
            if outputs and len(outputs) > 0:
                # 統一處理輸出格式
                if isinstance(outputs[0], dict) and 'generated_text' in outputs[0]:
                    result = outputs[0]['generated_text'].strip()
                elif isinstance(outputs[0], str):
                    result = outputs[0].strip()
                else:
                    result = str(outputs[0]).strip()
                
                # 檢查結果質量
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
    
    def _get_fallback_model(self, original_model: str, error_msg: str) -> Optional[str]:
        """
        智能選擇回退模型
        
        Args:
            original_model: 原始模型名稱
            error_msg: 錯誤信息
            
        Returns:
            合適的回退模型名稱，如果無法確定則返回 None
        """
        # 獲取可用的 GPU 記憶體
        total_memory = 0
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                total_memory += torch.cuda.get_device_properties(i).total_memory / (1024**3)
        
        # 根據錯誤類型和可用記憶體智能選擇
        if "expandable_segment" in error_msg or "out of memory" in error_msg.lower():
            # 嚴重記憶體錯誤，選擇最小可用模型
            if total_memory >= 8:
                return os.getenv('FALLBACK_SMALL_MODEL', 'microsoft/DialoGPT-small')
            else:
                return os.getenv('FALLBACK_TINY_MODEL', 'distilgpt2')
        
        # 根據原始模型大小智能降級
        model_lower = original_model.lower()
        
        # 大型模型 (>10B) 回退策略
        if any(size in model_lower for size in ['20b', '14b', '13b', '11b']):
            if total_memory >= 16:
                return os.getenv('FALLBACK_MEDIUM_MODEL', 'microsoft/DialoGPT-medium')
            else:
                return os.getenv('FALLBACK_SMALL_MODEL', 'microsoft/DialoGPT-small')
        
        # 中型模型 (3-10B) 回退策略
        elif any(size in model_lower for size in ['7b', '6b', '5b', '4b', '3b']):
            if total_memory >= 8:
                return os.getenv('FALLBACK_SMALL_MODEL', 'microsoft/DialoGPT-small')
            else:
                return os.getenv('FALLBACK_TINY_MODEL', 'distilgpt2')
        
        # 小型模型 (1-3B) 回退策略
        elif any(size in model_lower for size in ['2b', '1.5b', '1b']):
            return os.getenv('FALLBACK_TINY_MODEL', 'distilgpt2')
        
        # 如果無法從模型名稱判斷大小，根據記憶體選擇
        if total_memory >= 16:
            return os.getenv('FALLBACK_MEDIUM_MODEL', 'microsoft/DialoGPT-medium')
        elif total_memory >= 8:
            return os.getenv('FALLBACK_SMALL_MODEL', 'microsoft/DialoGPT-small')
        else:
            return os.getenv('FALLBACK_TINY_MODEL', 'distilgpt2')

    def _deep_gpu_cleanup(self):
        """深度 GPU 記憶體清理 - 處理記憶體碎片化問題"""
        import gc
        import time
        
        logger.info("執行深度 GPU 記憶體清理...")
        
        if torch.cuda.is_available():
            try:
                # 1. 清理所有模型緩存
                self.clear_cache()
                
                # 2. 強制垃圾回收
                gc.collect()
                
                # 3. 清理每個 GPU 設備
                for i in range(torch.cuda.device_count()):
                    with torch.cuda.device(i):
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()
                        # 重置記憶體統計
                        torch.cuda.reset_peak_memory_stats(i)
                        torch.cuda.reset_accumulated_memory_stats(i)
                
                # 4. 等待清理完成
                time.sleep(2)
                
                # 5. 再次強制清理
                torch.cuda.empty_cache()
                gc.collect()
                
                logger.info("深度 GPU 記憶體清理完成")
                
            except Exception as e:
                logger.error(f"深度 GPU 清理失敗: {e}")

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
