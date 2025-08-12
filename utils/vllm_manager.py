"""
vLLM 模型管理器
專門用於生產環境的高性能推理
"""

import os
import logging
from typing import List, Optional, Dict, Any
import asyncio
from concurrent.futures import ThreadPoolExecutor

from config.config import (
    VLLM_GPU_MEMORY_UTILIZATION, VLLM_MAX_MODEL_LEN, 
    VLLM_TENSOR_PARALLEL_SIZE, VLLM_DTYPE, DEFAULT_LLM_MODEL
)

logger = logging.getLogger(__name__)

class VLLMManager:
    """vLLM 模型管理器 - 生產環境高性能推理"""
    
    def __init__(self):
        self.llm = None
        self.model_name = None
        self.is_initialized = False
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    def initialize_model(self, model_name: Optional[str] = None) -> bool:
        """初始化 vLLM 模型"""
        try:
            from vllm import LLM, SamplingParams
            
            if model_name is None:
                model_name = DEFAULT_LLM_MODEL
            
            if self.model_name == model_name and self.is_initialized:
                logger.info(f"模型 {model_name} 已經初始化")
                return True
            
            logger.info(f"初始化 vLLM 模型: {model_name}")
            
            # vLLM 配置
            self.llm = LLM(
                model=model_name,
                gpu_memory_utilization=VLLM_GPU_MEMORY_UTILIZATION,
                max_model_len=VLLM_MAX_MODEL_LEN,
                tensor_parallel_size=VLLM_TENSOR_PARALLEL_SIZE,
                dtype=VLLM_DTYPE,
                trust_remote_code=True,
                enforce_eager=False,  # 使用 CUDA 圖優化
                disable_log_stats=False,
                # 針對 gpt-oss-20b 的特殊配置
                swap_space=16,  # 16GB swap space for large models
                cpu_offload_gb=0,  # 不使用 CPU offload，全 GPU
            )
            
            self.model_name = model_name
            self.is_initialized = True
            
            logger.info(f"vLLM 模型 {model_name} 初始化成功")
            return True
            
        except ImportError:
            logger.error("vLLM 未安裝，請運行: pip install vllm")
            return False
        except Exception as e:
            logger.error(f"vLLM 模型初始化失敗: {str(e)}")
            return False
    
    def generate_text(self, prompt: str, max_tokens: int = 512, 
                     temperature: float = 0.7, top_p: float = 0.9,
                     top_k: int = 50, **kwargs) -> str:
        """生成文本"""
        if not self.is_initialized:
            if not self.initialize_model():
                return "模型初始化失敗"
        
        try:
            from vllm import SamplingParams
            
            # 採樣參數
            sampling_params = SamplingParams(
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_tokens=max_tokens,
                stop=kwargs.get('stop', None),
                repetition_penalty=kwargs.get('repetition_penalty', 1.1),
                length_penalty=kwargs.get('length_penalty', 1.0),
            )
            
            # 生成
            outputs = self.llm.generate([prompt], sampling_params)
            
            if outputs and len(outputs) > 0:
                generated_text = outputs[0].outputs[0].text
                return generated_text.strip()
            else:
                return "生成失敗"
                
        except Exception as e:
            logger.error(f"vLLM 文本生成失敗: {str(e)}")
            return f"生成錯誤: {str(e)}"
    
    def generate_batch(self, prompts: List[str], max_tokens: int = 512,
                      temperature: float = 0.7, **kwargs) -> List[str]:
        """批量生成文本"""
        if not self.is_initialized:
            if not self.initialize_model():
                return ["模型初始化失敗"] * len(prompts)
        
        try:
            from vllm import SamplingParams
            
            sampling_params = SamplingParams(
                temperature=temperature,
                top_p=kwargs.get('top_p', 0.9),
                top_k=kwargs.get('top_k', 50),
                max_tokens=max_tokens,
                repetition_penalty=kwargs.get('repetition_penalty', 1.1),
            )
            
            outputs = self.llm.generate(prompts, sampling_params)
            
            results = []
            for output in outputs:
                if output.outputs:
                    results.append(output.outputs[0].text.strip())
                else:
                    results.append("生成失敗")
            
            return results
            
        except Exception as e:
            logger.error(f"vLLM 批量生成失敗: {str(e)}")
            return [f"生成錯誤: {str(e)}"] * len(prompts)
    
    async def generate_async(self, prompt: str, **kwargs) -> str:
        """異步生成文本"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor, 
            self.generate_text, 
            prompt, 
            kwargs.get('max_tokens', 512),
            kwargs.get('temperature', 0.7)
        )
    
    def get_model_info(self) -> Dict[str, Any]:
        """獲取模型信息"""
        return {
            "model_name": self.model_name,
            "is_initialized": self.is_initialized,
            "inference_engine": "vLLM",
            "gpu_memory_utilization": VLLM_GPU_MEMORY_UTILIZATION,
            "max_model_len": VLLM_MAX_MODEL_LEN,
            "tensor_parallel_size": VLLM_TENSOR_PARALLEL_SIZE,
            "dtype": VLLM_DTYPE
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """獲取推理統計信息"""
        if not self.is_initialized or not self.llm:
            return {}
        
        try:
            # vLLM 統計信息
            stats = {}
            if hasattr(self.llm, 'llm_engine'):
                engine = self.llm.llm_engine
                if hasattr(engine, 'stat_logger'):
                    stats = engine.stat_logger.stats
            
            return stats
        except Exception as e:
            logger.warning(f"獲取統計信息失敗: {str(e)}")
            return {}
    
    def cleanup(self):
        """清理資源"""
        try:
            if self.llm:
                # vLLM 沒有顯式的清理方法，依賴垃圾回收
                self.llm = None
            
            self.is_initialized = False
            self.model_name = None
            
            # 清理線程池
            self.executor.shutdown(wait=True)
            
            logger.info("vLLM 資源清理完成")
            
        except Exception as e:
            logger.error(f"vLLM 資源清理失敗: {str(e)}")

# 全局 vLLM 管理器實例
vllm_manager = VLLMManager()

def get_vllm_manager() -> VLLMManager:
    """獲取全局 vLLM 管理器實例"""
    return vllm_manager

def initialize_vllm(model_name: Optional[str] = None) -> bool:
    """初始化 vLLM"""
    return vllm_manager.initialize_model(model_name)

def generate_with_vllm(prompt: str, **kwargs) -> str:
    """使用 vLLM 生成文本"""
    return vllm_manager.generate_text(prompt, **kwargs)

def cleanup_vllm():
    """清理 vLLM 資源"""
    vllm_manager.cleanup()