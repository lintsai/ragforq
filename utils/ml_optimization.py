"""
機器學習框架優化工具
提供 PyTorch 和 TensorFlow 的最佳實踐配置和優化
"""

import os
import logging
import warnings
from typing import Optional, Dict, Any, Union
import numpy as np

# 抑制不必要的警告
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

logger = logging.getLogger(__name__)

class PyTorchOptimizer:
    """PyTorch 優化器"""
    
    def __init__(self):
        self.device = None
        self.is_initialized = False
    
    def initialize(self) -> Dict[str, Any]:
        """初始化 PyTorch 優化設置"""
        try:
            import torch
            import torch.backends.cudnn as cudnn
            
            # 設備檢測
            if torch.cuda.is_available():
                self.device = "cuda"
                device_name = torch.cuda.get_device_name()
                memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
                logger.info(f"使用 GPU: {device_name} ({memory_gb:.1f}GB)")
                
                # CUDA 優化
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
                torch.backends.cudnn.allow_tf32 = True
                torch.backends.cuda.matmul.allow_tf32 = True
                
                # 設置內存分配策略
                os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
                
            else:
                self.device = "cpu"
                logger.info("使用 CPU")
                
                # CPU 優化
                torch.set_num_threads(min(8, os.cpu_count()))
                torch.set_num_interop_threads(2)
            
            # 全局設置
            torch.set_float32_matmul_precision('medium')  # 平衡精度和性能
            
            self.is_initialized = True
            
            return {
                "device": self.device,
                "cuda_available": torch.cuda.is_available(),
                "torch_version": torch.__version__,
                "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
                "num_threads": torch.get_num_threads()
            }
            
        except ImportError:
            logger.warning("PyTorch 未安裝")
            return {"error": "PyTorch not installed"}
        except Exception as e:
            logger.error(f"PyTorch 初始化失敗: {str(e)}")
            return {"error": str(e)}
    
    def optimize_model(self, model, use_compile: bool = True, use_quantization: bool = False):
        """優化 PyTorch 模型"""
        try:
            import torch
            
            if not self.is_initialized:
                self.initialize()
            
            # 移動到設備
            if hasattr(model, 'to'):
                model = model.to(self.device)
            
            # 設置評估模式
            if hasattr(model, 'eval'):
                model.eval()
            
            # PyTorch 2.0 編譯優化
            if use_compile and hasattr(torch, 'compile'):
                try:
                    model = torch.compile(model, mode="reduce-overhead")
                    logger.info("PyTorch 2.0 編譯優化已啟用")
                except Exception as e:
                    logger.warning(f"編譯優化失敗: {str(e)}")
            
            # 量化（僅在 CPU 上）
            if use_quantization and self.device == "cpu":
                try:
                    model = torch.quantization.quantize_dynamic(
                        model, {torch.nn.Linear}, dtype=torch.qint8
                    )
                    logger.info("動態量化已啟用")
                except Exception as e:
                    logger.warning(f"量化失敗: {str(e)}")
            
            return model
            
        except Exception as e:
            logger.error(f"模型優化失敗: {str(e)}")
            return model
    
    def clear_cache(self):
        """清理 GPU 緩存"""
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                logger.info("GPU 緩存已清理")
        except Exception as e:
            logger.warning(f"清理 GPU 緩存失敗: {str(e)}")
    
    def get_memory_info(self) -> Dict[str, Any]:
        """獲取內存使用信息"""
        try:
            import torch
            
            info = {}
            
            if torch.cuda.is_available():
                info["gpu_memory_allocated"] = torch.cuda.memory_allocated() / 1e9
                info["gpu_memory_reserved"] = torch.cuda.memory_reserved() / 1e9
                info["gpu_memory_total"] = torch.cuda.get_device_properties(0).total_memory / 1e9
            
            # CPU 內存
            import psutil
            info["cpu_memory_percent"] = psutil.virtual_memory().percent
            info["cpu_memory_available"] = psutil.virtual_memory().available / 1e9
            
            return info
            
        except Exception as e:
            logger.error(f"獲取內存信息失敗: {str(e)}")
            return {}

class TensorFlowOptimizer:
    """TensorFlow 優化器"""
    
    def __init__(self):
        self.is_initialized = False
    
    def initialize(self) -> Dict[str, Any]:
        """TensorFlow 已移除"""
        try:
            # TensorFlow 已移除，返回空配置
            return {"error": "TensorFlow removed from dependencies"}
            
            # GPU 配置
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                try:
                    # 設置內存增長
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    
                    # 設置虛擬 GPU（可選）
                    # tf.config.experimental.set_virtual_device_configuration(
                    #     gpus[0],
                    #     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]
                    # )
                    
                    logger.info(f"TensorFlow GPU 配置完成，可用 GPU: {len(gpus)}")
                    
                except RuntimeError as e:
                    logger.warning(f"GPU 配置失敗: {str(e)}")
            
            # 混合精度
            try:
                policy = tf.keras.mixed_precision.Policy('mixed_float16')
                tf.keras.mixed_precision.set_global_policy(policy)
                logger.info("TensorFlow 混合精度已啟用")
            except Exception as e:
                logger.warning(f"混合精度設置失敗: {str(e)}")
            
            # XLA 編譯優化
            tf.config.optimizer.set_jit(True)
            
            # 線程配置
            tf.config.threading.set_inter_op_parallelism_threads(0)  # 使用所有可用核心
            tf.config.threading.set_intra_op_parallelism_threads(0)
            
            self.is_initialized = True
            
            return {
                "tensorflow_version": tf.__version__,
                "gpu_available": len(gpus) > 0,
                "gpu_count": len(gpus),
                "mixed_precision": True,
                "xla_enabled": True
            }
            
        except ImportError:
            logger.warning("TensorFlow 未安裝")
            return {"error": "TensorFlow not installed"}
        except Exception as e:
            logger.error(f"TensorFlow 初始化失敗: {str(e)}")
            return {"error": str(e)}
    
    def optimize_model(self, model, use_xla: bool = True):
        """TensorFlow 已移除"""
        try:
            # TensorFlow 已移除，直接返回原模型
            return model
            
            if not self.is_initialized:
                self.initialize()
            
            # XLA 編譯
            if use_xla and hasattr(model, 'compile'):
                try:
                    model.compile(
                        optimizer=model.optimizer,
                        loss=model.loss,
                        metrics=model.metrics,
                        jit_compile=True
                    )
                    logger.info("XLA 編譯優化已啟用")
                except Exception as e:
                    logger.warning(f"XLA 編譯失敗: {str(e)}")
            
            return model
            
        except Exception as e:
            logger.error(f"TensorFlow 模型優化失敗: {str(e)}")
            return model
    
    def clear_session(self):
        """TensorFlow 已移除"""
        logger.info("TensorFlow 已移除，無需清理會話")

class MLOptimizer:
    """統一的機器學習優化器"""
    
    def __init__(self):
        self.pytorch_optimizer = PyTorchOptimizer()
        self.tensorflow_optimizer = TensorFlowOptimizer()
        self.optimization_info = {}
    
    def initialize_all(self) -> Dict[str, Any]:
        """初始化 PyTorch 框架"""
        info = {}
        
        # 初始化 PyTorch
        pytorch_info = self.pytorch_optimizer.initialize()
        info["pytorch"] = pytorch_info
        
        # TensorFlow 已移除，不再初始化
        
        self.optimization_info = info
        logger.info("機器學習框架優化初始化完成")
        
        return info
    
    def optimize_for_inference(self):
        """針對推理優化"""
        # PyTorch 優化
        self.pytorch_optimizer.clear_cache()
        
        # TensorFlow 已移除
        
        logger.info("推理優化完成")
    
    def get_system_info(self) -> Dict[str, Any]:
        """獲取系統信息"""
        info = {
            "optimization_info": self.optimization_info,
            "memory_info": self.pytorch_optimizer.get_memory_info()
        }
        
        return info
    
    def cleanup(self):
        """清理資源"""
        self.pytorch_optimizer.clear_cache()
        # TensorFlow 已移除
        logger.info("機器學習資源清理完成")

# 全局優化器實例
ml_optimizer = MLOptimizer()

def initialize_ml_frameworks():
    """初始化機器學習框架"""
    return ml_optimizer.initialize_all()

def optimize_for_inference():
    """優化推理性能"""
    ml_optimizer.optimize_for_inference()

def cleanup_ml_resources():
    """清理機器學習資源"""
    ml_optimizer.cleanup()

def get_ml_system_info():
    """獲取機器學習系統信息"""
    return ml_optimizer.get_system_info()