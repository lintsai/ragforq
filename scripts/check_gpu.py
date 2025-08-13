#!/usr/bin/env python3
"""
檢查 GPU 可用性
"""

import sys
import os

def check_pytorch_gpu():
    """檢查 PyTorch GPU 支援"""
    print("🔍 檢查 PyTorch GPU 支援...")
    
    try:
        import torch
        print(f"✅ PyTorch 版本: {torch.__version__}")
        
        # 檢查 CUDA 可用性
        cuda_available = torch.cuda.is_available()
        print(f"CUDA 可用: {'✅ 是' if cuda_available else '❌ 否'}")
        
        if cuda_available:
            # GPU 詳細信息
            gpu_count = torch.cuda.device_count()
            print(f"GPU 數量: {gpu_count}")
            
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_props = torch.cuda.get_device_properties(i)
                memory_gb = gpu_props.total_memory / 1e9
                print(f"  GPU {i}: {gpu_name} ({memory_gb:.1f}GB)")
            
            # 當前設備
            current_device = torch.cuda.current_device()
            print(f"當前設備: GPU {current_device}")
            
            # CUDNN 版本
            if torch.backends.cudnn.is_available():
                cudnn_version = torch.backends.cudnn.version()
                print(f"CUDNN 版本: {cudnn_version}")
            else:
                print("CUDNN: ❌ 不可用")
        else:
            print("原因可能:")
            print("  - 沒有安裝 NVIDIA GPU")
            print("  - 沒有安裝 CUDA")
            print("  - PyTorch 是 CPU 版本")
            print("  - GPU 驅動問題")
        
        return cuda_available
        
    except ImportError:
        print("❌ PyTorch 未安裝")
        return False
    except Exception as e:
        print(f"❌ 檢查失敗: {e}")
        return False

def check_system_gpu():
    """檢查系統 GPU"""
    print("\n🔍 檢查系統 GPU...")
    
    try:
        # Windows
        if os.name == 'nt':
            import subprocess
            result = subprocess.run(['wmic', 'path', 'win32_VideoController', 'get', 'name'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                gpus = [line.strip() for line in lines[1:] if line.strip()]
                if gpus:
                    print("系統檢測到的 GPU:")
                    for gpu in gpus:
                        print(f"  - {gpu}")
                        if "NVIDIA" in gpu.upper():
                            print("    ✅ NVIDIA GPU 檢測到")
                        elif "AMD" in gpu.upper() or "RADEON" in gpu.upper():
                            print("    ⚠️ AMD GPU (PyTorch 主要支援 NVIDIA)")
                        elif "INTEL" in gpu.upper():
                            print("    ⚠️ Intel 集成顯卡 (不適合深度學習)")
                else:
                    print("❌ 沒有檢測到 GPU")
            else:
                print("❌ 無法檢測系統 GPU")
        else:
            # Linux/Mac
            try:
                result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
                if result.returncode == 0:
                    print("✅ NVIDIA GPU 檢測到:")
                    print(result.stdout)
                else:
                    print("❌ nvidia-smi 命令失敗，可能沒有 NVIDIA GPU 或驅動")
            except FileNotFoundError:
                print("❌ nvidia-smi 命令不存在")
                
    except Exception as e:
        print(f"❌ 系統 GPU 檢查失敗: {e}")

def check_environment_config():
    """檢查環境配置"""
    print("\n🔍 檢查環境配置...")
    
    # 添加項目路徑
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    try:
        from config.config import TORCH_DEVICE, HF_USE_GPU
        
        print(f"TORCH_DEVICE: {TORCH_DEVICE}")
        print(f"HF_USE_GPU: {HF_USE_GPU}")
        
        if TORCH_DEVICE == "auto":
            print("✅ 設備設置為自動檢測")
        elif TORCH_DEVICE == "cpu":
            print("⚠️ 強制使用 CPU")
        elif TORCH_DEVICE == "cuda":
            print("⚠️ 強制使用 CUDA")
        
        if not HF_USE_GPU:
            print("⚠️ HF_USE_GPU 設置為 False，將不使用 GPU")
            
    except Exception as e:
        print(f"❌ 配置檢查失敗: {e}")

def check_cuda_installation():
    """檢查 CUDA 安裝"""
    print("\n🔍 檢查 CUDA 安裝...")
    
    try:
        # 檢查 CUDA 環境變數
        cuda_path = os.environ.get('CUDA_PATH')
        if cuda_path:
            print(f"CUDA_PATH: {cuda_path}")
        else:
            print("❌ CUDA_PATH 環境變數未設置")
        
        # 檢查 nvcc
        import subprocess
        try:
            result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ NVCC 可用:")
                print(result.stdout.strip())
            else:
                print("❌ NVCC 不可用")
        except FileNotFoundError:
            print("❌ NVCC 命令不存在")
            
    except Exception as e:
        print(f"❌ CUDA 檢查失敗: {e}")

def main():
    """主函數"""
    print("🚀 GPU 可用性檢查")
    print("=" * 50)
    
    # 檢查系統 GPU
    check_system_gpu()
    
    # 檢查 CUDA 安裝
    check_cuda_installation()
    
    # 檢查 PyTorch GPU 支援
    pytorch_gpu = check_pytorch_gpu()
    
    # 檢查環境配置
    check_environment_config()
    
    # 總結
    print("\n" + "=" * 50)
    print("📊 總結:")
    
    if pytorch_gpu:
        print("✅ GPU 可用，系統將自動使用 GPU 加速")
        print("💡 建議:")
        print("  - 可以使用 vLLM 推理引擎獲得更好性能")
        print("  - 可以載入更大的模型")
    else:
        print("❌ GPU 不可用，系統將使用 CPU")
        print("💡 建議:")
        print("  - 使用較小的模型（如 Qwen2 0.5B）")
        print("  - 使用 Transformers 推理引擎")
        print("  - 如果需要 GPU，請安裝 CUDA 和 GPU 版本的 PyTorch")

if __name__ == "__main__":
    main()