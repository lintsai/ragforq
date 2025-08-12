#!/usr/bin/env python
"""
Hugging Face 環境檢查腳本
"""

import os
import sys
import subprocess
from pathlib import Path

def check_python_packages():
    """檢查 Python 包"""
    print("📦 檢查 Python 包...")
    
    required_packages = {
        "transformers": "4.35.0",
        "torch": "2.0.0", 
        "accelerate": "0.20.0",
        "datasets": "2.14.0",
        "sentence-transformers": "2.2.2",
        "langchain-huggingface": "0.1.0"
    }
    
    optional_packages = {
        "vllm": "0.2.0",
        "ray": "2.8.0",
        "tensorflow": "2.13.0"
    }
    
    print("  必要包:")
    for package, min_version in required_packages.items():
        try:
            __import__(package)
            import importlib.metadata
            version = importlib.metadata.version(package)
            print(f"    ✅ {package} ({version})")
        except ImportError:
            print(f"    ❌ {package} - 未安裝")
        except Exception as e:
            print(f"    ⚠️ {package} - 檢查失敗: {str(e)}")
    
    print("  可選包:")
    for package, min_version in optional_packages.items():
        try:
            __import__(package)
            import importlib.metadata
            version = importlib.metadata.version(package)
            print(f"    ✅ {package} ({version})")
        except ImportError:
            print(f"    ⚠️ {package} - 未安裝（可選）")
        except Exception as e:
            print(f"    ⚠️ {package} - 檢查失敗: {str(e)}")

def check_gpu_support():
    """檢查 GPU 支援"""
    print("\n🎮 檢查 GPU 支援...")
    
    # 檢查 NVIDIA GPU
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("  ✅ NVIDIA GPU 可用")
            # 解析 GPU 信息
            lines = result.stdout.split('\n')
            for line in lines:
                if 'GeForce' in line or 'RTX' in line or 'GTX' in line or 'Tesla' in line or 'A100' in line:
                    gpu_info = line.strip()
                    print(f"    🎯 GPU: {gpu_info}")
                    break
        else:
            print("  ❌ NVIDIA GPU 不可用或驅動未安裝")
    except FileNotFoundError:
        print("  ❌ nvidia-smi 命令不存在")
    
    # 檢查 PyTorch GPU 支援
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✅ PyTorch CUDA 支援: {torch.version.cuda}")
            print(f"  🔢 GPU 數量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
                print(f"    GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
        else:
            print("  ⚠️ PyTorch CUDA 不可用，將使用 CPU")
    except ImportError:
        print("  ❌ PyTorch 未安裝")
    except Exception as e:
        print(f"  ⚠️ PyTorch GPU 檢查失敗: {str(e)}")

def check_model_cache():
    """檢查模型緩存目錄"""
    print("\n📁 檢查模型緩存...")
    
    # 從環境變數或默認路徑獲取緩存目錄
    cache_dir = os.getenv("HF_MODEL_CACHE_DIR", "./models/cache")
    cache_path = Path(cache_dir)
    
    print(f"  📂 緩存目錄: {cache_path.absolute()}")
    
    if cache_path.exists():
        print("  ✅ 緩存目錄存在")
        
        # 檢查目錄大小
        total_size = 0
        file_count = 0
        for file_path in cache_path.rglob('*'):
            if file_path.is_file():
                total_size += file_path.stat().st_size
                file_count += 1
        
        if file_count > 0:
            print(f"  📊 緩存大小: {total_size / 1e9:.2f}GB ({file_count} 個文件)")
        else:
            print("  📊 緩存目錄為空")
    else:
        print("  ⚠️ 緩存目錄不存在，將自動創建")
        try:
            cache_path.mkdir(parents=True, exist_ok=True)
            print("  ✅ 緩存目錄創建成功")
        except Exception as e:
            print(f"  ❌ 緩存目錄創建失敗: {str(e)}")
    
    # 檢查磁盤空間
    try:
        import shutil
        total, used, free = shutil.disk_usage(cache_path.parent)
        free_gb = free / 1e9
        print(f"  💾 可用磁盤空間: {free_gb:.1f}GB")
        
        if free_gb < 10:
            print("  ⚠️ 磁盤空間不足，建議至少保留 50GB 用於模型緩存")
        elif free_gb < 50:
            print("  ⚠️ 磁盤空間較少，建議保留更多空間用於大型模型")
        else:
            print("  ✅ 磁盤空間充足")
    except Exception as e:
        print(f"  ⚠️ 磁盤空間檢查失敗: {str(e)}")

def check_network_connectivity():
    """檢查網路連接"""
    print("\n🌐 檢查網路連接...")
    
    import requests
    
    # 檢查 Hugging Face Hub 連接
    try:
        response = requests.get("https://huggingface.co", timeout=10)
        if response.status_code == 200:
            print("  ✅ Hugging Face Hub 可訪問")
        else:
            print(f"  ⚠️ Hugging Face Hub 響應異常: {response.status_code}")
    except requests.exceptions.Timeout:
        print("  ⚠️ Hugging Face Hub 連接超時")
    except requests.exceptions.ConnectionError:
        print("  ❌ Hugging Face Hub 連接失敗")
    except Exception as e:
        print(f"  ⚠️ Hugging Face Hub 連接檢查失敗: {str(e)}")
    
    # 檢查 Hugging Face Token
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        print("  ✅ Hugging Face Token 已設置")
        # 可以添加 Token 驗證邏輯
    else:
        print("  ⚠️ Hugging Face Token 未設置（可選，但建議設置以提高下載速度）")

def test_model_loading():
    """測試模型載入"""
    print("\n🧪 測試模型載入...")
    
    try:
        from transformers import AutoTokenizer, AutoModel
        
        # 測試載入一個小型模型
        model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        print(f"  🔄 測試載入模型: {model_name}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        
        print("  ✅ 模型載入成功")
        
        # 測試推理
        inputs = tokenizer("Hello, world!", return_tensors="pt")
        outputs = model(**inputs)
        
        print("  ✅ 模型推理成功")
        
    except Exception as e:
        print(f"  ❌ 模型載入測試失敗: {str(e)}")
        print("  💡 建議檢查網路連接和磁盤空間")

def generate_recommendations():
    """生成建議"""
    print("\n💡 建議和下一步:")
    print("=" * 50)
    
    print("🚀 如果所有檢查都通過:")
    print("  1. 運行 python scripts/quick_start.py")
    print("  2. 在前端選擇 Hugging Face 平台")
    print("  3. 選擇適合的模型組合")
    
    print("\n⚠️ 如果遇到問題:")
    print("  1. GPU 不可用: 系統會自動使用 CPU，但速度較慢")
    print("  2. 網路問題: 檢查防火牆和代理設置")
    print("  3. 磁盤空間不足: 清理不必要的文件或擴展儲存")
    print("  4. 包缺失: 運行 pip install -r requirements.txt")
    
    print("\n📚 參考文檔:")
    print("  • docs/huggingface_setup.md - 詳細設置指南")
    print("  • README.md - 系統概覽")
    
    print("\n🔧 進階配置:")
    print("  • 設置 HF_TOKEN 以提高下載速度")
    print("  • 調整 INFERENCE_ENGINE 選擇推理引擎")
    print("  • 配置 GPU 記憶體使用率")

def main():
    """主函數"""
    print("🔍 Hugging Face 環境檢查工具")
    print("=" * 60)
    
    check_python_packages()
    check_gpu_support()
    check_model_cache()
    check_network_connectivity()
    test_model_loading()
    generate_recommendations()
    
    print("\n" + "=" * 60)
    print("✅ 環境檢查完成！")

if __name__ == "__main__":
    main()
