#!/usr/bin/env python
"""
快速啟動腳本 - 檢查系統狀態並啟動服務
"""

import os
import sys
import subprocess
import time
import requests
from pathlib import Path

# 添加項目根目錄到路徑
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

def is_in_poetry_env():
    """檢查是否在 Poetry 虛擬環境中"""
    return os.environ.get('POETRY_ACTIVE') == '1' or 'poetry' in sys.executable.lower()

def get_python_executable():
    """獲取正確的 Python 執行檔路徑"""
    if is_in_poetry_env():
        return sys.executable
    else:
        # 嘗試使用 poetry run python
        try:
            result = subprocess.run(['poetry', 'run', 'which', 'python'], 
                                  capture_output=True, text=True, cwd=project_root)
            if result.returncode == 0:
                return result.stdout.strip()
        except:
            pass
        return 'poetry run python'

def check_dependencies():
    """檢查依賴"""
    print("🔍 檢查系統依賴...")
    
    required_packages = [
        "streamlit",
        "fastapi", 
        "uvicorn",
        "transformers",
        "torch",
        "sentence_transformers"  # 注意：導入時使用下劃線
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - 缺失")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  缺少以下依賴: {', '.join(missing_packages)}")
        print("請運行: poetry install")
        return False
    
    print("✅ 所有依賴檢查通過")
    return True

def check_directories():
    """檢查必要目錄"""
    print("\n📁 檢查目錄結構...")
    
    required_dirs = [
        "logs",
        "config",
        "models/cache",
        "vector_db"
    ]
    
    for dir_path in required_dirs:
        full_path = Path(project_root) / dir_path
        if not full_path.exists():
            print(f"📁 創建目錄: {dir_path}")
            full_path.mkdir(parents=True, exist_ok=True)
        else:
            print(f"✅ {dir_path}")
    
    print("✅ 目錄結構檢查完成")
    return True

def start_api_server():
    """啟動 API 服務"""
    print("\n🚀 啟動 API 服務...")
    
    try:
        # 檢查是否已經在運行
        response = requests.get("http://localhost:8000/", timeout=2)
        if response.status_code == 200:
            print("✅ API 服務已在運行")
            return True
    except:
        pass
    
    # 啟動 API 服務
    python_cmd = get_python_executable()
    if isinstance(python_cmd, str) and python_cmd.startswith('poetry run'):
        cmd = ["poetry", "run", "python", "app.py"]
    else:
        cmd = [python_cmd, "app.py"]
    
    api_process = subprocess.Popen(
        cmd,
        cwd=project_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # 等待服務啟動
    print("⏳ 等待 API 服務啟動...")
    for i in range(30):  # 等待最多 30 秒
        try:
            response = requests.get("http://localhost:8000/", timeout=1)
            if response.status_code == 200:
                print("✅ API 服務啟動成功")
                return True
        except:
            pass
        time.sleep(1)
        print(f"⏳ 等待中... ({i+1}/30)")
    
    print("❌ API 服務啟動失敗")
    return False

def start_frontend():
    """啟動前端服務"""
    print("\n🎨 啟動前端服務...")
    
    try:
        # 檢查是否已經在運行
        response = requests.get("http://localhost:8501/", timeout=2)
        print("✅ 前端服務已在運行")
        return True
    except:
        pass
    
    # 啟動前端服務
    try:
        python_cmd = get_python_executable()
        if isinstance(python_cmd, str) and python_cmd.startswith('poetry run'):
            cmd = ["poetry", "run", "streamlit", "run", "frontend/streamlit_app.py", 
                   "--server.port=8501", "--server.headless=true"]
        else:
            cmd = [python_cmd, "-m", "streamlit", "run", "frontend/streamlit_app.py", 
                   "--server.port=8501", "--server.headless=true"]
        
        frontend_process = subprocess.Popen(
            cmd,
            cwd=project_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        print("✅ 前端服務啟動中...")
        print("🌐 前端地址: http://localhost:8501")
        
        # 等待前端服務啟動
        print("⏳ 等待前端服務啟動...")
        for i in range(15):  # 等待最多 15 秒
            try:
                response = requests.get("http://localhost:8501/", timeout=1)
                if response.status_code == 200:
                    print("✅ 前端服務啟動成功")
                    return True
            except:
                pass
            time.sleep(1)
            print(f"⏳ 等待中... ({i+1}/15)")
        
        print("⚠️ 前端服務可能需要更多時間啟動")
        return True
        
    except Exception as e:
        print(f"❌ 前端服務啟動失敗: {str(e)}")
        return False

def show_system_info():
    """顯示系統信息"""
    print("\n📊 系統信息:")
    print(f"📁 項目路徑: {project_root}")
    print(f"🐍 Python 版本: {sys.version}")
    print(f"💻 操作系統: {os.name}")
    
    # 檢查 GPU
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"🎮 GPU: {gpu_name} ({gpu_memory:.1f}GB)")
        else:
            print("🎮 GPU: 不可用 (將使用 CPU)")
    except:
        print("🎮 GPU: 檢查失敗")

def show_next_steps():
    """顯示後續步驟"""
    print("\n🎯 後續步驟:")
    print("1. 🌐 打開瀏覽器訪問: http://localhost:8501")
    print("2. 📝 完成系統初始設置流程:")
    print("   • 選擇 AI 平台 (推薦 Hugging Face)")
    print("   • 選擇語言模型 (推薦 gpt-oss-20b)")
    print("   • 選擇嵌入模型")
    print("   • 選擇 RAG 模式")
    print("3. 🚀 開始使用智能問答功能")
    print("\n💡 提示:")
    print("• 首次使用需要下載模型，請耐心等待")
    print("• 如遇問題，請查看 logs/app.log 日誌文件")
    print("• 管理員功能需要 Token，請查看 .env 文件")

def main():
    """主函數"""
    print("=" * 60)
    print("🚀 Q槽文件智能助手 - 快速啟動")
    print("=" * 60)
    
    # 1. 檢查依賴
    if not check_dependencies():
        print("\n❌ 依賴檢查失敗，請安裝缺失的依賴")
        return 1
    
    # 2. 檢查目錄
    if not check_directories():
        print("\n❌ 目錄檢查失敗")
        return 1
    
    # 3. 顯示系統信息
    show_system_info()
    
    # 4. 啟動 API 服務
    if not start_api_server():
        print("\n❌ API 服務啟動失敗")
        return 1
    
    # 5. 啟動前端服務
    if not start_frontend():
        print("\n❌ 前端服務啟動失敗")
        return 1
    
    # 6. 顯示後續步驟
    show_next_steps()
    
    print("\n" + "=" * 60)
    print("✅ 系統啟動完成！")
    print("=" * 60)
    
    return 0

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⏹️  用戶中斷啟動")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 啟動過程中發生錯誤: {str(e)}")
        sys.exit(1)