#!/usr/bin/env python
"""
環境配置檢查腳本
"""

import os
import sys
from pathlib import Path

def check_env_files():
    """檢查環境配置文件"""
    project_root = Path(__file__).parent.parent
    
    print("🔍 環境配置文件檢查")
    print("=" * 50)
    
    # 定義環境文件
    env_files = {
        ".env.example": "配置範本",
        ".env": "本地開發配置",
        ".env.local": "Docker 本地測試配置", 
        ".env.production": "生產環境配置"
    }
    
    # 檢查文件存在性
    print("📁 文件存在性檢查:")
    for file_name, description in env_files.items():
        file_path = project_root / file_name
        if file_path.exists():
            print(f"  ✅ {file_name} - {description}")
        else:
            print(f"  ❌ {file_name} - {description} (不存在)")
    
    # 檢查當前環境配置
    print("\n⚙️ 當前環境配置:")
    current_env = project_root / ".env"
    
    if current_env.exists():
        print("  ✅ 找到 .env 文件")
        
        # 讀取關鍵配置
        with open(current_env, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 檢查平台配置（現在通過前端設置）
        print("  🎯 平台選擇: 通過前端設置流程進行（不在環境變數中）")
        
        # 檢查 Ollama 配置
        if 'OLLAMA_HOST=' in content:
            for line in content.split('\n'):
                if line.startswith('OLLAMA_HOST='):
                    host = line.split('=')[1].strip('"')
                    print(f"  🔗 Ollama 主機: {host}")
                    break
        
        # 檢查 Q 槽配置
        if 'Q_DRIVE_PATH=' in content:
            for line in content.split('\n'):
                if line.startswith('Q_DRIVE_PATH='):
                    path = line.split('=')[1].strip('"')
                    print(f"  📁 Q槽路徑: {path}")
                    break
        
        # 檢查管理員 Token
        if 'ADMIN_TOKEN=' in content:
            for line in content.split('\n'):
                if line.startswith('ADMIN_TOKEN='):
                    token = line.split('=')[1].strip('"')
                    if token and token != "your_admin_token_here":
                        print(f"  🔐 管理員 Token: 已設置")
                    else:
                        print(f"  ⚠️ 管理員 Token: 未設置或使用預設值")
                    break
    else:
        print("  ❌ 未找到 .env 文件")
        print("  💡 建議: cp .env.example .env")
    
    # 檢查必要目錄
    print("\n📂 必要目錄檢查:")
    required_dirs = [
        ("logs", "日誌目錄"),
        ("vector_db", "向量資料庫目錄"),
        ("models/cache", "模型緩存目錄")
    ]
    
    for dir_name, description in required_dirs:
        dir_path = project_root / dir_name
        if dir_path.exists():
            print(f"  ✅ {dir_name}/ - {description}")
        else:
            print(f"  ⚠️ {dir_name}/ - {description} (將自動創建)")
    
    print("\n" + "=" * 50)

def suggest_configuration():
    """建議配置"""
    print("💡 配置建議:")
    print("=" * 50)
    
    print("🏠 本地開發環境:")
    print("  1. 複製配置: cp .env.example .env")
    print("  2. 設置平台: SELECTED_PLATFORM=\"ollama\"")
    print("  3. 設置 Q 槽路徑為測試目錄")
    print("  4. 啟動: python scripts/quick_start.py")
    
    print("\n🐳 Docker 本地測試:")
    print("  1. 使用配置: cp .env.local .env")
    print("  2. 確保 Ollama 在宿主機運行")
    print("  3. 構建: docker build -t ragforq .")
    print("  4. 運行: docker run --gpus all -p 8000:8000 -p 8501:8501 ragforq")
    
    print("\n🚀 生產環境:")
    print("  1. 使用配置: cp .env.production .env")
    print("  2. 設置正確的 Q 槽路徑")
    print("  3. 配置管理員 Token")
    print("  4. 使用 Docker Compose 或 Kubernetes 部署")
    
    print("\n🎯 平台選擇建議:")
    print("  • Ollama: 本地推理，隱私保護，適合大多數場景")
    print("  • Hugging Face: 豐富模型，雲端推理，適合實驗和研究")
    
    print("\n" + "=" * 50)

def main():
    """主函數"""
    print("🔧 Q槽文件智能助手 - 環境配置檢查工具")
    print("=" * 60)
    
    check_env_files()
    suggest_configuration()
    
    print("✅ 檢查完成！")
    print("💡 如需幫助，請參考 README.md 或運行 python scripts/quick_start.py")

if __name__ == "__main__":
    main()