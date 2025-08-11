#!/usr/bin/env python3
"""
依賴檢查腳本
檢查所有必要的依賴是否正確安裝
"""

import sys
import importlib
from typing import List, Tuple

def check_dependency(module_name: str, import_name: str = None) -> Tuple[bool, str]:
    """
    檢查單個依賴
    
    Args:
        module_name: 模組名稱
        import_name: 導入名稱（如果與模組名稱不同）
    
    Returns:
        (是否成功, 錯誤信息或版本信息)
    """
    try:
        if import_name:
            module = importlib.import_module(import_name)
        else:
            module = importlib.import_module(module_name)
        
        # 嘗試獲取版本信息
        version = "未知版本"
        if hasattr(module, '__version__'):
            version = module.__version__
        elif hasattr(module, 'VERSION'):
            version = str(module.VERSION)
        elif hasattr(module, 'version'):
            version = str(module.version)
        
        return True, f"✅ {module_name}: {version}"
    except ImportError as e:
        return False, f"❌ {module_name}: {str(e)}"
    except Exception as e:
        return False, f"⚠️  {module_name}: {str(e)}"

def main():
    """主檢查函數"""
    print("🔍 依賴檢查開始")
    print("=" * 60)
    
    # 定義需要檢查的依賴
    dependencies = [
        # Web框架
        ("streamlit", None),
        ("fastapi", None),
        ("uvicorn", None),
        
        # HTTP請求
        ("requests", None),
        
        # 數據處理
        ("numpy", None),
        ("pandas", None),  # 可能被某些組件使用
        
        # 機器學習
        ("sentence-transformers", "sentence_transformers"),
        ("langchain", None),
        ("langchain-ollama", "langchain_ollama"),
        ("langchain-community", "langchain_community"),
        
        # 向量數據庫
        ("faiss-cpu", "faiss"),
        
        # 文檔處理
        ("unstructured", None),
        ("pypdf", None),
        ("pymupdf", "fitz"),
        ("python-docx", "docx"),
        ("openpyxl", None),
        ("xlrd", None),
        ("olefile", None),
        
        # 數據驗證
        ("pydantic", None),
        ("annotated-types", "annotated_types"),
        
        # 工具庫
        ("python-dotenv", "dotenv"),
        ("tqdm", None),
        ("psutil", None),
        ("pytz", None),
        
        # Streamlit擴展
        ("streamlit-autorefresh", "streamlit_autorefresh"),
    ]
    
    success_count = 0
    total_count = len(dependencies)
    failed_deps = []
    
    for module_name, import_name in dependencies:
        success, message = check_dependency(module_name, import_name)
        print(message)
        
        if success:
            success_count += 1
        else:
            failed_deps.append(module_name)
    
    print("\n" + "=" * 60)
    print("📊 檢查結果")
    print("=" * 60)
    print(f"總依賴數: {total_count}")
    print(f"成功: {success_count}")
    print(f"失敗: {len(failed_deps)}")
    print(f"成功率: {(success_count/total_count*100):.1f}%")
    
    if failed_deps:
        print(f"\n❌ 失敗的依賴:")
        for dep in failed_deps:
            print(f"  - {dep}")
        
        print(f"\n💡 安裝建議:")
        print(f"poetry install")
        print(f"或者單獨安裝失敗的依賴:")
        for dep in failed_deps:
            print(f"poetry add {dep}")
    else:
        print(f"\n🎉 所有依賴都已正確安裝！")
    
    # 檢查Python版本
    print(f"\n🐍 Python版本檢查")
    print(f"當前版本: {sys.version}")
    
    if sys.version_info >= (3, 10) and sys.version_info < (3, 11):
        print("✅ Python版本符合要求 (3.10.x)")
    else:
        print("⚠️  建議使用Python 3.10.x版本")

if __name__ == "__main__":
    main()