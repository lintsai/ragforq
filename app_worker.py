#!/usr/bin/env python
"""
使用兼容的環境啟動 API 服務
"""

import os
import sys
import subprocess
import importlib.metadata

def is_pydantic_v2_installed():
    """檢查是否安裝了 Pydantic v2"""
    try:
        # 使用 importlib.metadata 替代 pkg_resources
        pydantic_version = importlib.metadata.version("pydantic")
        return pydantic_version.startswith("2.")
    except:
        return False

def main():
    """主函數 - 檢查環境並啟動服務"""
    if is_pydantic_v2_installed():
        print("檢測到 Pydantic v2，這可能與某些 LangChain 組件不兼容")
        print("嘗試使用環境變量啟用 Pydantic v1 兼容模式...")
        
        # 設置環境變量以啟用 Pydantic v1 兼容
        os.environ["PYDANTIC_V1"] = "1"
        
        print("正在啟動 API 服務...")
    else:
        print("檢測到 Pydantic v1，正在啟動 API 服務...")
    
    # 導入並運行 app.py 中的 main 函數
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from app import main as app_main
    app_main()

if __name__ == "__main__":
    main() 