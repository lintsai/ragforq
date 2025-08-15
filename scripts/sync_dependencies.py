#!/usr/bin/env python
"""
同步 Poetry 和 requirements.txt 依賴的腳本
"""

import subprocess
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def sync_dependencies():
    """同步 Poetry 和 requirements.txt"""
    
    if not Path("pyproject.toml").exists():
        logger.error("❌ 未找到 pyproject.toml 文件")
        return False
    
    try:
        # 1. 安裝 Poetry 依賴
        logger.info("安裝 Poetry 依賴...")
        subprocess.run(["poetry", "install"], check=True)
        logger.info("✅ Poetry 依賴安裝完成")
        
        # 2. 導出到 requirements.txt
        logger.info("導出依賴到 requirements.txt...")
        result = subprocess.run(
            ["poetry", "export", "-f", "requirements.txt", "--output", "requirements.txt", "--without-hashes"],
            check=True, capture_output=True, text=True
        )
        logger.info("✅ requirements.txt 已更新")
        
        # 3. 更新 poetry.lock
        logger.info("更新 poetry.lock...")
        subprocess.run(["poetry", "lock"], check=True)
        logger.info("✅ poetry.lock 已更新")
        
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ 同步失敗: {e}")
        return False
    except FileNotFoundError:
        logger.error("❌ Poetry 未安裝，請先安裝 Poetry")
        logger.info("安裝命令: curl -sSL https://install.python-poetry.org | python3 -")
        return False

def main():
    """主函數"""
    logger.info("開始同步依賴...")
    
    if sync_dependencies():
        logger.info("🎉 依賴同步完成！")
        print("\n✅ 同步完成！")
        print("- pyproject.toml: Poetry 依賴配置")
        print("- poetry.lock: 鎖定的依賴版本")
        print("- requirements.txt: pip 兼容的依賴列表")
    else:
        logger.error("❌ 依賴同步失敗")
        sys.exit(1)

if __name__ == "__main__":
    main()