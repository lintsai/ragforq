#!/usr/bin/env python3
"""
同步 Poetry 和 requirements.txt 的依賴版本
以 pyproject.toml 為主要來源
"""

import toml
import re
from pathlib import Path

def sync_dependencies():
    """同步依賴版本"""
    
    # 讀取 pyproject.toml
    pyproject_path = Path("pyproject.toml")
    if not pyproject_path.exists():
        print("❌ pyproject.toml 不存在")
        return
    
    with open(pyproject_path, 'r', encoding='utf-8') as f:
        pyproject_data = toml.load(f)
    
    dependencies = pyproject_data.get('tool', {}).get('poetry', {}).get('dependencies', {})
    
    # 生成新的 requirements.txt
    requirements_lines = []
    
    for package, version in dependencies.items():
        if package == 'python':
            continue
            
        # 處理不同的版本格式
        if isinstance(version, str):
            if version.startswith('^'):
                # ^1.2.3 -> >=1.2.3
                clean_version = version[1:]
                requirements_lines.append(f"{package}>={clean_version}")
            elif version.startswith('>='):
                requirements_lines.append(f"{package}{version}")
            else:
                requirements_lines.append(f"{package}>={version}")
        elif isinstance(version, dict):
            # 處理 extras 和 version
            if 'extras' in version and 'version' in version:
                extras = ','.join(version['extras'])
                ver = version['version']
                if ver.startswith('^'):
                    ver = '>=' + ver[1:]
                requirements_lines.append(f"{package}[{extras}]{ver}")
            elif 'version' in version:
                ver = version['version']
                if ver.startswith('^'):
                    ver = '>=' + ver[1:]
                requirements_lines.append(f"{package}{ver}")
    
    # 寫入 requirements.txt
    requirements_path = Path("requirements.txt")
    with open(requirements_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(sorted(requirements_lines)) + '\n')
    
    print("✅ 依賴版本已同步")
    print(f"📦 共同步 {len(requirements_lines)} 個套件")
    
    # 顯示主要差異
    print("\n主要套件版本:")
    key_packages = ['transformers', 'torch', 'fastapi', 'streamlit', 'langchain']
    for package in key_packages:
        if package in dependencies:
            version = dependencies[package]
            if isinstance(version, str):
                print(f"  {package}: {version}")
            elif isinstance(version, dict) and 'version' in version:
                print(f"  {package}: {version['version']}")

if __name__ == "__main__":
    sync_dependencies()