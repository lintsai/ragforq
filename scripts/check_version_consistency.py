#!/usr/bin/env python3
"""
檢查 pyproject.toml, requirements.txt, 和 Dockerfile 中的版本一致性
"""

import toml
import re
from pathlib import Path

def parse_requirements_txt():
    """解析 requirements.txt"""
    requirements = {}
    req_file = Path("requirements.txt")
    
    if req_file.exists():
        with open(req_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    # 處理 package[extras]>=version 格式
                    match = re.match(r'^([a-zA-Z0-9_-]+)(?:\[[^\]]+\])?(.*)', line)
                    if match:
                        package = match.group(1)
                        version = match.group(2)
                        requirements[package] = version
    
    return requirements

def parse_pyproject_toml():
    """解析 pyproject.toml"""
    dependencies = {}
    pyproject_path = Path("pyproject.toml")
    
    if pyproject_path.exists():
        with open(pyproject_path, 'r', encoding='utf-8') as f:
            pyproject_data = toml.load(f)
        
        deps = pyproject_data.get('tool', {}).get('poetry', {}).get('dependencies', {})
        
        for package, version in deps.items():
            if package == 'python':
                continue
                
            if isinstance(version, str):
                dependencies[package] = version
            elif isinstance(version, dict) and 'version' in version:
                dependencies[package] = version['version']
    
    return dependencies

def parse_dockerfile():
    """解析 Dockerfile 中的版本信息"""
    dockerfile_versions = {}
    dockerfile_path = Path("Dockerfile")
    
    if dockerfile_path.exists():
        with open(dockerfile_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 查找版本模式
        patterns = [
            (r'"torch>=([^"]+)"', 'torch'),
            (r'"transformers>=([^"]+)"', 'transformers'),
            (r'"faiss-cpu>=([^"]+)"', 'faiss-cpu'),
            (r'"faiss-gpu>=([^"]+)"', 'faiss-gpu'),
            (r'"numpy>=([^"]+),<2"', 'numpy'),
            (r'"bitsandbytes>=([^"]+)"', 'bitsandbytes'),
            (r'"vllm>=([^"]+)"', 'vllm'),
        ]
        
        for pattern, package in patterns:
            match = re.search(pattern, content)
            if match:
                dockerfile_versions[package] = f">={match.group(1)}"
    
    return dockerfile_versions

def normalize_version(version):
    """標準化版本格式"""
    if version.startswith('^'):
        return f">={version[1:]}"
    return version

def check_consistency():
    """檢查版本一致性"""
    print("🔍 檢查版本一致性...")
    print("=" * 60)
    
    # 解析各個文件
    pyproject_deps = parse_pyproject_toml()
    requirements_deps = parse_requirements_txt()
    dockerfile_deps = parse_dockerfile()
    
    # 重要套件列表
    key_packages = [
        'torch', 'transformers', 'fastapi', 'streamlit', 
        'langchain', 'sentence-transformers', 'numpy',
        'bitsandbytes', 'faiss-cpu'
    ]
    
    print("📦 主要套件版本對比:")
    print("-" * 60)
    print(f"{'套件名稱':<20} {'pyproject.toml':<15} {'requirements.txt':<15} {'Dockerfile':<15}")
    print("-" * 60)
    
    inconsistencies = []
    
    for package in key_packages:
        pyproject_ver = normalize_version(pyproject_deps.get(package, 'N/A'))
        requirements_ver = requirements_deps.get(package, 'N/A')
        dockerfile_ver = dockerfile_deps.get(package, 'N/A')
        
        print(f"{package:<20} {pyproject_ver:<15} {requirements_ver:<15} {dockerfile_ver:<15}")
        
        # 檢查不一致
        versions = [v for v in [pyproject_ver, requirements_ver, dockerfile_ver] if v != 'N/A']
        if len(set(versions)) > 1:
            inconsistencies.append(package)
    
    print("-" * 60)
    
    if inconsistencies:
        print(f"⚠️  發現 {len(inconsistencies)} 個套件版本不一致:")
        for package in inconsistencies:
            print(f"   - {package}")
        print("\n💡 建議運行 sync_dependencies.py 來同步版本")
    else:
        print("✅ 所有主要套件版本一致!")
    
    print(f"\n📊 統計:")
    print(f"   pyproject.toml: {len(pyproject_deps)} 個依賴")
    print(f"   requirements.txt: {len(requirements_deps)} 個依賴")
    print(f"   Dockerfile: {len(dockerfile_deps)} 個明確版本")

if __name__ == "__main__":
    check_consistency()