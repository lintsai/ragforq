#!/usr/bin/env python
"""Pin selected core dependencies in pyproject.toml to the exact versions found in requirements.txt.

Workflow:
 1. Read requirements.txt locked versions.
 2. Load pyproject.toml dependencies.
 3. For packages in CORE_LIST present in both, replace caret/loose spec with exact version.
 4. Write updated pyproject.toml (in-place) and print a diff-like summary.

After running:
  poetry lock --no-update
  poetry export -f requirements.txt --output requirements.txt --without-hashes

Use cautiously: review git diff before commit.
"""
from __future__ import annotations
import re, sys
from pathlib import Path

# 解析 pyproject.toml：優先使用 tomllib / tomli；若皆不可用則使用簡單行級解析 fallback。
_TOML_AVAILABLE = True
try:  # Python 3.11+
    import tomllib  # type: ignore
except ModuleNotFoundError:  # Python 3.10 需要 tomli
    try:
        import tomli as tomllib  # type: ignore
    except ModuleNotFoundError:
        tomllib = None  # type: ignore
        _TOML_AVAILABLE = False

PYPROJECT = Path('pyproject.toml')
REQ = Path('requirements.txt')

CORE_LIST = [
    'fastapi','uvicorn','langchain','langchain-community','langchain-core','langchain-huggingface',
    'langchain-ollama','langchain-text-splitters','transformers','torch','sentence-transformers',
    'numpy','pydantic','faiss-cpu','streamlit'
]

if not (PYPROJECT.exists() and REQ.exists()):
    print('pyproject.toml 或 requirements.txt 缺失', file=sys.stderr)
    sys.exit(2)

req_versions: dict[str,str] = {}
pat = re.compile(r'^(?P<name>[A-Za-z0-9_.-]+)==(?P<ver>[^=]+)$')
for line in REQ.read_text(encoding='utf-8').splitlines():
    line=line.strip()
    if not line or line.startswith('#'): continue
    m = pat.match(line)
    if m:
        req_versions[m.group('name').lower()] = m.group('ver')

text = PYPROJECT.read_text(encoding='utf-8')
if _TOML_AVAILABLE:
    try:
        data = tomllib.loads(text)
        deps = data.get('tool',{}).get('poetry',{}).get('dependencies',{})
    except Exception:
        _TOML_AVAILABLE = False
        deps = {}
else:
    # Fallback: 只解析 [tool.poetry.dependencies] 區段內的簡單 key = "version" 行
    deps = {}
    in_dep = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith('[tool.poetry.dependencies]'):
            in_dep = True
            continue
        if in_dep and stripped.startswith('['):  # 下一節
            break
        if in_dep and '=' in stripped and not stripped.startswith('#'):
            # 只處理 key = "..." 或 key="..."
            m = re.match(r'([A-Za-z0-9_.-]+)\s*=\s*"([^"]+)"', stripped)
            if m:
                deps[m.group(1)] = m.group(2)

changes = []
for pkg in CORE_LIST:
    for key in list(deps.keys()):
        if key.lower() == pkg and pkg in req_versions:
            locked = req_versions[pkg]
            cur = deps[key]
            if isinstance(cur, dict):  # 高級格式（附加 extras / markers）
                cur_ver = cur.get('version','')
                if cur_ver and cur_ver != locked:
                    cur['version'] = locked
                    changes.append((pkg, cur_ver, locked))
            else:  # 字串格式
                if cur != locked and locked:
                    deps[key] = locked
                    changes.append((pkg, cur, locked))

if not changes:
    print('無需調整，核心依賴已與 requirements.txt 一致')
    sys.exit(0)

def replace_version_lines(original: str, pkg: str, new_version: str) -> str:
    # 支援格式：pkg = "^1.2.3" / pkg="1.2.3" / pkg = "~1.2" 等，全部替換為精確 "{new_version}"
    pattern = re.compile(rf'^(\s*{re.escape(pkg)}\s*=\s*)("[^"]+")', re.MULTILINE)
    return pattern.sub(lambda m: f"{m.group(1)}\"{new_version}\"", original)

new_text = text
for pkg, old, new in changes:
    new_text = replace_version_lines(new_text, pkg, new)

PYPROJECT.write_text(new_text, encoding='utf-8')
print('已更新下列核心依賴為精確鎖定版本:')
for pkg, old, new in changes:
    print(f" - {pkg}: {old} -> {new}")
print('\n接下來建議執行:')
print('  poetry lock --no-update')
print('  poetry export -f requirements.txt --output requirements.txt --without-hashes')
