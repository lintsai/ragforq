# 版本管理指南

本專案使用 Poetry 作為主要的依賴管理工具，並確保 `pyproject.toml`、`requirements.txt` 和 `Dockerfile` 中的版本一致性。

## 📋 文件說明

### 主要配置文件
- **`pyproject.toml`** - Poetry 主配置文件，定義所有依賴版本
- **`requirements.txt`** - pip 兼容的依賴列表，從 pyproject.toml 同步生成
- **`Dockerfile`** - Docker 鏡像構建配置，包含關鍵套件的版本約束

### 輔助腳本
- **`scripts/sync_dependencies.py`** - 同步依賴版本腳本
- **`scripts/check_version_consistency.py`** - 版本一致性檢查腳本
- **`scripts/build_docker.bat`** - Windows Docker 構建腳本
- **`scripts/build_docker.sh`** - Linux/macOS Docker 構建腳本

## 🔄 版本同步流程

### 1. 更新依賴版本
```bash
# 使用 Poetry 更新依賴
poetry update

# 或添加新依賴
poetry add package_name@^1.0.0
```

### 2. 同步到 requirements.txt
```bash
# 運行同步腳本
python scripts/sync_dependencies.py
```

### 3. 檢查版本一致性
```bash
# 檢查所有文件的版本一致性
python scripts/check_version_consistency.py
```

### 4. 更新 Docker 配置
如果有新的關鍵依賴，需要手動更新 `Dockerfile` 中的版本約束。

## 📦 主要套件版本

| 套件名稱 | 當前版本 | 說明 |
|---------|---------|------|
| Python | 3.10.x | 基礎 Python 版本 |
| PyTorch | >=2.0.0 | 深度學習框架 |
| Transformers | >=4.35.0 | Hugging Face 模型庫 |
| FastAPI | >=0.111.0 | API 框架 |
| Streamlit | >=1.47.0 | Web UI 框架 |
| LangChain | >=0.3.26 | LLM 應用框架 |
| Sentence-Transformers | >=2.7.0 | 文本嵌入模型 |
| FAISS | >=1.8.0 | 向量搜索引擎 |
| BitsAndBytes | >=0.43.2 | 模型量化工具 |

## 🐳 Docker 構建

### CPU 版本
```bash
# Windows
scripts\build_docker.bat cpu latest

# Linux/macOS
./scripts/build_docker.sh cpu latest
```

### GPU 版本
```bash
# Windows
scripts\build_docker.bat gpu latest

# Linux/macOS
./scripts/build_docker.sh gpu latest
```

## ⚠️ 注意事項

### 版本約束原則
1. **主要依賴** - 使用 `^` 語法允許小版本更新
2. **關鍵依賴** - 使用 `>=` 語法確保最低版本要求
3. **問題依賴** - 使用精確版本或範圍約束

### 常見問題
1. **NumPy 版本衝突** - 限制 NumPy < 2.0 避免兼容性問題
2. **CUDA 版本** - GPU 版本需要對應的 CUDA 工具鏈
3. **記憶體限制** - 大型模型需要足夠的 GPU 記憶體

### 最佳實踐
1. 定期運行版本一致性檢查
2. 更新依賴後測試核心功能
3. 記錄重要的版本變更原因
4. 保持 Docker 鏡像的最新狀態

## 🔧 故障排除

### 依賴衝突
```bash
# 清理 Poetry 緩存
poetry cache clear --all pypi

# 重新安裝依賴
poetry install --no-cache
```

### Docker 構建失敗
```bash
# 清理 Docker 緩存
docker system prune -a

# 重新構建
docker build --no-cache --build-arg ENABLE_GPU=false -t ragforq:latest .
```

### 版本不一致
```bash
# 強制同步版本
python scripts/sync_dependencies.py
python scripts/check_version_consistency.py
```