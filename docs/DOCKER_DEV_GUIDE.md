# Docker 開發環境指南

## 🚀 快速啟動

### 使用管理腳本（推薦）

```bash
# GPU 版本（推薦）
python scripts/docker_dev.py start

# CPU 版本
python scripts/docker_dev.py start --cpu

# 查看日誌
python scripts/docker_dev.py logs

# 停止容器
python scripts/docker_dev.py stop
```

### 手動命令

```bash
# 1. 停止現有容器
docker stop ragforq-dev
docker rm ragforq-dev

# 2. 構建 GPU 版本鏡像
docker build --build-arg ENABLE_GPU=true -t ragforq-local-gpu .

# 3. 運行容器
docker run -d --rm --gpus all \
  -p 8000:8000 -p 8501:8501 \
  --name ragforq-dev \
  -v "$(pwd)/.env.local:/app/.env" \
  -v "$(pwd):/ragforq" \
  -v "$(pwd)/vector_db:/app/vector_db" \
  -v "$(pwd)/models:/app/models" \
  -v "$(pwd)/backups:/app/backups" \
  -v "$(pwd)/logs:/app/logs" \
  ragforq-local-gpu
```

## 📋 你的原始命令分析

你的命令基本正確，但有幾個優化建議：

### ✅ 正確的部分
- `--build-arg ENABLE_GPU=true` - 正確啟用 GPU
- `--gpus all` - 正確的 GPU 支援
- 端口映射 `-p 8501:8501 -p 8000:8000` - 正確
- 數據卷映射 - 基本正確

### 🔧 建議優化

1. **容器名稱**: 使用更簡潔的名稱
   ```bash
   # 原始
   --name ragforq-test-container
   # 建議
   --name ragforq-dev
   ```

2. **鏡像名稱**: 使用更清晰的標籤
   ```bash
   # 原始
   -t ragforq-local-test
   # 建議
   -t ragforq-local-gpu
   ```

3. **路徑使用**: Windows 路徑在 Docker 中的處理
   ```bash
   # 原始（Windows 絕對路徑）
   -v C:\Users\user\source\ragforq\.env.local:/app/.env
   # 建議（相對路徑，跨平台）
   -v "$(pwd)/.env.local:/app/.env"
   ```

4. **添加 `--rm`**: 自動清理停止的容器
   ```bash
   docker run --rm -d ...
   ```

## 🎯 推薦的完整流程

```bash
# 1. 使用管理腳本（最簡單）
python scripts/docker_dev.py start

# 2. 檢查狀態
python scripts/docker_dev.py status

# 3. 查看日誌
python scripts/docker_dev.py logs

# 4. 訪問服務
# 前端: http://localhost:8501
# API: http://localhost:8000
# API 文檔: http://localhost:8000/docs
```

## 🔍 故障排除

### GPU 支援檢查
```bash
# 檢查系統 GPU
python scripts/check_gpu.py

# 檢查 Docker GPU 支援
python scripts/build_docker.py --check-gpu
```

### 容器調試
```bash
# 進入容器
docker exec -it ragforq-dev bash

# 查看容器內 GPU 狀態
docker exec ragforq-dev python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### 日誌查看
```bash
# 實時日誌
python scripts/docker_dev.py logs -f

# 或直接使用 docker
docker logs -f ragforq-dev
```

## 📊 版本比較

| 特性 | 你的命令 | 推薦命令 |
|------|----------|----------|
| GPU 支援 | ✅ | ✅ |
| 端口映射 | ✅ | ✅ |
| 數據持久化 | ✅ | ✅ |
| 自動清理 | ❌ | ✅ |
| 跨平台路徑 | ❌ | ✅ |
| 錯誤處理 | ❌ | ✅ |
| 狀態檢查 | ❌ | ✅ |

你的原始命令是可以工作的，但使用管理腳本會更方便和可靠！