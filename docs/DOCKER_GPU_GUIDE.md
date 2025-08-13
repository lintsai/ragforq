# Docker GPU 支援指南

## 🎯 概述

Dockerfile 現在支援根據構建參數自動選擇 CPU 或 GPU 版本的 PyTorch 和相關依賴。

## 🔧 構建選項

### CPU 版本（默認）
```bash
# 構建 CPU 版本
docker build -t ragforq-cpu .

# 或使用構建腳本
python scripts/build_docker.py --cpu
```

### GPU 版本
```bash
# 構建 GPU 版本
docker build --build-arg ENABLE_GPU=true -t ragforq-gpu .

# 或使用構建腳本
python scripts/build_docker.py --gpu
```

### 同時構建兩個版本
```bash
python scripts/build_docker.py --both
```

## 🚀 運行容器

### CPU 版本
```bash
docker run -p 8000:8000 -p 8501:8501 ragforq-cpu
```

### GPU 版本
```bash
# 需要 nvidia-docker 支援
docker run --gpus all -p 8000:8000 -p 8501:8501 ragforq-gpu
```

## 🔍 檢查 GPU 支援

### 檢查系統 GPU 支援
```bash
python scripts/build_docker.py --check-gpu
```

### 檢查本地 GPU 支援
```bash
python scripts/check_gpu.py
```

## 📋 版本差異

| 功能 | CPU 版本 | GPU 版本 |
|------|----------|----------|
| PyTorch | CPU 版本 | CUDA 12.1 版本 |
| FAISS | faiss-cpu | faiss-gpu |
| vLLM | ❌ 不安裝 | ✅ 安裝 |
| 推理速度 | 標準 | 2-10x 加速 |
| 記憶體需求 | 較低 | 較高 |
| 硬體需求 | 任何 CPU | NVIDIA GPU |

## ⚙️ 環境變數

容器啟動時會自動檢測 GPU 並顯示：
- PyTorch 版本
- CUDA 可用性
- GPU 數量和型號

## 🎯 推薦使用

- **開發/測試**: 使用 CPU 版本
- **生產環境**: 如果有 GPU，使用 GPU 版本
- **大型模型**: 強烈建議使用 GPU 版本

## 🔧 故障排除

### GPU 版本無法啟動
1. 確認安裝了 nvidia-docker
2. 確認 GPU 驅動正常
3. 檢查 CUDA 版本兼容性

### 構建失敗
1. 檢查網路連接
2. 確認 Docker 版本
3. 檢查磁盤空間

## 📚 相關文檔

- [NVIDIA Docker 安裝指南](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- [PyTorch CUDA 支援](https://pytorch.org/get-started/locally/)
- [vLLM 文檔](https://docs.vllm.ai/)