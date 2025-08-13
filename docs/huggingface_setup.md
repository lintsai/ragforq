# Hugging Face 平台設置指南

## 📋 概述

本指南將幫助您準備使用 Hugging Face 平台所需的環境和配置。

**重要說明**: Hugging Face 平台是完全本地化部署的解決方案，適合企業內部使用：
- ✅ **完全離線運行**: 模型下載後可完全離線使用
- ✅ **數據不外流**: 所有推理在本地進行，數據不會發送到外部服務
- ✅ **企業級隱私**: 符合企業數據安全要求
- ✅ **自主可控**: 完全掌控模型和數據處理流程

## 🔧 環境準備

### 1. 硬體需求

#### 最低需求
- **CPU**: 4核心以上
- **記憶體**: 16GB RAM
- **儲存**: 50GB 可用空間（用於模型緩存）

#### 推薦配置
- **CPU**: 8核心以上
- **記憶體**: 32GB RAM 或更多
- **GPU**: NVIDIA GPU（支援 CUDA）
  - RTX 3080/4080: 適合中型模型
  - RTX 3090/4090: 適合大型模型
  - A100/H100: 適合超大型模型
- **儲存**: 100GB+ SSD（用於模型緩存）

### 2. 軟體依賴

系統已包含所有必要的 Python 依賴包：

```bash
# 核心 Hugging Face 庫
transformers>=4.35.0
accelerate>=0.20.0
datasets>=2.14.0

# 深度學習框架
torch>=2.0.0
tensorflow>=2.13.0

# 高性能推理引擎
vllm>=0.2.0
ray>=2.8.0

# LangChain 集成
langchain-huggingface>=0.1.0
```

### 3. GPU 支援設置

#### CUDA 安裝（如果使用 NVIDIA GPU）

1. **檢查 GPU 支援**：
   ```bash
   nvidia-smi
   ```

2. **安裝 CUDA Toolkit**（如果尚未安裝）：
   - 訪問 [NVIDIA CUDA 下載頁面](https://developer.nvidia.com/cuda-downloads)
   - 選擇適合您系統的版本
   - 按照安裝指南進行安裝

3. **驗證 PyTorch GPU 支援**：
   ```python
   import torch
   print(f"CUDA 可用: {torch.cuda.is_available()}")
   print(f"CUDA 版本: {torch.version.cuda}")
   print(f"GPU 數量: {torch.cuda.device_count()}")
   ```

## 🚀 快速開始

### 1. 啟動系統

```bash
# 使用快速啟動腳本
python scripts/quick_start.py
```

### 2. 選擇 Hugging Face 平台

1. 打開瀏覽器訪問 http://localhost:8501
2. 在設置流程中選擇 "Hugging Face" 平台
3. 選擇適合的語言模型和嵌入模型

### 3. 推薦的模型組合

#### 輕量級配置（適合開發和測試）
- **語言模型**: `Qwen/Qwen2-0.5B-Instruct`
- **嵌入模型**: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- **記憶體需求**: ~4GB

#### 標準配置（適合生產環境）
- **語言模型**: `Qwen/Qwen2.5-0.5B-Instruct`
- **嵌入模型**: `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`
- **記憶體需求**: ~8GB

#### 高性能配置（適合大規模部署）
- **語言模型**: `openai/gpt-oss-20b`（生產，建議 vLLM）
- **嵌入模型**: `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`
- **記憶體需求**: ~16GB

## ⚙️ 進階配置

### 1. 推理引擎選擇

#### Transformers（默認）
- 適合：開發、測試、小規模部署
- 優點：穩定、兼容性好
- 缺點：性能較低

```bash
# 推理引擎現在通過前端設置流程選擇
# 在模型選擇步驟中會提供 Transformers 和 vLLM 選項
```

#### vLLM（推薦用於生產）
- 適合：生產環境、高並發
- 優點：高性能、低延遲
- 缺點：對硬體要求較高

```bash
# 推理引擎現在通過前端設置流程選擇
# vLLM 相關配置仍在 .env 文件中設置（作為默認值）
VLLM_GPU_MEMORY_UTILIZATION="0.9"
VLLM_MAX_MODEL_LEN="4096"
```

### 2. 模型緩存配置

```bash
# 設置模型緩存目錄
HF_MODEL_CACHE_DIR="./models/cache"

# 如果使用共享儲存
HF_MODEL_CACHE_DIR="/shared/models/cache"
```

### 3. Hugging Face Token（可選）

如果需要訪問私有模型或提高下載速度：

1. 註冊 [Hugging Face 帳號](https://huggingface.co/join)
2. 生成 [Access Token](https://huggingface.co/settings/tokens)
3. 在 `.env` 文件中設置：
   ```bash
   HF_TOKEN="your_token_here"
   ```

## 🔍 故障排除

### 常見問題

#### 1. 模型下載緩慢
**解決方案**：
- 使用 Hugging Face Token
- 設置代理（如果在企業網路環境）
- 選擇較小的模型進行測試

#### 2. GPU 記憶體不足
**解決方案**：
- 降低 `VLLM_GPU_MEMORY_UTILIZATION`
- 選擇較小的模型
- 使用 CPU 推理（設置 `TORCH_DEVICE="cpu"`）

#### 3. 模型載入失敗
**解決方案**：
- 檢查網路連接
- 清理模型緩存：`rm -rf ./models/cache/*`
- 檢查磁盤空間

### 診斷命令

```bash
# 檢查系統狀態
python tests/test_complete_system.py --check

# 檢查 GPU 狀態
nvidia-smi

# 檢查 Python 環境
python -c "import torch; print(torch.cuda.is_available())"
python -c "import transformers; print(transformers.__version__)"
```

## 📊 性能優化

### 1. 批處理大小調整

```bash
# 在 .env 文件中調整
EMBEDDING_BATCH_SIZE=16  # 增加批處理大小（如果記憶體充足）
FILE_BATCH_SIZE=50       # 增加文件批處理大小
```

### 2. 並行處理

```bash
# 設置工作線程數
MAX_WORKERS=8  # 根據 CPU 核心數調整
```

### 3. 記憶體優化

```bash
# PyTorch 設置
TORCH_DTYPE="float16"    # 使用半精度浮點數
TF_MEMORY_GROWTH="true"  # TensorFlow 記憶體動態增長
```

## 🔐 安全考量

### 1. 模型安全
- 只使用信任的模型來源
- 定期更新模型版本
- 監控模型輸出品質

### 2. 資料隱私
- 確保敏感資料不會被發送到外部服務
- 使用本地模型進行推理
- 定期清理模型緩存中的敏感資料

## 📚 參考資源

- [Hugging Face 官方文檔](https://huggingface.co/docs)
- [Transformers 庫文檔](https://huggingface.co/docs/transformers)
- [vLLM 文檔](https://docs.vllm.ai/)
- [CUDA 安裝指南](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)
