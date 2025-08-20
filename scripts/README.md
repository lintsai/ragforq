# Scripts 目錄說明

這個目錄包含了項目的各種工具腳本。

## 🔧 系統工具

### check_dependencies.py
**依賴檢查工具**
- 檢查所有必要的Python依賴是否正確安裝
- 驗證Python版本兼容性
- 提供安裝建議

```bash
python scripts/check_dependencies.py
```

### check_gpu.py
**GPU 支援檢查**
- 檢查 PyTorch GPU 支援狀況
- 驗證 CUDA 可用性
- 提供 GPU 配置資訊

```bash
python scripts/check_gpu.py
```

## 🔧 診斷工具

### rag_diagnostic_tool.py
**RAG 系統診斷和修復工具**
- 檢測和解決常見的 RAG 問題
- 系統健康狀況檢查
- 自動修復功能

```bash
python scripts/rag_diagnostic_tool.py
```

## 📚 索引管理腳本

### initial_indexing.py
**初始索引建立**
- 全新建立文檔索引
- 適用於首次部署或完全重建

```bash
python scripts/initial_indexing.py
```

### reindex.py
**重新索引**
- 完全重建現有索引
- 清除舊數據並重新開始

```bash
python scripts/reindex.py
```

## 📊 監控腳本

### monitor_indexing.py
**索引監控工具**
- 實時監控索引進度
- 查看系統狀態
- 重置進度（謹慎使用）

```bash
# 實時監控
python scripts/monitor_indexing.py --monitor

# 查看狀態
python scripts/monitor_indexing.py --status

# 查看進度
python scripts/monitor_indexing.py --progress

# 重置進度（謹慎）
python scripts/monitor_indexing.py --reset
```

### monitor_changes.py
**文件變更監控**
- 監控Q槽文件變更
- 自動觸發增量索引

```bash
# 每小時檢查一次
python scripts/monitor_changes.py --interval 3600
```

## 🤖 模型管理

### model_training_manager.py
**模型訓練管理器**
- 管理多模型訓練
- 支持初始訓練、增量訓練、重新索引

```bash
# 初始訓練
python scripts/model_training_manager.py initial \
  --ollama-model phi3:mini \
  --ollama-embedding-model nomic-embed-text

# 增量訓練
python scripts/model_training_manager.py incremental \
  --ollama-model phi3:mini \
  --ollama-embedding-model nomic-embed-text

# 重新索引
python scripts/model_training_manager.py reindex \
  --ollama-model phi3:mini \
  --ollama-embedding-model nomic-embed-text
```

## 使用建議

### 首次部署
1. `python scripts/check_dependencies.py` - 檢查依賴
2. `python scripts/initial_indexing.py` - 建立初始索引
3. `python scripts/monitor_indexing.py --monitor` - 監控進度

### 日常維護
1. `python scripts/monitor_changes.py --interval 3600` - 文件監控
2. `python scripts/monitor_indexing.py --status` - 檢查狀態

### 故障恢復
1. `python scripts/rag_diagnostic_tool.py` - 系統診斷和修復
2. `python scripts/monitor_indexing.py --progress` - 檢查進度

### 完全重建
1. `python scripts/reindex.py` - 重新索引
2. `python scripts/monitor_indexing.py --monitor` - 監控進度